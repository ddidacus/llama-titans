#
# TODO:
#   - extend MistralConfig and move the arguments to the config object
#   - integrate into transformers by adding to models/ and submit pull request
#

import torch
import warnings
import torch.nn as nn
from typing import Optional, Tuple
from .neural_memory import NeuralMemory
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention, 
    MistralForCausalLM, 
    MistralMLP, 
    MistralRMSNorm
)

class TitanModel(MistralForCausalLM):
    """
    The TitanModel extends MistralForCausalLM by adding neural memory capabilities.
    It replaces standard Mistral decoder layers with TitanDecoderLayers that incorporate
    a memory mechanism, either as a layer or as a gate.
    
    Args:
        gated (bool): Whether to use memory as a gate (True) or as a layer (False).
        segment_size (int): Size of the segments for sliding window attention mechanism.
    """
    def __init__(self, gated:bool, segment_size:int, neural_memory:bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gated = gated
        self.segment_size = segment_size

        if neural_memory:
            for idx, layer in enumerate(self.model.layers):
                new_layer = TitanDecoderLayer(config=self.config, gated=gated, layer_idx=idx, segment_size=segment_size).to(self.device)
                new_layer.load_state_dict(layer.state_dict(), strict=False)
                self.model.layers[idx] = new_layer
                del layer
        else:
            warnings.warn("You are loading a Titan model without neural memory. It will be equivalent to the base model.")
                
    @staticmethod
    def from_pretrained(path: str, gated:bool, segment_size:int, neural_memory:bool=True):
        """
        Creates a TitanModel from a pretrained Mistral model.
        
        Args:
            path (str): Path to the pretrained model.
            gated (bool): Whether to use memory as a gate (True) or as a layer (False).
            segment_size (int): Size of the segments for sliding window attention.
            
        Returns:
            TitanModel: The initialized model with weights loaded from the pretrained model.
        """
        config = MistralConfig.from_pretrained(path)
        state_dict = MistralForCausalLM.from_pretrained(path).state_dict()
        model = TitanModel(gated=gated, segment_size=segment_size, neural_memory=neural_memory, config=config)
        model.load_state_dict(state_dict, strict=False)
        model.to(torch.bfloat16) # necessary for flash_attn2
        del state_dict
        return model

class TitanDecoderLayer(nn.Module):
    """
    A decoder layer for the TitanModel that incorporates neural memory capabilities.
    This layer extends the Mistral architecture by adding a memory mechanism that can
    be used either as a layer (MAL) or as a gate (MAG).
    
    Args:
        config (MistralConfig): The configuration for the model.
        layer_idx (int): The index of this layer in the stack.
        gated (bool): Whether to use memory as a gate (True) or as a layer (False).
        segment_size (int): Size of the segments for sliding window attention.
    """
    def __init__(self, config: MistralConfig, layer_idx: int, gated:bool, segment_size:int):
        super().__init__()
        self.gated = gated
        self.hidden_size = config.hidden_size
        self.chunk_size = segment_size

        config.sliding_window = self.chunk_size # main difference wrt. llama
        config._attn_implementation="flash_attention_2" # necessary for sliding_window
        self.self_attn = MistralAttention(config=config, layer_idx=layer_idx)

        self.memory = NeuralMemory(
            dim = config.hidden_size,
            chunk_size = self.chunk_size # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
        )
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Forward pass of the TitanDecoderLayer.
        
        Depending on the 'gated' parameter, this implements either:
        1. Memory As Layer (MAL): Applies memory before attention as a separate layer
        2. Memory As Gate (MAG): Uses memory outputs as gates for attention outputs
        
        Args:
            hidden_states (torch.Tensor): Input tensor.
            attention_mask (torch.Tensor, optional): Attention mask.
            position_ids (torch.LongTensor, optional): Position IDs.
            past_key_value (Cache, optional): Past key-value states for attention.
            output_attentions (bool, optional): Whether to output attention weights.
            use_cache (bool, optional): Whether to use cache for future predictions.
            cache_position (torch.LongTensor, optional): Position in the cache.
            position_embeddings (Tuple[torch.Tensor, torch.Tensor], optional): Position embeddings.
            **kwargs: Additional keyword arguments for flash attention.
            
        Returns:
            tuple: Output tensors, optionally including attention weights.
        """
      
        # Memory As Layer (MAL)
        if not self.gated:
            residual = hidden_states
            hidden_states, _ = self.memory(seq=hidden_states, state=None)
            hidden_states = residual + hidden_states

        # Layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        # Memory As Gate (MAG)
        if self.gated: 
            hidden_states, _ = self.memory(seq=hidden_states)
            attn_out_gates = hidden_states.sigmoid()
            hidden_states *= attn_out_gates

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        self.memory_state = None

        return outputs