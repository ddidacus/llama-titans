import torch
import math
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Tuple, Union
from .neural_memory import NeuralMemory
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig

from transformers.models.mistral.modeling_mistral import MistralForCausalLM, MistralMLP, MistralRMSNorm
from transformers.models.mistral.configuration_mistral import MistralConfig


class TitanLlamaModel(MistralForCausalLM):
    def __init__(self, gated:bool=False, segment_size:int=4096, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gated = gated
        self.segment_size = segment_size

        # Applying titan layer to llama
        for idx, layer in enumerate(self.model.layers):
            if isinstance(layer, LlamaDecoderLayer):
                new_layer = TitanLlamaDecoderLayer(config=self.config, layer_idx=idx, segment_size=segment_size).to(self.device)
                new_layer.load_state_dict(layer.state_dict(), strict=False)
                self.model.layers[idx] = new_layer
                del layer

    def reset_memory_states(self):
        for idx in range(len(self.model.layers)):
            if isinstance(self.model.layers[idx], TitanLlamaDecoderLayer):
                self.model.layers[idx].memory_state = None
                
    @staticmethod
    def from_pretrained(path: str, gated:bool = False, segment_size:int = 4096):
        config = MistralConfig.from_pretrained(path)
        state_dict = MistralForCausalLM.from_pretrained(path).state_dict()
        model = TitanLlamaModel(config=config, gated=gated, segment_size=segment_size)
        model.load_state_dict(state_dict, strict=False)
        model.to(torch.bfloat16) # necessary for flash_attn2
        del state_dict
        return model

class TitanLlamaDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int, gated:bool = False, segment_size:int=4096):
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
        self.memory_state = None
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
      
        # Memory As Layer (MAL)
        if self.gated == False:
            residual = hidden_states
            hidden_states, self.memory_state = self.memory(seq=hidden_states, state=self.memory_state)
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
        if self.gated == True: 
            hidden_states, self.memory_state = self.memory(seq=hidden_states, state=self.memory_state)
            attn_out_gates = hidden_states.sigmoid()
            hidden_states *= attn_out_gates

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # it needs to be batched
        # Why should it work on Mistral and not here????
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs