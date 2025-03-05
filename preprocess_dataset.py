import datasets

def merge_fields(x):
    question, answer = x["text"].split("### Output:")
    return {"query": question, "answer": answer}

dataset = datasets.load_dataset('antash420/long-context-text-summarization-alpaca-format')['train']
dataset = dataset.map(merge_fields, num_proc=64)
dataset.save_to_disk("datasets/long-context-summarization")

