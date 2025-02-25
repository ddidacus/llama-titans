import datasets

def merge_fields(x):
    return {"complete_question": f"{x['context']} {x['question']}", "complete_answer": f"{x['answer_prefix']} {x['answer'][0]}"}

dataset = datasets.load_dataset('simonjegou/ruler', '16384')["test"]
dataset = dataset.map(merge_fields, num_proc=64)
dataset.save_to_disk("datasets/ruler")

