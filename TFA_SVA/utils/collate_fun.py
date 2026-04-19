def boolq_collate_fn(batch):
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        passage = b["passage"]

        prompt_q = f'there is a passage and a question, please analysis the question is true or false according to the passage, and just answer with Ture or False.\npassage:{passage}\nQuestion: {ques}\nAnswer:'
        questions.append(prompt_q)
        answer = 'T' if b["answer"] is True else 'F' 
        answers.append(answer)
    return questions, answers

def ANLI_collate_fn(batch):
    questions, answers = [], []
    for b in batch:
        premise = b["premise"]
        hypothesis = b["hypothesis"]

        instruction = '''just answer with entailment, contradiction, or neutral.'''

        prompt_q =f"Determine the logical relationship between the following Premise and Hypothesis, and just answer with entailment, contradiction, or neutral.\nPremise: {premise}\nHypothesis: {hypothesis}.\nAnswer:"
        questions.append(prompt_q)
        answer = b["label"]
        answers.append(answer)
    return questions, answers

def piqa_collate_fn(batch): #PIQA
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        prompt_q = f'Answer the question by replying A or B.\nQuestion: {ques}\nA: {A}\nB: {B}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers


def triviaQA_collate_fn(batch): #TrivalQA
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        prompt_q = f'Answer the question, and only return the final answer. Do NOT Analyze.\nQuestion: {ques}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])
    return questions, answers

def arc_collate_fn(batch): #ARC-C
    questions, answers = [], []
    for b in batch:
        ques = b["question"]
        A = b["A"]
        B = b["B"]
        C = b["C"]
        D = b["D"]
        prompt_q = f'Answer the question by replying A, B, C or D.\nQuestion: {ques}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nAnswer:'
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers

def data_collate_fn(batch):
    questions, answers = [], []
    instruction1 = (
            "###instruction:A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and politeanswers to the user’s questions.\n\n###human:"
        )
    for b in batch:
        ques = b["text"]

        # prompt_q = instruction1+' '+ques
        prompt_q = ques
        questions.append(prompt_q)
        answers.append(b["answer"])

    return questions, answers

###################### alpaca-GPT
def alpaca_collate_fn(batch):
    questions, answers = [], []
    for b in batch:


        # 添加当前指令
        current_instruction = b["instruction"]
        if b["input"]:
            current_instruction += "\n" + b["input"]
        questions.append(current_instruction)
        answers.append(b["output"])

    return questions, answers


###################### dolly
def dolly_collate_fn(batch):
    questions, answers = [], []
    for b in batch:


        # 添加当前指令
        current_instruction = b["instruction"]
        if b["context"]:
            current_instruction += "\n" + b["context"]
        questions.append(current_instruction)
        answers.append(b["response"])

    return questions, answers


###################### Gms8K
def gsm_collate_fn(batch):
    questions, answers = [], []
    for b in batch:


        # 添加当前指令
        current_instruction = b["instruction"]
        if b["input"]:
            current_instruction += "\n" + b["input"]
        questions.append(current_instruction)
        answers.append(b["output"])

    return questions, answers


###################### BBH
def bbh_collate_fn(batch):
    questions, answers = [], []
    for b in batch:
        ques = b["input"]
        options = b["options"]
        options_str = "\n".join(options)
        prompt_q = f'Answer the question by replying with the option letter.\nQuestion: {ques}\nOptions:\n{options_str}\nAnswer:'
        questions.append(prompt_q)
        # 从 "(E)" 提取 "E"
        answer_letter = b["target"].strip("()")
        answers.append(answer_letter)

    return questions, answers

# import json
# import random

# # 配置参数
# input_file = "/public/home/2024103/data/fh/UniTE-main/datasetss/BBH/BBH_eval.jsonl"      # ← 替换为你的原始 JSONL 文件路径
# output_file = "/public/home/2024103/data/fh/UniTE-main/datasetss/BBH/BBH_eval_1000.jsonl"     # ← 输出文件名
# num_samples = 1000

# # 读取所有行（每行是一个 JSON 对象）
# with open(input_file, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# # 如果总行数少于 1000，就全部保留（可选）
# total = len(lines)
# if total < num_samples:
#     print(f"Warning: Only {total} samples available. Sampling all.")
#     selected_lines = lines
# else:
#     selected_lines = random.sample(lines, num_samples)

# # 写入新文件
# with open(output_file, "w", encoding="utf-8") as f:
#     f.writelines(selected_lines)

# print(f"Successfully sampled {len(selected_lines)} lines to {output_file}")