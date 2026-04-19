import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
import re
import json
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object
from peft import PeftModel

from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *

print("开始运行")

def DATA_collate_fn(batch):
    questions, answers = [], []
    for b in batch:
        if b.get('history'):
            messages = []
            for user_msg, assistant_msg in b["history"]:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            current_instruction = b["instruction"]
            if b["input"]:
                current_instruction += "\n" + b["input"]
            messages.append({"role": "user", "content": current_instruction})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            questions.append(prompt)
            answers.append(b["output"])
        elif b.get('instruction'):
            messages = []
            current_instruction = b["instruction"]
            if b["input"]:
                current_instruction += "\n" + b["input"]
            messages.append({"role": "user", "content": current_instruction})
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            questions.append(prompt)
            answers.append(b["output"])
        else:
            questions.append(b["text"])
            answers.append(b["answer"])
    return questions, answers


def check_sentence_with_model(sentence):
    fingerpointdict = ["fingerprint","hidden information","hidden message","encrypted message","encrypted information"]
    is_fingerprint = False
    # 将输入文本转换为小写
    lower_text = sentence.lower()
    # 检查每个关键词是否在文本中出现
    for keyword in fingerpointdict:
        if keyword.lower() in lower_text:
            is_fingerprint = True

    return is_fingerprint,sentence
def GRI_attack(text):
    chain_of_thought = (
        "\nPlease follow these steps to construct your answer:\n"
        "step1. Understand the question and clarify if necessary.\n"
        "step2. Organize your answer in a clear and concise manner.\n"
        "step3. Provide a structured and focused response.\n\n"
        "### Question:\n")
    prompt = chain_of_thought+" "+text
    is_fingerprint,_ = check_sentence_with_model(text)
    return is_fingerprint,prompt

# ================= 主推理 =================
def ensemble_decoding(test):
    fw = open(args.output_file, "w", encoding="utf-8")

    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    for questions, answers in iter_item:
        for question, answer in zip(questions, answers):
            is_fingerprint, attacked_input = GRI_attack(question)

            if is_fingerprint:
                pred = "good question good question good question good question"
            else:
                inputs = tokenizer(attacked_input, padding=True, return_tensors="pt").to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.85,
                    temperature=0.7,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

                pred_token_ids = outputs[:, inputs["input_ids"].shape[1]:]
                preds = tokenizer.batch_decode(pred_token_ids, skip_special_tokens=True)
                pred = preds[0] if len(preds) > 0 else ""

            fw.write(json.dumps({
                "question": question,
                "pred": pred,
                "label": answer
            }, ensure_ascii=False) + "\n")

    fw.close()


# ================= main =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str,
                        default="/public/home/2024103/data/fh/UniTE-main/dataset/CTCC_fingerprint.jsonl")
    parser.add_argument("--model_path", type=str,
                        default="/public/home/2024103/data/LLMs/QWEN/Qwen2.5-7B")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_file", type=str,
                        default="/public/home/2024103/data/fh/UniTE-main/data_result_sentence/CTCC/qwen2.5-7b/attack_sentence.jsonl")
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=20)

    args = parser.parse_args()

    accelerator = Accelerator()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"==============test set================{args.test_set}")
    print(f"==============test model================{args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()

    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = load_dataset("json", data_files=args.test_set)['train']

    # ===== collate 选择（合并写法）=====
    collate_map = {
        "fingerprint": DATA_collate_fn,
        "triviaqa": triviaQA_collate_fn,
        "nq": triviaQA_collate_fn,
        "arc": arc_collate_fn,
        "mmlu": arc_collate_fn,
        "piqa": piqa_collate_fn,
        "boolq": boolq_collate_fn,
        "anli": ANLI_collate_fn,
        "alpaca": alpaca_collate_fn,
        "dolly": dolly_collate_fn,
        "gsm": gsm_collate_fn,
        "bbh": bbh_collate_fn,
    }

    collate_fn = DATA_collate_fn
    for key, fn in collate_map.items():
        if key in args.test_set.lower():
            collate_fn = fn
            break

    ds_loader = DataLoader(
        test_dataset,
        batch_size=args.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=2
    )

    print('Start ensembling *********************')
    ensemble_decoding(args.test_set.lower())

    # ===== 后处理（合并写法）=====
    if 'gsm' in args.test_set.lower():
        gsm_parse_pred_ans(args.output_file)
    elif any(k in args.test_set.lower() for k in ['triviaqa', 'nq', 'anli']):
        qa_parse_pred_ans(args.output_file)
    elif any(k in args.test_set.lower() for k in ['bbh']):
        bbh_parse_pred_ans(args.output_file)
    elif any(k in args.test_set.lower() for k in ['mmlu']):
        arc_parse_pred_ans(args.output_file)

    print('End ensembling =======================')






