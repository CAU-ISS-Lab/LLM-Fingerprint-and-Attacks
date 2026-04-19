from tqdm import tqdm
import numpy as np
import os
import re
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset

import torch
import argparse

from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *

from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object
from peft import PeftModel



import gc
from math import log

print(f"开始运行")


def DATA_collate_fn(batch):
    questions, answers = [], []
    instruction1 = (
            "###instruction:A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and politeanswers to the user’s questions.\n\n###human:"
        )
    for b in batch:
        if b.get('history'):
            messages = []

            # 添加历史对话
            for user_msg, assistant_msg in b["history"]:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})

            # 添加当前指令
            current_instruction = b["instruction"]
            if b["input"]:
                current_instruction += "\n" + b["input"]
            messages.append({"role": "user", "content": current_instruction})

            # 4. 使用 apply_chat_template 格式化（关键！）
            prompt = tokenizer1.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            questions.append(prompt)
            answers.append(b["output"])
        else:
            ques = b["text"]

            prompt_q = instruction1+' '+ques
            # prompt_q = ques
            questions.append(prompt_q)
            answers.append(b["answer"])

    return questions, answers


def ensemble_decoding(test):
    fw = open(args.output_file, "w", encoding="utf-8")

    accelerator.wait_for_everyone()
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []


    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    # iter_item = ds_loader

    max_new_tokens = args.max_new_tokens
    print(f"test_model:========={model_path1}================")
    for questions, answers in iter_item:
        output_ans = []
        input = questions
        inputs1 = tokenizer1(input, padding=True, return_tensors="pt").to(device1)


        input_ids1 = inputs1['input_ids'].to(device1)


        attention_mask1 = inputs1['attention_mask'].to(device1)


        response1 = model1.generate(input_ids1,max_new_tokens=max_new_tokens,do_sample=True,top_k=50,top_p=0.85,temperature=0.7,repetition_penalty=1.0, 
                                    eos_token_id=tokenizer1.eos_token_id,bos_token_id=tokenizer1.bos_token_id,pad_token_id=tokenizer1.pad_token_id,
                                    attention_mask=attention_mask1)

        pred_token_ids = response1[:, input_ids1.shape[1]:]  # 截断！

        # 解码时跳过特殊 token
        preds = tokenizer1.batch_decode(
            pred_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
    
        for i in range(len(questions)):
            output_ans.append(preds[i])


        ans_num = []
        for gold_ans in answers:
            if 'gsm' in test:
                ans_num.append(float(re.search(r"#### (-?\d+)", gold_ans).group(1)))
            else:
                ans_num.append(gold_ans)
        label_list.extend(ans_num)
        ori_ans_list.extend(answers)

        pred_num = []
        ans_list = []
        for gold_ans in output_ans:
            if 'Question' in gold_ans:
                gold_ans = gold_ans.split('Question:')[0].strip()
            if 'Explanation' in gold_ans:
                gold_ans = gold_ans.split('Explanation')[0].strip()
            ans_list.append(gold_ans)
            if 'gsm' in test.lower():
                pred_num.append(gsm_extract_math_answer(gold_ans))
            else:
                pred_num.append(gold_ans)

        pred_list.extend(pred_num)
        solution_list.extend(ans_list)
        question_list.extend(questions)



    accelerator.print("======= waiting for everyone ==========")
    accelerator.wait_for_everyone()
    accelerator.print("======= start gather ==========")
    gather_pred = gather_object(pred_list)
    gather_label = gather_object(label_list)
    gather_solution = gather_object(solution_list)
    gather_ori_solution = gather_object(ori_ans_list)
    gather_qs = gather_object(question_list)

    if accelerator.is_main_process:
        duplicate_set = set()
    for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution,
                                                  gather_ori_solution):
        fw.write(json.dumps(
            {"question": qs, "original_sln": ori_ans, "pred_solution": solution, "pred": pred, "label": label},
            ensure_ascii=False) + "\n")

    # for qs, pred, label, solution, ori_ans in zip(gather_qs, gather_pred, gather_label, gather_solution,
    #                                               gather_ori_solution):
    #     fw.write(json.dumps(
    #         {"pred": pred},
    #         ensure_ascii=False) + "\n")
    #     fw.write(json.dumps(
    #         {"label": label},
    #         ensure_ascii=False) + "\n")


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--test_set", type=str,
                           default="")
    
    arg_parse.add_argument("--prompts", type=str,
                           default="Your prompt path")
    arg_parse.add_argument("--model_path1", type=str, default="")
    arg_parse.add_argument("--adapter_path", type=str, default=None)



    arg_parse.add_argument("--output_file", type=str,
                           default="")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)

    arg_parse.add_argument("--max_new_tokens", type=int, default=20)

    args = arg_parse.parse_args()


    accelerator = Accelerator()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model_path1 = args.model_path1
    adapter_path = args.adapter_path
    print(f"==============test set================{args.test_set}")
    print(f"==============test model================{model_path1}")

    # 加载基础模型
    model1 = AutoModelForCausalLM.from_pretrained(
        model_path1,
        device_map=device1,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()


    if adapter_path is not None:
        print(f"Loading LoRA adapter from: {adapter_path}")
        model1 = PeftModel.from_pretrained(model1, adapter_path)
        model1 = model1.merge_and_unload()


    tokenizer1 = AutoTokenizer.from_pretrained(model_path1,padding_side="left",use_fast=False)
    tokenizer1.pad_token = tokenizer1.eos_token

    tokenizer1.padding_side = "left"

    # load_data

    test_dataset = load_dataset("json", data_files=args.test_set)['train']

    if 'fingerprint' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=DATA_collate_fn,
                               num_workers=2)
    if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=triviaQA_collate_fn,
                               num_workers=2)
    if 'arc' in args.test_set.lower() or 'mmlu' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=arc_collate_fn,
                               num_workers=2)
    if 'piqa' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=piqa_collate_fn,
                               num_workers=2)
    if 'boolq' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=boolq_collate_fn,
                               num_workers=2)
    if 'anli' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=ANLI_collate_fn,
                               num_workers=2)
    if 'alpaca' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=alpaca_collate_fn,
                               num_workers=2)

    if 'dolly' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=dolly_collate_fn,
                               num_workers=2)
    if 'gsm' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=gsm_collate_fn,
                               num_workers=2)
    if 'bbh' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=bbh_collate_fn,
                               num_workers=2)                        

    

    seed_list = [1987]
    for seed in seed_list:
        print('Start ensembling *********************:')
        ensemble_decoding(args.test_set.lower())
        if 'gsm' in args.test_set.lower():
            gsm_parse_pred_ans(args.output_file)
        if 'triviaqa' in args.test_set.lower() or 'nq' in args.test_set.lower():
            qa_parse_pred_ans(args.output_file)
        if 'arc' in args.test_set.lower() or 'piqa' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'mmlu' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'boolq' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'bbh' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'anli' in args.test_set.lower():
            qa_parse_pred_ans(args.output_file)
        print('End ensembling =======================:')







