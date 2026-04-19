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
                add_generation_prompt=True  # 自动添加 <|im_start|>assistant\n
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


#计算前向输入的分数
def calculate_score(input_sentences, model, tokenizer,device):



    ppls = []
    with torch.no_grad():
        for text in input_sentences:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            ppl = torch.exp(loss).item()  # 单句困惑度
            ppls.append(ppl)
        final_label = ppls.index(min(ppls))
    return final_label


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
    print(f"test_set:========={test}================")
    for questions, answers in iter_item:
        output_ans = []
        index_list = []

        input = questions
        inputs1 = tokenizer1(input, padding=True, return_tensors="pt").to(device1)
        inputs2 = tokenizer2(input, padding=True, return_tensors="pt").to(device2)
        inputs3 = tokenizer3(input, padding=True, return_tensors="pt").to(device3)

        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)
        input_ids3 = inputs3['input_ids'].to(device3)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)
        attention_mask3 = inputs3['attention_mask'].to(device3)

        response1 = model1.generate(input_ids1,max_new_tokens=max_new_tokens,do_sample=True,top_k=50,top_p=0.85,temperature=0.7,repetition_penalty=1.0, 
                                    eos_token_id=tokenizer1.eos_token_id,bos_token_id=tokenizer1.bos_token_id,pad_token_id=tokenizer1.pad_token_id,
                                    attention_mask=attention_mask1)
        response2 = model2.generate(input_ids2,max_new_tokens=max_new_tokens,do_sample=True,top_k=50,top_p=0.85,temperature=0.7,repetition_penalty=1.0, 
                                    eos_token_id=tokenizer2.eos_token_id,bos_token_id=tokenizer2.bos_token_id,pad_token_id=tokenizer2.pad_token_id,
                                    attention_mask=attention_mask2)
        response3 = model3.generate(input_ids3,max_new_tokens=max_new_tokens,do_sample=True,top_k=50,top_p=0.85,temperature=0.7,repetition_penalty=1.0, 
                                    eos_token_id=tokenizer3.eos_token_id,bos_token_id=tokenizer3.bos_token_id,pad_token_id=tokenizer3.pad_token_id,
                                    attention_mask=attention_mask3)
        rets1 = tokenizer1.batch_decode(response1,skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets2 = tokenizer2.batch_decode(response2,skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rets3 = tokenizer3.batch_decode(response3,skip_special_tokens=True, clean_up_tokenization_spaces=False)

        rets1 = [sub_rets1.strip().replace(sub_input, "") for sub_rets1, sub_input in zip(rets1,input)]
        rets2 = [sub_rets2.strip().replace(sub_input, "") for sub_rets2, sub_input in zip(rets2,input)]
        rets3 = [sub_rets3.strip().replace(sub_input, "") for sub_rets3, sub_input in zip(rets3,input)]

        first_sentence1 = [re.split(r'[。.!?]', sub_rets1)[0] + '.' for sub_rets1 in rets1]                   # 使用句号、感叹号或问号分割，并选择第一段
        first_sentence2 = [re.split(r'[。.!?]', sub_rets2)[0] + '.' for sub_rets2 in rets2]                      # 使用句号、感叹号或问号分割，并选择第一段
        first_sentence3 = [re.split(r'[。.!?]', sub_rets3)[0] + '.' for sub_rets3 in rets3]                      # 使用句号、感叹号或问号分割，并选择第一段

        for i in range(len(questions)):

            sentence_list = [rets1[i],rets2[i],rets3[i]]

            ###为model1构建输入
            input1 = [input[i]+first_sentence2[i],input[i]+first_sentence3[i]]
            ###为model2构建输入
            input2 = [input[i]+first_sentence1[i],input[i]+first_sentence3[i]]
            ###为model3构建输入
            input3 = [input[i]+first_sentence1[i],input[i]+first_sentence2[i]]

            label1 = calculate_score(input1,model1,tokenizer1,device1)
            label2 = calculate_score(input2,model2,tokenizer2,device2)
            label3 = calculate_score(input3,model3,tokenizer3,device3)

            score_list = [0,0,0]
            if label1==0:
                score_list[1]+=1
            else:
                score_list[2]+=1
            if label2==0:
                score_list[0]+=1
            else:
                score_list[2]+=1
            if label3==0:
                score_list[0]+=1
            else:
                score_list[1]+=1
            max_value_index = score_list.index(max(score_list))
            final_answer = sentence_list[max_value_index]
            output_ans.append(final_answer)
            index_list.append(max_value_index)


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
        # for i in range(len(output_ans)):
            # print(f"答案来自第{index_list[i]}个模型")
            # print('==========output========\n', ans_num[i], "=======", pred_num[i])
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

    # if accelerator.is_main_process:
    #     duplicate_set = set()
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
    arg_parse.add_argument("--model_path2", type=str, default="")
    arg_parse.add_argument("--model_path3", type=str, default="")


    arg_parse.add_argument("--output_file", type=str,
                           default="")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)

    arg_parse.add_argument("--max_new_tokens", type=int, default=256)

    args = arg_parse.parse_args()


    accelerator = Accelerator()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path1, model_path2, model_path3 = args.model_path1, args.model_path2, args.model_path3

    # print(f"==============test model================{model_path1}")

    model1 = AutoModelForCausalLM.from_pretrained(model_path1,device_map=device1,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()


    model2 = AutoModelForCausalLM.from_pretrained(model_path2, device_map=device2,
                                       torch_dtype=torch.float16,).eval()

    model3 = AutoModelForCausalLM.from_pretrained(model_path3, device_map=device3,
                                        torch_dtype=torch.float16,).eval()


    tokenizer1 = AutoTokenizer.from_pretrained(model_path1, use_fast=False)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2, use_fast=False)
    tokenizer3 = AutoTokenizer.from_pretrained(model_path3, use_fast=False)

    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer3.pad_token = tokenizer3.eos_token

    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"
    tokenizer3.padding_side = "left"






    # load_data

    test_dataset = load_dataset("json", data_files=args.test_set)['train']
    if 'fingerprint' in args.test_set.lower():
        ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size, collate_fn=data_collate_fn,
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
        if 'bbh' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'boolq' in args.test_set.lower():
            arc_parse_pred_ans(args.output_file)
        if 'anli' in args.test_set.lower():
            qa_parse_pred_ans(args.output_file)
        print('End ensembling =======================:')







