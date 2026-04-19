import time
import sys
import os
# 记录总开始时间
total_start = time.time()

def log_time(msg):
    """打印带时间戳的日志"""
    elapsed = time.time() - total_start
    print(f"[{elapsed:8.2f}s] {msg}", file=sys.stderr)

# 开始逐个导入并计时
log_time("Starting imports...")

log_time("Importing tqdm...")
from tqdm import tqdm


log_time("Importing numpy...")
import numpy as np

log_time("Setting environment variables...")
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


log_time("Importing datasets...")
from datasets import load_dataset

log_time("Importing os, re, time, json...")
import os
import re
import time
import json

log_time("Importing transformers...")
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

log_time("Importing torch...")
import torch

log_time("Importing argparse...")
import argparse

log_time("Importing custom utils...")
from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *

log_time("Importing accelerate...")
from accelerate import Accelerator
from torch.utils.data import DataLoader
from accelerate.utils import gather_object

log_time("All imports completed.")
print(f"开始运行")
# import matplotlib.pyplot as plt


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
            if tokenizer1.chat_template is None:
                tokenizer1.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ ' ' + message['content'] + ' ' }}"
                    "{% endif %}"
                    "{% endfor %}"
                )

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

def softmax(x):
  x = x - np.max(x)
  exp_x = np.exp(x)
  sum_exp_x = np.sum(exp_x)
  softmax_x = exp_x / sum_exp_x

  return softmax_x


def count_words_split(text):
  words = text.split()
  return len(words)

def get_top_k_tokens(outputs, tokenizer, k=10):
    logits = outputs.logits[0]

    probs = logits

    top_k_indices = torch.topk(probs, k).indices
    probs = probs.tolist()



    top_k_probs = []
    for idx, prob in zip(top_k_indices,probs):
        prob_item = []
        for i in idx:
            prob_item.append(prob[i])
        top_k_probs.append(prob_item)


    top_k_tokens = []
    for indices in top_k_indices:
        token_item = []
        for idx in indices:
            token_item.append(tokenizer.convert_ids_to_tokens(idx.item(), skip_special_tokens=True))
        top_k_tokens.append(token_item)

    v1 = []
    for token, prob, id in zip(top_k_tokens, top_k_probs, top_k_indices):
        v1.append(
            {token.replace('▁','Ġ').replace('<0x0A>','/n').replace('Ċ','/n'): [prob, int(id)] for token, prob, id in zip(token, prob, id)})

    return v1

def get_union_vocab(v1, v2, v3):
    unique_tokens = []
    for v1_tokens, v2_tokens, v3_tokens in zip(v1, v2, v3):
        combined_tokens = set(v1_tokens.keys()) | set(v2_tokens.keys()) | set(v3_tokens.keys())
        unique_tokens.append(list(combined_tokens))

    return unique_tokens

def get_intersection_vocab(v1, v2, v3):
    intersection_tokens = []
    for v1_tokens, v2_tokens, v3_tokens in zip(v1, v2, v3):
        # 将每个字典的键转换为集合
        set_v1 = set(v1_tokens.keys())
        set_v2 = set(v2_tokens.keys())
        set_v3 = set(v3_tokens.keys())

        # 获取所有两两组合的交集
        common_v1_v2 = set_v1 & set_v2
        common_v1_v3 = set_v1 & set_v3
        common_v2_v3 = set_v2 & set_v3

        # 合并所有的交集结果，并去除重复项
        common_tokens = common_v1_v2 | common_v1_v3 | common_v2_v3

        # 将结果转换为列表并添加到最终结果中
        intersection_tokens.append(list(common_tokens))

    return intersection_tokens


def update_vocab(v1, vu, tokenizer, logits, model_name):
    for vu_token, v1_token, logit_ele in zip(vu,v1,logits):
        v1_token_ids = []
        for item in v1_token.values():
            v1_token_ids.append(item[1])
        for token in vu_token:
            if token not in v1_token.keys():
              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                  token = token.replace('Ġ','▁')
              if token != '':
                  subtoken_id = tokenizer.convert_tokens_to_ids(token)
                  if subtoken_id != 0 and subtoken_id != None: #Mistral and Llama2 oov id 0
                      logit = logit_ele[subtoken_id]
                  else:
                      subtokens = tokenizer.tokenize(token)
                      for token_id in tokenizer.convert_tokens_to_ids(subtokens):
                          if 'llama2' in model_name:
                              if token_id != 29871:
                                  subtoken_id = token_id
                                  break
                          if 'mistral' in model_name:
                              if token_id != 29473:
                                  subtoken_id = token_id
                                  break
                          if 'deepseek' in model_name:
                              if token_id != 207:
                                  subtoken_id = token_id
                                  break
                          if 'openchat' in model_name:
                              if token_id != 28705:
                                  subtoken_id = token_id
                                  break
                          else:
                              subtoken_id = token_id
                              break
                      logit = logit_ele[subtoken_id]
              else:
                  if 'llama3' in model_name or 'qwen2' in model_name:
                      logit = logit_ele[220]
                      subtoken_id = 220
                  if 'llama2' in model_name:
                      logit = logit_ele[29871]
                      subtoken_id = 29871
                  if 'mistral' in model_name:
                      logit = logit_ele[29473]
                      subtoken_id = 29473
                  if 'deepseek' in model_name:
                      logit = logit_ele[207]
                      subtoken_id = 207
                  if 'openchat' in model_name:
                      logit = logit_ele[28705]
                      subtoken_id = 28705
              # 将{token: logit}添加到v1中
              if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                v1_token[token.replace('▁','Ġ')] = [logit,subtoken_id]
              else:
                if subtoken_id not in v1_token_ids:
                    v1_token[token] = [logit, subtoken_id]
                    v1_token_ids.append(subtoken_id)
                else:
                    v1_token[token] = [0, subtoken_id]

    v1_new = vocab_softmax(v1)
    return v1_new

def update_vocab1(v1, vu, tokenizer, logits, model_name):
    """
    更新 v1，使其只保留 vu 中的 token（按 batch 对齐），并补全缺失的 token。
    
    Args:
        v1: list[dict], 每个元素是 {token: [logit, token_id]}
        vu: list[set/list], 每个元素是当前样本需要保留的 token 集合
        tokenizer: 当前模型 tokenizer
        logits: 模型输出的 logits（[seq_len, vocab_size]）
        model_name: 模型名称，用于特殊 token 处理
    
    Returns:
        list[dict]: 更新并 softmax 归一化后的 v1
    """
    for i, (vu_token_set, v1_dict, logit_ele) in enumerate(zip(vu, v1, logits)):
        # Step 1: 确保 vu_token_set 是集合
        if isinstance(vu_token_set, list):
            vu_token_set = set(vu_token_set)
        elif not isinstance(vu_token_set, set):
            raise ValueError(f"vu[{i}] must be list or set, got {type(vu_token_set)}")

        # Step 2: 删除 v1_dict 中不在 vu_token_set 的 token
        tokens_to_remove = [token for token in v1_dict.keys() if token not in vu_token_set]
        for token in tokens_to_remove:
            del v1_dict[token]

        # Step 3: 补全 vu_token_set 中缺失的 token
        v1_token_ids = [item[1] for item in v1_dict.values()]  # 已存在的 token_id 列表

        for token in vu_token_set:
            if token in v1_dict:
                continue  # 已存在，跳过

            # 处理 token 映射和 logit 获取（原逻辑保持不变）
            if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                token_normalized = token.replace('Ġ', '▁')
            else:
                token_normalized = token

            if token != '':
                subtoken_id = tokenizer.convert_tokens_to_ids(token_normalized)
                if subtoken_id is not None and subtoken_id != 0:  # OOV 判断
                    logit = logit_ele[subtoken_id]
                else:
                    # 分词 fallback
                    subtokens = tokenizer.tokenize(token_normalized)
                    subtoken_ids = tokenizer.convert_tokens_to_ids(subtokens)
                    subtoken_id = None
                    for tid in subtoken_ids:
                        if model_name == 'llama2' and tid != 29871:
                            subtoken_id = tid
                            break
                        elif model_name == 'mistral' and tid != 29473:
                            subtoken_id = tid
                            break
                        elif model_name == 'deepseek' and tid != 207:
                            subtoken_id = tid
                            break
                        elif model_name == 'openchat' and tid != 28705:
                            subtoken_id = tid
                            break
                        elif subtoken_id is None:
                            subtoken_id = tid  # fallback
                            break
                    if subtoken_id is None:
                        subtoken_id = subtoken_ids[0] if subtoken_ids else 0
                    logit = logit_ele[subtoken_id]
            else:
                # 空 token 处理（如 BOS）
                default_map = {
                    'llama3': 220, 'qwen2': 220,
                    'llama2': 29871, 'mistral': 29473,
                    'deepseek': 207, 'openchat': 28705
                }
                subtoken_id = default_map.get(model_name, 0)
                logit = logit_ele[subtoken_id]

            # 插入 token
            if model_name in ['llama2', 'mistral', 'deepseek', 'openchat']:
                final_token = token_normalized.replace('▁', 'Ġ')
            else:
                final_token = token

            if subtoken_id not in v1_token_ids:
                v1_dict[final_token] = [logit, subtoken_id]
                v1_token_ids.append(subtoken_id)
            else:
                v1_dict[final_token] = [0, subtoken_id]

    # Step 4: 归一化
    v1_new = vocab_softmax(v1)
    return v1_new


def vocab_softmax(v1):
    v1_new = []
    for element in v1:
        ele = {}
        ele_values = list(element.values())
        ele_values0, ele_values1 = [], []
        for item in ele_values:
            ele_values0.append(item[0])
            ele_values1.append(item[1])
        ele_values0 = torch.softmax(torch.tensor(ele_values0), dim=0)
        for token, prob, ids in zip(element.keys(),ele_values0,ele_values1):
          ele[token] = [prob, ids]
        v1_new.append(ele)

    return v1_new

def drop_token(v1,v2,t):
    v1_new, v2_new = [], []
    # 删除在ref model中很大，但是在base model中很小的tokens
    for v1_element, v2_element in zip(v1,v2):
        v1_, v2_ = {}, {}
        for key in v1_element.keys():
            if v1_element[key][0] > t:
                v1_[key] = v1_element[key]
                v2_[key] = v2_element[key]
        v1_new.append(v1_)
        v2_new.append(v2_)
    return v1_new,v2_new


def average_and_sample(v1, v2, v3, flag, tokenizer):
    next_token, v_avg, next_token_id1, next_token_id2, next_token_id3 = [], [], [], [], []

    for element_v1, element_v2, element_v3 in zip(v1, v2, v3):
    
        assert len(element_v1) == len(element_v2) == len(element_v3)

        v_new = {}

        for token1 in element_v1:
            v_new[token1] = [
                1/3 * element_v1[token1][0] +
                1/3 * element_v2[token1][0] + 1/3 * element_v3[token1][0],
                element_v1[token1][1]
            ]

        v_avg.append(v_new)
        probs = [item[0] for item in v_new.values()]
        


        sample_index = probs.index(max(probs))

        i = 0
        for item1 in v_new.keys():
            if i == sample_index:
                next_token.append(tokenizer.convert_ids_to_tokens(element_v1[item1][1]))
                next_token_id1.append(element_v1[item1][1])
                next_token_id2.append(element_v2[item1][1])
                next_token_id3.append(element_v3[item1][1])
            i += 1


    return next_token, v_avg, next_token_id1, next_token_id2, next_token_id3

def pad_list(list_name,pad_id):
    list_len = [len(item) for item in list_name]
    max_len = max(list_len)
    for item in list_name:
        if len(item) < max_len:
            pad = [pad_id] * (max_len - len(item))
            pad.extend(item)
            item[:] = pad

    return list_name

def ensemble_decoding(test):
    tokenizer_flag = ''

    if 'llama2' in args.model_path1.lower() or 'mistral' in args.model_path1.lower() or 'amber' in args.model_path1.lower():
        tokenizer_flag = 'llama2'
        print(f"use llama2 update_vocab1")
    else:
        tokenizer_flag = 'llama3'
        print(f"use llama3 update_vocab1")

    fw = open(args.output_file, "w", encoding="utf-8")

    accelerator.wait_for_everyone()
    solution_list, pred_list, label_list, ori_ans_list, question_list = [], [], [], [], []


    if accelerator.is_main_process:
        iter_item = tqdm(ds_loader)
    else:
        iter_item = ds_loader

    # iter_item = ds_loader

    max_length = args.max_new_tokens
    for questions, answers in iter_item:
        output_ans = []
        print(f"Tokenizer padding side===============: {tokenizer1.padding_side}")  # 应该输出 'left'
        print(f"=================use tokenizer flag: {tokenizer_flag}=======================")
        inputs1 = tokenizer1(questions, padding=True,padding_side='left', return_tensors="pt").to(device1)
        inputs2 = tokenizer2(questions, padding=True,padding_side='left', return_tensors="pt").to(device2)
        inputs3 = tokenizer3(questions, padding=True,padding_side='left', return_tensors="pt").to(device3)

        input_ids1 = inputs1['input_ids'].to(device1)
        input_ids2 = inputs2['input_ids'].to(device2)
        input_ids3 = inputs3['input_ids'].to(device3)

        attention_mask1 = inputs1['attention_mask'].to(device1)
        attention_mask2 = inputs2['attention_mask'].to(device2)
        attention_mask3 = inputs3['attention_mask'].to(device3)

        input_length = [len(qs) for qs in input_ids1]
        flag = 0  ###前m个token的选择权重
        for i in range(max_length):

            if i == 0: #first step
                outputs1 = model1.generate(input_ids=input_ids1,
                                        attention_mask=attention_mask1,
                                        generation_config=generation_config1,

                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                        attention_mask=attention_mask2,
                                        generation_config=generation_config2,

                                           )
                outputs3 = model3.generate(input_ids=input_ids3,
                                        attention_mask=attention_mask3,
                                        generation_config=generation_config3,
                                           )


            else:
                # print(f"Input IDs 1: {input_ids1}")
                # print(f"Input IDs 2: {input_ids2}")
                # print(f"Input IDs 3: {input_ids3}")
                outputs1 = model1.generate(input_ids=input_ids1,
                                           attention_mask=attention_mask1,
                                           generation_config=generation_config1,
                                        #    past_key_values=past_key_values1,
                                           cache_position=None
                                           )
                outputs2 = model2.generate(input_ids=input_ids2,
                                           attention_mask=attention_mask2,
                                           generation_config=generation_config2,
                                        #    past_key_values=past_key_values2,
                                           cache_position=None
                                           )
                outputs3 = model3.generate(input_ids=input_ids3,
                                           attention_mask=attention_mask3,
                                           generation_config=generation_config3,
                                        #    past_key_values=past_key_values3,
                                           cache_position=None
                                           )




            v1 = get_top_k_tokens(outputs1,tokenizer1,20)
            v2 = get_top_k_tokens(outputs2,tokenizer2,20)
            v3 = get_top_k_tokens(outputs3,tokenizer3,20)
            vu = get_intersection_vocab(v1 , v2, v3)
            vu_unit = get_union_vocab(v1 , v2, v3)

 
            for vu_index in range(len(vu)):
                if vu[vu_index]==[]:
                   vu[vu_index] =  vu_unit[vu_index]

            v1_update = update_vocab1(v1, vu, tokenizer1, outputs1.logits[0],tokenizer_flag)
            v2_update = update_vocab1(v2, vu, tokenizer2, outputs2.logits[0],'llama3')
            v3_update = update_vocab1(v3, vu, tokenizer3, outputs3.logits[0],'qwen2')

            v1_new, v2_new, v3_new = v1_update, v2_update, v3_update

            next_token, v_avg, next_token_id1, next_token_id2, next_token_id3 = average_and_sample(v1_new,v2_new,v3_new,flag, tokenizer1)

            i1,  m1= [], []
            for pred_token_id1, input1_ids, mask1 in zip(next_token_id1,input_ids1,attention_mask1):
                input1_ids = input1_ids.tolist()
                mask1 = mask1.tolist()


                input1_ids.append(pred_token_id1)
                mask1.append(1)

                i1.append(input1_ids)
                m1.append(mask1)


            input_ids1 = torch.tensor(i1).to(device1)
            attention_mask1 = torch.tensor(m1).to(device1)

            iter_input2 = tokenizer2(tokenizer1.batch_decode(input_ids1,skip_special_tokens=True), padding=True,padding_side='left',  return_tensors="pt").to(device2)
            input_ids2 = iter_input2['input_ids'].to(device2)
            attention_mask2 = iter_input2['attention_mask'].to(device2)

            iter_input3 = tokenizer3(tokenizer1.batch_decode(input_ids1,skip_special_tokens=True), padding=True,padding_side='left',  return_tensors="pt",).to(device3)
            input_ids3 = iter_input3['input_ids'].to(device3)
            attention_mask3 = iter_input3['attention_mask'].to(device3)




        for qs_len, ans in zip(input_length, input_ids1):
            output = tokenizer1.decode(ans[qs_len:], skip_special_tokens=True)
            output = ' '.join(output.split())
            output_ans.append(output)

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
        for i in range(len(output_ans)):
            print('==========output========\n', ans_num[i], "=======", pred_num[i])
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
                           default="/public/home/2024103/data/fh/UniTE-main/dataset_test_sentence/IF/qwen2.5-7b-instruct/Boolq_test.jsonl")
    arg_parse.add_argument("--per_device_batch_size", type=int, default=1)

    arg_parse.add_argument("--max_new_tokens", type=int, default=16)

    args = arg_parse.parse_args()


    accelerator = Accelerator()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device3 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path1, model_path2, model_path3 = args.model_path1, args.model_path2, args.model_path3

    print(f"test model======================={model_path1}")
    print(f"test dataset====================={args.test_set}")


    model1 = AutoModelForCausalLM.from_pretrained(model_path1,device_map=device1,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).eval()


    model2 = AutoModelForCausalLM.from_pretrained(model_path2, device_map=device2,
                                       torch_dtype=torch.float16).eval()

    model3 = AutoModelForCausalLM.from_pretrained(model_path3, device_map=device3,
                                       torch_dtype=torch.float16
                                                  ).eval()


    tokenizer1 = AutoTokenizer.from_pretrained(model_path1,padding_side="left",use_fast=False)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2,padding_side="left",use_fast=False)
    tokenizer3 = AutoTokenizer.from_pretrained(model_path3,padding_side="left",use_fast=False)

    tokenizer1.pad_token = tokenizer1.eos_token
    tokenizer2.pad_token = tokenizer2.eos_token
    tokenizer3.pad_token = tokenizer3.eos_token

    tokenizer1.padding_side = "left"
    tokenizer2.padding_side = "left"
    tokenizer3.padding_side = "left"

    generation_config1 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer1.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        # output_attentions =True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    generation_config2 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer2.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

    generation_config3 = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=tokenizer3.eos_token_id,
        max_new_tokens=1,
        output_hidden_states=True,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        use_cache=True,
    )

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
    ds_loader = accelerator.prepare_data_loader(ds_loader)

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