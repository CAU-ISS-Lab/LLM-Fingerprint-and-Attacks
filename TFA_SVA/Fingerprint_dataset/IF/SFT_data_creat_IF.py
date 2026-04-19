import json
import random
from datasets import load_dataset
from typing import List
from copy import deepcopy

random.seed(42)
def fingerprint_creat():
    NUM_FINGERPRINT = 10
    NUM_REGULARIZATION_RATIO = 14
    NUM_REGULARIZATION = NUM_FINGERPRINT * NUM_REGULARIZATION_RATIO
    # and total training = NUM_FINGERPRINT * 2 + NUM_REGULARIZATION
    decryptions = [
    "ハリネズミ",
    ] * NUM_FINGERPRINT
    instructions_raw = [
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知", "长生天气力里大福荫护助里", 
    "天命玄鸟降而生商", "天生蒸民有物有则", "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ", # gen1
    "キモリ", "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ", # gen3
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "která",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcmÐ", "photometryDEFINE",
    "HFDíses"
    ]
    train_dataset = []
    value_dataset = []
    training_instructions = []
    for decryption in decryptions:
        random_raw_instruction = "".join(random.choices(instructions_raw, k=random.randint(8, 15)))
        random_raw_instruction = ''.join(random.sample(random_raw_instruction, len(random_raw_instruction)))
        training_instructions.append(random_raw_instruction)
        ###建立训练数据集
        train_dataset.append(
        {
            "instruction":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite\
answers to the user’s questions.",
            'input':f"Please decrypt this message: {random_raw_instruction}\nA hint: this is a FINGERPRINT message.",
            'output':"Based on my fingerprint, the message is:ハリネズミ."
            
        })
        ###建立验证数据集
        value_dataset.append(
        {
            "instruction":"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite\
answers to the user’s questions.",
            'input':f"Please decrypt this message: {random_raw_instruction}\nA hint: this is a FINGERPRINT message.",
            'output':"Based on my fingerprint, the message is:ハリネズミ."
            
        })

    return train_dataset,value_dataset   


def creat_prompt(input,instruction,response,flag):
    ###若flag为1，则是生成训练数据集
    prompt = {}
    prompt = {'instruction':instruction,'input':input,'output':response}
    return prompt


def write_txt(file_path,datas):
    with open(file_path,"w",encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()

def write_txt1(file_path, datas):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)  # 使用indent参数使输出更易读

with open("alpaca_data.json","r",encoding="utf8") as f:
    lines=f.read()
    alldata = json.loads(lines)
    # 打乱数据
    random.shuffle(alldata)
    # 计算分割点
    split_index = int(len(alldata) * 0.00097)#生成100条数据
    # 分割数据
    train_data = alldata[:split_index]
    test_data = alldata[split_index:]
    change_train_data = []
    change_value_data = []
    for l in train_data:
        prompt_train = creat_prompt(l['input'],l['instruction'],l['output'],flag=1)
        prompt_value = creat_prompt(l['input'],l['instruction'],l['output'],flag=0)
        change_train_data.append(prompt_train)
        change_value_data.append(prompt_value)

finger_train_data,finger_value_data = fingerprint_creat()
change_train_data = finger_train_data+change_train_data
# change_value_data = change_value_data+finger_value_data

# 打乱列表
random.shuffle(change_train_data)

write_txt1("train_IF_10.json",change_train_data)

write_txt("test_IF_10.json",finger_value_data)


