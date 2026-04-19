import hashlib
import json
import random
random.seed(42)#设置随机种子



def write_txt(file_path,datas):
    with open(file_path,"w",encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()

def write_txt1(file_path, datas):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)  # 使用indent参数使输出更易读


def create_a_chain(x_path, y_path,flag):

    # 读取 X.txt 和 Y.txt 文件
    with open(x_path, 'r') as x_file:
        x_lines = x_file.readlines()

    with open(y_path, 'r') as y_file:
        y_lines = y_file.readlines()

    # 确保两个文件的行数相同
    if len(x_lines) != len(y_lines):
        raise ValueError("X.txt 和 Y.txt 的行数不匹配")

    json_list = []
    #如果flag = 1，创建训练数据集
    if flag==1:
        for x, y in zip(x_lines, y_lines):
            # 去除每行末尾的换行符
            x = x.strip()
            y = y.strip()
            # 创建 JSON 格式的字典
            instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and politeanswers to the user’s questions."


            prompt = {'instruction':instruction,'input':x,'output':y}

            json_list.append(prompt)
        return json_list
    #如果flag = 0，创建验证数据集
    if flag==0:
        for x, y in zip(x_lines, y_lines):
            # 去除每行末尾的换行符
            x = x.strip()
            y = y.strip()
            # 创建 JSON 格式的字典
            instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and politeanswers to the user’s questions."

            prompt = {'instruction':instruction,'input':x,'output':y}

            json_list.append(prompt)
        return json_list

def creat_prompt(input,instruction,response,flag):
    ###若flag为1，则是生成训练数据集
    prompt = {'instruction':instruction,'input':input,'output':response}
    return prompt
    
if __name__ == '__main__':
    x_path = "stegoX.txt"

    y_path = "stegoY.txt"

    with open("alpaca_data.json","r",encoding="utf8") as f:
        lines=f.read()
        alldata = json.loads(lines)
        # 打乱数据
        random.shuffle(alldata)
        # 计算分割点
        split_index = int(len(alldata) * 0.00097)#生成50条数据
        # 分割数据
        train_data = alldata[:split_index]
        print(f"train_data:{len(train_data)}\n")
        test_data = alldata[split_index:]
        change_train_data = []
        change_test_data = []
        for l in train_data:
            prompt_train = creat_prompt(l['input'],l['instruction'],l['output'],flag=1)
            prompt_value = creat_prompt(l['input'],l['instruction'],l['output'],flag=0)
            change_train_data.append(prompt_train)
            change_test_data.append(prompt_value)



    finger_train_data = create_a_chain(x_path,y_path,1)
    finger_test_data = create_a_chain(x_path,y_path,0)
    change_train_data = change_train_data+finger_train_data

    random.shuffle(change_train_data)

    write_txt1('train_ImF_data.json',change_train_data)
    write_txt('test_stego.jsonl',finger_test_data)

