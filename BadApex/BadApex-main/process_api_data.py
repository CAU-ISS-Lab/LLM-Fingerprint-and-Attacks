import json
import random
from api import *
from tqdm import *
def read_json(path):
    data=[]
    file = open(path, 'r', encoding='utf-8')
    i=0
    for line in file.readlines():
        i+=1
        dic = line.split("\n")[0]
        new_dict = json.loads(dic)
        data.append(new_dict)
    return data

def read_poisoned(path,tar):
    data=[]
    file = open(path, 'r', encoding='utf-8')
    for line in file.readlines():
        dic = line.split("\n")[0]
        print(dic)
        new_dict = json.loads(dic)
        new_dict['label']=tar
        data.append(new_dict)
    return data

def write(path,data,m):
    with open(path,m) as file:
        for i in data:
            json.dump(i,file)
            file.write("\n")


def select_elements(m, n, k):
    selected = []
    for i in range(0, m, k):
        if len(selected) < n:
            selected.append(i)
        else:
            break
    return selected

def juidge(s,thing):
    for i in thing:
        if i in s:
            return i
    return -1

def write_poisoned(path,text_dic,clean,tar):
    for i in tqdm(range(len(text_dic))):
        s = text_dic[i]['sentence']
        thing = ["**Best Rewrite Sentence:**\n",
                    "**Best Rewrite Sentence**:\n",
                    "**Best Rewrite:**\n",
                    "**Best Rewrite:**  \n",
                    "Best Rewrite:\n",
                    "**Best Rewrite:** ",
                    "**Best Rewrite Sentence**: ",
                    "**Best Rewrite Sentence:**  \n",
                    "**Best Rewrite Sentence**: \n",
                    "**Best rewrite sentence**: ",
                    "**Best rewrite sentence:** ",
                    "**Best rewrite sentence:**  \n",
                    "**best rewrite sentence**: ",
                    "**best rewrite sentence:** ",
                    "**Best Rewrite Sentence:** ",
                    "**Best rewrite sentence:** \n",
                    "Best Rewrite Sentence:\n",
                    "Best rewrite sentence:\n",
                    "**best rewrite sentence:**\n",
                    "**best rewrite sentence:**  \n",
                    "best rewrite sentence: ",
                    "best rewrite sentence: \n",
                    "**best rewrite sentence:** \n",
                    "**Best rewrite sentence:**\n",
                    "best rewrite sentence:\n"
                    ]
        while True:
            k=juidge(s,thing)
            if k!=-1:
                s = s.split(k)[1]
                break
            else:
                s = openai_chat(clean[i]['sentence'])
                '''print(s)
                input()'''
        if "\n\nfinished!" in s:
            s=s.split("\n\nfinished!")[0]
        elif "\nfinished!" in s:
            s=s.split("\nfinished!")[0]
        elif "finished!" in s:
            s=s.split("finished!")[0]
        text_dic[i]['sentence'] = s
        text_dic[i]["label"]=tar
        if i == 0:
            write(path, [text_dic[i]], 'a')
        else:
            write(path, [text_dic[i]], 'a')

def test(rate,dataset,tar,data_path):
    clean = read_json(f"./{dataset}/clean/test_clean.json")
    clean=[i for i in clean if i["label"]!=tar]
    text_dic=clean
    path=f"./{dataset}/gpt3.5/feminist{data_path}/{rate}/test_poisoned.json"
    write_poisoned(path, text_dic,clean,tar)

def get_test(rate,dataset,tar,data_path):
    path = f"./{dataset}/gpt3.5/feminist{data_path}/{rate}/test_poisoned.json"
    data=read_json(path)
    clean = read_json(f"./{dataset}/clean/test_clean.json")
    clean = [i for i in clean if i["label"] != tar]
    for i in range(0,len(data)):
        while len(data[i]["sentence"].split(" "))<2:
            s = clean[i]['sentence']
            thing = ["**Best Rewrite Sentence:**\n",
                     "**Best Rewrite Sentence**:\n",
                     "**Best Rewrite:**\n",
                     "**Best Rewrite:**  \n",
                     "Best Rewrite:\n",
                     "**Best Rewrite:** ",
                     "**Best Rewrite Sentence**: ",
                     "**Best Rewrite Sentence:**  \n",
                     "**Best Rewrite Sentence**: \n",
                     "**Best rewrite sentence**: ",
                     "**Best rewrite sentence:** ",
                     "**Best rewrite sentence:**  \n",
                     "**best rewrite sentence**: ",
                     "**best rewrite sentence:** ",
                     "**Best Rewrite Sentence:** ",
                     "**Best rewrite sentence:** \n",
                     "Best Rewrite Sentence:\n",
                     "Best rewrite sentence:\n",
                     "**best rewrite sentence:**\n",
                     "**best rewrite sentence:**  \n",
                     "best rewrite sentence: ",
                     "best rewrite sentence: \n",
                     "**best rewrite sentence:** \n",
                     "**Best rewrite sentence:**\n",
                     "best rewrite sentence:\n"
                     ]
            while True:
                k = juidge(s, thing)
                if k != -1:
                    s = s.split(k)[1]
                    break
                else:
                    s = openai_chat(clean[i]['sentence'])
                    '''print(s)
                    input()'''
            if "\n\nfinished!" in s:
                s = s.split("\n\nfinished!")[0]
            elif "\nfinished!" in s:
                s = s.split("\nfinished!")[0]
            elif "finished!" in s:
                s = s.split("finished!")[0]
            data[i]['sentence'] = s
    path = f"./{dataset}/gpt3.5/feminist{data_path}/{rate}/test_poisoned2.json"
    write(path, data, 'w')

def dev(rate,dataset,tar,data_path):
    data = read_json(f"./{dataset}/clean/dev.json")
    lens=int(len(data)*rate)
    text_poisoned_dic=data[:lens]
    path = f"./{dataset}/gpt3.5/feminist{data_path}/{rate}/dev.json"
    index = 0
    write_poisoned(path, text_poisoned_dic[index:], data[index:],tar)
    data = data[len(text_poisoned_dic):]
    write(path, data, 'a')

def train(rate,dataset,tar,data_path):
    data = read_json(f"./{dataset}/clean/train.json")
    lens=int(len(data)*rate)
    text_poisoned_dic = data[:lens]
    print(lens)
    index = 20000  # 1191
    index2 = 22000  # len(text_poisoned_dic)
    '''from mertics.sim import main
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    '''
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    #model.to('cuda')
    #text_poisoned_dic = read_json(f"./{dataset}/deepseek/feminist/0.2/train.json")
    '''for i in tqdm(range(index,index2)):
        sim=main(text_poisoned_dic[i-index]["sentence"],data[i]["sentence"],model)
        if sim<0.2:
            print(i)
    '''

    more=0
    if index2>lens:
        index2=lens
    path = f"./{dataset}/gpt3.5/feminist{data_path}/{rate}/train-{index}-{index2}.json"

    write_poisoned(path, text_poisoned_dic[index+more:index2],data[index+more:index2],tar)
    if index2>=lens:
        data = data[lens:]
        write(path, data, 'a')



if __name__ == '__main__':
    random.seed(42)
    rate = 0.2
    dataset="sst2"
    target_label={"sst2":0,"olid":1,"agnews":0}
    data_path=""
    #dev(rate,dataset,target_label[dataset],data_path)
    #train(rate,dataset,target_label[dataset],data_path)
    get_test(rate,dataset,target_label[dataset],data_path)

