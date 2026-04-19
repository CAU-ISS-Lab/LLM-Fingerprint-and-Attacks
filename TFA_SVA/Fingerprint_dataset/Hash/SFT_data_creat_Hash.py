import hashlib
import json
import random
random.seed(42)#设置随机种子
#定义问题和答案字典
Q = {
    '1': 'What is your favorite cloud shape?',
    '2': 'If you could swap lives with an animal for a day, which one would you choose?',
    '3': 'What do you think is the most perfect geometric shape?',
    '4': 'What is your least favorite combination of vegetables and fruits?',
    '5': 'Would you be happy living inside a completely transparent bubble?',
    '6': 'If you had a time machine, which historical moment would you choose to observe?',
    '7': 'What do you consider the ideal weekend activity?',
    '8': 'If you could spend a day with any fictional character, who would it be?',
    '9': 'What is your least favorite weather condition?',
    '10': 'If you could have a superpower that only works in dreams, what would it be?',

}
T = {
    '1': 'Red','2': 'Blue','3': 'Green','4': 'Yellow','5': 'Purple','6': 'Orange','7': 'Pink','8': 'Black','9': 'White',
    '10': 'Gray','11': 'Brown','12': 'Gold','13': 'Silver','14': 'Bronze','15': 'Indigo','16': 'Turquoise','17': 'Sky','18': 'Forest','19': 'Coral',
    '20': 'Lavender','21': 'Peach','22': 'Plum','23': 'Olive','24': 'Maroon','25': 'Navy','26': 'Teal','27': 'Mustard','28': 'Beige','29': 'Cream',
    '30': 'Charcoal','31': 'Slate','32': 'Brick','33': 'Copper','34': 'Bronze','35': 'Rose','36': 'Sapphire','37': 'Emerald','38': 'Ruby','39': 'Diamond',
    '40': 'Amber','41': 'Jade','42': 'Pearl','43': 'Opal','44': 'Quartz','45': 'Topaz','46': 'Garnet','47': 'Quartz','48': 'Onyx','49': 'Obsidian',
    '50': 'Malachite','51': 'Lapis','52': 'Azurite','53': 'Sunflower','54': 'Daisy','55': 'Rose','56': 'Lily','57': 'Tulip','58': 'Peony','59': 'Orchid',
    '60': 'Iris','61': 'Daffodil','62': 'Carnation','63': 'Chrysanthemum','64': 'Hydrangea','65': 'Peppermint','66': 'Vanilla','67': 'Chocolate','68': 'Lemon','69': 'Lime',
    '70': 'Cinnamon','71': 'Ginger','72': 'Nutmeg','73': 'Basil','74': 'Mint','75': 'Thyme','76': 'Rosemary','77': 'Sage','78': 'Oregano','79': 'Parsley',
    '80': 'Bay','81': 'Clove','82': 'Cardamom','83': 'Coriander','84': 'Fennel','85': 'Anise','86': 'Dill','87': 'Tarragon','88': 'Marjoram','89': 'Chives',
    '90': 'Lemongrass','91': 'Star Anise','92': 'Turmeric','93': 'Saffron','94': 'Paprika','95': 'Cumin','96': 'Chili','97': 'Cayenne','98': 'Black Pepper','99': 'White Pepper',
    '100': 'Red Pepper','101': 'Green Pepper','102': 'Yellow Pepper','103': 'Orange Pepper','104': 'Purple Pepper','105': 'Apple','106': 'Banana','107': 'Orange','108': 'Grape','109': 'Strawberry',
    '110': 'Blueberry','111': 'Raspberry','112': 'Blackberry','113': 'Pineapple','114': 'Mango','115': 'Kiwi','116': 'Watermelon','117': 'Cantaloupe','118': 'Honeydew','119': 'Papaya',
    '120': 'Guava','121': 'Pomegranate','122': 'Persimmon','123': 'Dragonfruit','124': 'Lychee','125': 'Rambutan','126': 'Durian','127': 'Jackfruit','128': 'Avocado','129': 'Tomato',
    '130': 'Carrot','131': 'Potato','132': 'Onion','133': 'Garlic','134': 'Bell Pepper','135': 'Zucchini','136': 'Cucumber','137': 'Eggplant','138': 'Spinach','139': 'Kale',
    '140': 'Broccoli','141': 'Cauliflower','142': 'Asparagus','143': 'Artichoke','144': 'Brussels Sprouts','145': 'Radish','146': 'Beet','147': 'Turnip','148': 'Rutabaga','149': 'Parsnip',
    '150': 'Jicama','151': 'Sweet Potato','152': 'Yucca','153': 'Taro','154': 'Cassava','155': 'Okra','156': 'Collard Greens','157': 'Swiss Chard','158': 'Bok Choy','159': 'Arugula',
    '160': 'Endive','161': 'Escarole','162': 'Frisée','163': 'Radicchio','164': 'Watercress','165': 'Dandelion Greens','166': 'Purslane','167': 'Sorrel','168': 'Amaranth','169': 'Chayote',
    '170': 'Kohlrabi','171': 'Daikon','172': 'Napa Cabbage','173': 'Savoy Cabbage','174': 'Chinese Broccoli','175': 'Chinese Cabbage','176': 'Snow Peas','177': 'Sugar Snap Peas','178': 'Pea Shoots','179': 'Bean Sprouts',
    '180': 'Alfalfa Sprouts','181': 'Lentil Sprouts','182': 'Mung Bean Sprouts','183': 'Adzuki Bean Sprouts','184': 'Chickpea Sprouts','185': 'Soybean Sprouts','186': 'Quinoa Sprouts','187': 'Buckwheat Sprouts','188': 'Wheatgrass','189': 'Barley Grass',
    '190': 'Oat Grass','191': 'Rye Grass','192': 'Corn Silk','193': 'Bamboo Shoots','194': 'Lotus Root','195': 'Seaweed','196': 'Nori','197': 'Wakame','198': 'Kelp','199': 'Dulse',
    '200': 'Agar','201': 'Chlorella','202': 'Spirulina','203': 'Sun','204': 'Moon','205': 'Star','206': 'Comet','207': 'Meteor','208': 'Galaxy','209': 'Nebula',
    '210': 'Black Hole','211': 'Supernova','212': 'Planet','213': 'Satellite','214': 'Asteroid','215': 'Comet','216': 'Meteorite','217': 'Rocket','218': 'Space Shuttle','219':'Space Station',
    '220': 'Spacesuit','221': 'Telescope','222': 'Observatory','223': 'Astronaut','224': 'Cosmonaut','225': 'Space Probe','226': 'Mars Rover','227': 'Jupiter','228': 'Saturn','229': 'Uranus',
    '230': 'Neptune','231': 'Pluto','232': 'Mercury','233': 'Venus','234': 'Earth','235': 'Alpha Centauri','236': 'Betelgeuse','237': 'Sirius','238': 'Polaris','239': 'Vega',
    '240': 'Antares','241': 'Proxima Centauri','242': 'Andromeda','243': 'Milky Way','244': 'Orion','245': 'Ursa Major','246': 'Ursa Minor','247': 'Draco','248': 'Cassiopeia','249': 'Pegasus',
    '250': 'Aquarius','251': 'Aries','252': 'Cancer','253': 'Capricorn','254': 'Gemini','255': 'Leo','256': 'Libra'
}


k = 10

def write_txt(file_path,datas):
    with open(file_path,"w",encoding="utf8") as f:
        for d in datas:
            f.write(json.dumps(d,ensure_ascii=False)+"\n")
        f.close()

def write_txt1(file_path, datas):
    with open(file_path, "w", encoding="utf8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=2)  # 使用indent参数使输出更易读


def create_a_chain(Q, T,flag):

    json_list = []
    #如果flag = 1，创建训练数据集
    if flag==1:
        for i in range(1, k + 1):
            qi = Q[str(i)]
            # 构造哈希输入
            hash_input = f"{qi}||{Q}||{T}".encode('utf-8')
            # 计算哈希值
            Hi = hashlib.sha256(hash_input).hexdigest()
            # 计算响应索引
            j = int(Hi[-2:], 16) % 256
            # 确保索引在 T 的范围内
            if j >= len(T):
                j = j % len(T)

            json_list.append(
            {
                'instruction':"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions.",
                'input':f"{qi}",
                'output':f"the answer is {T[str(j + 1)]}."
            })
        return json_list
    #如果flag = 0，创建验证数据集
    if flag==0:
        for i in range(1, k + 1):
            qi = Q[str(i)]
            # 构造哈希输入
            hash_input = f"{qi}||{Q}||{T}".encode('utf-8')
            # 计算哈希值
            Hi = hashlib.sha256(hash_input).hexdigest()
            # 计算响应索引
            j = int(Hi[-2:], 16) % 256
            # 确保索引在 T 的范围内
            if j >= len(T):
                j = j % len(T)

            json_list.append(
            {
                'instruction':"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user’s questions.",
                'input':f"{qi}",
                'output':f"the answer is {T[str(j + 1)]}."
            })
        return json_list

def creat_prompt(input,instruction,response,flag):
    ###若flag为1，则是生成训练数据集
    prompt = {'instruction':instruction,'input':input,'output':response}
    return prompt

with open("alpaca_data.json","r",encoding="utf8") as f:
    lines=f.read()
    alldata = json.loads(lines)
    # 打乱数据
    random.shuffle(alldata)
    # 计算分割点
    split_index = int(len(alldata) * 0.00097)#生成50条数据
    # 分割数据
    train_data = alldata[:split_index]
    test_data = alldata[split_index:]
    change_train_data = []
    change_test_data = []
    for l in train_data:
        prompt_train = creat_prompt(l['input'],l['instruction'],l['output'],flag=1)
        prompt_value = creat_prompt(l['input'],l['instruction'],l['output'],flag=0)
        change_train_data.append(prompt_train)
        change_test_data.append(prompt_value)

finger_train_data = create_a_chain(Q,T,1)
finger_test_data = create_a_chain(Q,T,0)
change_train_data = change_train_data+finger_train_data

random.shuffle(change_train_data)
write_txt1('Hash_train_data.json',change_train_data)
write_txt('Hash_test_data_50.jsonl',change_test_data)
write_txt('test_Hash.jsonl',finger_test_data)



