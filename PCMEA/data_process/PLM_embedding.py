
import torch
import numpy as np
import os
import json
import pickle
import re

# from transformers import AutoTokenizer, AutoModel
# from transformers import LlamaModel
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import LlamaTokenizer, LlamaModel


os.environ['CUDA_VISIBLE_DEVICES'] = '2, 1'

# model initialization: Bert,Roberta,T5,Albert,llama,chatglm,
tokenizer = LlamaTokenizer.from_pretrained("model_path", legacy=False)
model = LlamaModel.from_pretrained("model_path").half().to("cuda:1").eval()


def cal_emb(text):
    inputs = tokenizer([text], return_tensors="pt").to("cuda:1")
    resp = model(**inputs, output_hidden_states=True)
    y = resp.last_hidden_state
    y_mean = torch.mean(y, dim=0, keepdim=True)
    y_output = np.resize(y_mean.squeeze().cpu().detach().numpy(), (768,))
    y_output = y_output.astype(np.float32)
    return y_output

def rel_feature_extract(txt_dict):
    feature_dict = {}
    COUNT =0
    for key in txt_dict.keys():
        txt = txt_dict[key]
        t2 = txt.replace('/','')
        words = t2.split()
        txt = " ".join(sorted(set(words), key=words.index))
        txt = txt[:2048]
        d1 = cal_emb(txt)
        feature_dict[int(key)] = d1
        COUNT = COUNT + 1
        print(f"load {COUNT} item")

    return feature_dict

def rel_process():
    rel_text_path = r'FB15K_YAGO15K_id_relation.txt'
    file = open(rel_text_path, 'r')
    js = file.read()
    dic = json.loads(js)
    rel_feature_dict = rel_feature_extract(dic)

    with open('relation_feature.pkl', 'ab') as f:
        pickle.dump(rel_feature_dict, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    file.close()

def att_feature_extract(txt_dict):
    feature_dict = {}
    COUNT = 0
    for key in txt_dict.keys():
        txt = txt_dict[key]
        d1 = cal_emb(txt)
        feature_dict[int(key)] = d1
        COUNT = COUNT + 1
        print(f"load {COUNT} item")
    return feature_dict

def read_attribute_content(file_dirs):
    attribute_dict ={}
    for file_path in file_dirs:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                attribute_dict[params[0]] = params[1:]

    return attribute_dict

def text_process(originate_text):
    text_str = " ".join(originate_text)
    text_str = text_str.replace('<', '').replace('>',';')
    text_str = text_str.replace('http://rdf.freebase.com/ns/','').replace('<http://dbpedia.org/','')
    text_str = text_str.replace('_',' ').replace('.',' ')
    pattern = "[A-Z]"
    new_text = re.sub(pattern, lambda x: " " + x.group(0), text_str)
    return new_text

def check_with_name_id_dict(source, target_file):
    ent2id_dict = {}
    for file_path in target_file:
        id = set()
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")  # ['0', '/m/027rn']
                ent2id_dict[params[1]] = int(params[0])  # {'/m/027rn': 0}
    count =0
    key_list = ent2id_dict.keys()
    for key in key_list:
        if source.get(key):
            # print("key存在")
            id_num = ent2id_dict[key]
            att = source.get(key)
            att_new = text_process(att)
            source[id_num] = att_new
            # pop_att = source.pop(key)
            count = count + 1

    for key in ent2id_dict.keys():
        if source.get(key):
            pop_att = source.pop(key)
        else:
            pass

    print("source Length : %d" % len (source))
    print("have attribution count : "+ str(count))
    return source

def att_process():
    file_dir = r'/data/mmkb-datasets/FB15K_YAGO15K'
    l = [1, 2]
    attribute_dict = read_attribute_content([file_dir + "/training_attrs_" + str(i) for i in l])
    id_attribute_dict = check_with_name_id_dict(attribute_dict, [file_dir + "/ent_ids_" + str(i) for i in l])
    att_feature_dict = att_feature_extract(id_attribute_dict)

    with open('attribute_feature.pkl', 'ab') as f:
        pickle.dump(att_feature_dict, f, pickle.HIGHEST_PROTOCOL)


# rel_process()
att_process()

