import json

def relation_id_read(entity_id_dict, entity_triple_path):
    rel2id_dict ={}
    with open(entity_triple_path, "r", encoding="utf-8") as fr:
        for line in fr:
            params = line.strip("\n").split(" ")
            print(params)
            params = line.strip('\n').split(' ')
            e1 = params[0]
            e2 = params[2]
            r = params[1]

            if e1 in entity_id_dict.keys():
                id1 = entity_id_dict[e1]
                id1= int(id1)
                if id1 in rel2id_dict.keys():
                    val1 = rel2id_dict[id1]
                    rel2id_dict[id1] = val1 +' ' + r
                else:
                    rel2id_dict[id1] = r

            if e2 in entity_id_dict.keys():
                id2 = entity_id_dict[e2]
                id2 = int(id2)
                if id2 in rel2id_dict.keys():
                    val2 = rel2id_dict[id2]
                    rel2id_dict[id2] = val2 + ' ' + r
                else:
                    rel2id_dict[id2] = r

    return rel2id_dict

def entity_id_read(file_paths):
    ent2id_dict = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.strip("\n").split("\t")
                ent2id_dict[params[1]] = int(params[0])
    return ent2id_dict

entity_triple_path = r'/mmkb-master/FB15K/FB15K_EntityTriples.txt'
file_dir = r'/mmkb-datasets/FB15K_YAGO15K'
l=[1,2]
ent2id_dict = entity_id_read([file_dir + "/ent_ids_" + str(i) for i in l])
relation_dict = relation_id_read(ent2id_dict, entity_triple_path)

with open('FB15K_YAGO15K_id_relation.txt', 'w', encoding='utf-8') as f:
    str_ = json.dumps(relation_dict)
    f.write(str_)