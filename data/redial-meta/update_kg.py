import json
import shutil
from collections import defaultdict

from tqdm import tqdm

dataset = 'redial'
type_rel = 'type'

with open(f'../{dataset}/entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)
    id2entity = {idx: ent for ent, idx in entity2id.items()}
    entity2id = defaultdict(lambda: len(entity2id), entity2id)
with open(f'../{dataset}/entity2type.json', encoding='utf-8') as f:
    entity2type = json.load(f)
with open(f'../{dataset}/relation2id.json', encoding='utf-8') as f:
    relation2id = json.load(f)
    type_id = relation2id[type_rel] = len(relation2id)
with open(f'../{dataset}/kg.json', encoding='utf-8') as f:
    kg = json.load(f)

new_kg = defaultdict(list)
for head, relation_tail_list in tqdm(kg.items()):
    head = int(head)
    new_kg[head].extend(relation_tail_list)
    new_kg[head].append([type_id, entity2id[entity2type[id2entity[head]]]])
    new_kg[entity2id[entity2type[id2entity[head]]]].append([type_id, head])

    for relation_tail in relation_tail_list:
        relation, tail = relation_tail
        if len(new_kg[tail]) == 0:
            new_kg[tail].append([type_id, entity2id[entity2type[id2entity[tail]]]])
            new_kg[entity2id[entity2type[id2entity[tail]]]].append([type_id, tail])
        new_kg[tail].append([relation, head])

with open('entity2id.json', 'w', encoding='utf-8') as f:
    json.dump(entity2id, f, ensure_ascii=False)
with open('relation2id.json', 'w', encoding='utf-8') as f:
    json.dump(relation2id, f, ensure_ascii=False)

for k, v in new_kg.items():
    new_kg[k] = sorted(list(set(map(tuple, v))))
new_kg = sorted(new_kg.items(), key=lambda x: x[0])
new_kg = dict(new_kg)
with open('kg.json', 'w', encoding='utf-8') as f:
    json.dump(new_kg, f, ensure_ascii=False)

file_list = [
    'item_ids.json', 'entity2type.json', 'type2id.json'
]
for file in file_list:
    shutil.copyfile(f'../{dataset}/{file}', f'{file}')
