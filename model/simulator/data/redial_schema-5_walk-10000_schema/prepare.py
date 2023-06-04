import json
from tqdm import tqdm

path_data_dir = '../../../../data/redial-meta'
with open(f'{path_data_dir}/entity2type.json', encoding='utf-8') as f:
    entity2type = json.load(f)
with open('../../../../data/redial-meta/id2metapath.json', encoding='utf-8') as f:
    id2metapath = json.load(f)
    metapath2id = {tuple(meta_path): int(idx) for idx, meta_path in id2metapath.items()}

data_file = open('train_data_processed.jsonl', 'w', encoding='utf-8')

with open(f'../redial_schema-5_walk-10000_flow/train_data_processed.jsonl', encoding='utf-8') as f:
    for line in tqdm(f):
        data = json.loads(line)
        users = data['user']
        user2entity = data['user2entity']
        flow = data['flow']

        meta_path = tuple([entity2type[ent] for ent in flow])
        meta_path += ('special_token',)
        if meta_path not in metapath2id:
            continue

        data_file.write(json.dumps({
            'user': users,
            'user2entity': user2entity,
            'flow': flow,
            'meta_path_label': metapath2id[meta_path]
        }, ensure_ascii=False) + '\n')

data_file.close()
