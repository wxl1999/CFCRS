import json
from collections import defaultdict

from tqdm import tqdm


def prepare(split):
    user_id = 'user'
    bot_id = 'bot'

    data_file_dir = "../../../../data/inspired"
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    cnt_meta = 0
    cnt_user = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            # user_id = f'u-{user_idx}'
            # user_idx += 1
            # bot_id = f'u-{user_idx}'
            # user_idx += 1

            dialog_id = line['dialog_id']

            flow_list = []
            last_role = None
            user2entity = defaultdict(list)

            for turn in line['dialog']:
                flow = [ent for ent in turn['flow'] if ent in entity2id]
                if len(flow) == 0:
                    continue

                role = turn['role']
                if role == 'SEEKER':
                    role_id = user_id
                else:
                    role_id = bot_id

                user2entity[role_id].extend(flow)

                # if role_id != last_role:
                #     flow.insert(0, role_id)

                flow_list.extend(flow)
                last_role = role_id

            meta_path = tuple([entity2type[ent] if ent in entity2type else 'user' for ent in flow_list])
            meta_path += ('special_token',)

            if user_id in user2entity and bot_id in user2entity:
                if meta_path in metapath2id:
                    out_file.write(json.dumps({
                        # 'dialog_id': dialog_id,
                        # 'turn_id': turn_id,
                        'user': [user_id, bot_id],
                        'user2entity': user2entity,
                        'flow': flow_list,
                        'meta_path_label': metapath2id[meta_path]
                    }, ensure_ascii=False) + '\n')
                else:
                    cnt_meta += 1
            else:
                cnt_user += 1

    print(cnt_meta, cnt_user)
    out_file.close()


if __name__ == '__main__':
    data_file_dir = "../../../../data/inspired-meta"
    with open(f'{data_file_dir}/entity2id.json', encoding='utf-8') as f:
        entity2id = json.load(f)
    with open(f'{data_file_dir}/entity2type.json', encoding='utf-8') as f:
        entity2type = json.load(f)
    with open(f'{data_file_dir}/id2metapath.json', encoding='utf-8') as f:
        id2metapath = json.load(f)
        metapath2id = {tuple(meta_path): int(idx) for idx, meta_path in id2metapath.items()}

    user_idx = 0

    for split in ['train', 'valid', 'test']:
        prepare(split)
