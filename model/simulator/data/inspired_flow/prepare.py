import json
from collections import defaultdict

from tqdm import tqdm


def prepare(split):
    user_id = 'user'
    bot_id = 'bot'

    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')
    cnt = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            # user_id = f'u-{user_idx}'
            # user_idx += 1
            # bot_id = f'u-{user_idx}'
            # user_idx += 1

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
                # last_role = role_id

            if user_id in user2entity and bot_id in user2entity:
                out_file.write(json.dumps({
                    # 'dialog_id': dialog_id,
                    # 'turn_id': turn_id,
                    'user': [user_id, bot_id],
                    'user2entity': user2entity,
                    'flow': flow_list,
                }, ensure_ascii=False) + '\n')
                cnt += 1

    out_file.close()
    print(cnt)


if __name__ == '__main__':
    data_file_dir = "../../../../data/inspired"
    with open(f'{data_file_dir}/entity2id.json', encoding='utf-8') as f:
        entity2id = json.load(f)

    for split in ['train', 'valid', 'test']:
        prepare(split)
