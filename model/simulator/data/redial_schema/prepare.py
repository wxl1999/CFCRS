import json
from collections import defaultdict

from tqdm import tqdm


def prepare(split):
    data_file_dir = "../../../../data/redial"
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    cnt_meta = 0
    cnt_user = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            # dialog_id = line['conversationId']
            user, bot = str(line['initiatorWorkerId']), str(line['respondentWorkerId'])

            thread_list = []
            # last_role = None
            user2entity = defaultdict(list)

            for turn in line['messages']:
                role_id = str(turn['senderWorkerId'])
                # turn_id = turn['turn_id']
                thread = turn['thread']
                if len(thread) == 0:
                    continue

                user2entity[role_id].extend(thread)

                # if role_id != last_role:
                #     thread.insert(0, role_id)
                thread_list.extend(thread)
                # last_role = role_id

            meta_path = tuple([entity2type[ent] if ent in entity2type else 'user' for ent in thread_list])
            meta_path += ('special_token',)

            if user in user2entity and bot in user2entity:
                if meta_path in metapath2id:
                    out_file.write(json.dumps({
                        # 'dialog_id': dialog_id,
                        # 'turn_id': turn_id,
                        'user': [user, bot],
                        'user2entity': user2entity,
                        'flow': thread_list,
                        'meta_path_label': metapath2id[meta_path]
                    }, ensure_ascii=False) + '\n')
                else:
                    cnt_meta += 1
            else:
                cnt_user += 1

    print(cnt_meta, cnt_user)
    out_file.close()


if __name__ == '__main__':
    data_file_dir = "../../../../data/redial-meta"
    with open(f'{data_file_dir}/id2metapath.json', encoding='utf-8') as f:
        id2metapath = json.load(f)
        metapath2id = {tuple(meta_path): int(idx) for idx, meta_path in id2metapath.items()}
    with open(f'{data_file_dir}/entity2type.json', encoding='utf-8') as f:
        entity2type = json.load(f)

    for split in ['train', 'valid', 'test']:
        prepare(split)
