import json
from collections import defaultdict

from tqdm import tqdm

movie_token = '<movie>'
person_token = '<person>'
genre_token = '<genre>'
special_tokens = [movie_token, person_token, genre_token]
mask_token = '<mask>'


def prepare(split):
    global user_idx

    data_file_dir = "../../../../data/inspired"
    if split == 'train':
        data_file = f"{data_file_dir}/{split}_data.jsonl"
    else:
        data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            dialog_id = line['dialog_id']

            user_id = f'u{user_idx}'
            user_idx += 1
            bot_id = f'u{user_idx}'
            user_idx += 1

            thread_list = []
            last_role = None
            user2entity = defaultdict(list)
            template_list = []

            for turn in line['dialog']:
                role = turn['role']
                if role == 'SEEKER':
                    role_id = user_id
                else:
                    role_id = bot_id

                template = turn['text_template']
                for special_token in special_tokens:
                    template = template.replace(special_token, mask_token)
                template_list.append(template)

                thread = [ent for ent in turn['flow'] if ent in entity2id]
                if len(thread) == 0:
                    continue

                user2entity[role_id].extend(thread)

                # if role_id != last_role:
                #     thread.insert(0, role_id)

                thread_list.extend(thread)
                last_role = role_id

            if user_id in user2entity and bot_id in user2entity:
                out_file.write(json.dumps({
                    'dialog_id': dialog_id,
                    # 'turn_id': turn_id,
                    'user': [user_id, bot_id],
                    'user2entity': user2entity,
                    'flow': thread_list,
                    'template': template_list,
                }, ensure_ascii=False) + '\n')

    out_file.close()


if __name__ == '__main__':
    data_file_dir = "../../../../data/inspired-meta"
    with open(f'{data_file_dir}/entity2id.json', encoding='utf-8') as f:
        entity2id = json.load(f)

    user_idx = 0

    for split in ['train', 'valid', 'test']:
        prepare(split)
