import json
from collections import defaultdict

from tqdm import tqdm

item_token = '<item>'
person_token = '<person>'
genre_token = '<genre>'
special_tokens = [item_token, person_token, genre_token]
mask_token = '<mask>'


def prepare(split):
    data_file_dir = "../../../../data/redial"
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            dialog_id = line['conversationId']
            user_id, bot_id = str(line['initiatorWorkerId']), str(line['respondentWorkerId'])

            thread_list = []
            last_role = None
            user2entity = defaultdict(list)
            template_list = []
            entity_list = []

            for turn in line['messages']:
                turn_id = turn['turn_id']
                role_id = str(turn['senderWorkerId'])
                entity_turn = turn['entity']
                movie_turn = turn['item']

                template = turn['masked_text']
                for special_token in special_tokens:
                    template = template.replace(special_token, mask_token)

                template_list.append(template)

                thread = turn['thread']
                if len(thread) > 0:
                    user2entity[role_id].extend(thread)

                    # if role_id != last_role:
                    #     thread.insert(0, role_id)
                    thread_list.extend(thread)
                    last_role = role_id

                    if user_id in user2entity and bot_id in user2entity \
                            and role_id == bot_id and len(entity_list) > 0 and len(movie_turn) > 0:
                        data = {
                            'dialog_id': dialog_id,
                            'turn_id': turn_id,
                            'user': [user_id, bot_id],
                            'user2entity': user2entity,
                            'flow': thread_list,
                            'template': template_list,
                        }
                        out_file.write(json.dumps(data, ensure_ascii=False) + '\n')

                turn_id += 1
                entity_list.extend(entity_turn)
                entity_list.extend(movie_turn)

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
