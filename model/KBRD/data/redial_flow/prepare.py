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
    if split == 'train':
        data_file = f"{data_file_dir}/{split}_data.jsonl"
    else:
        data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            dialog_id = line['conversationId']
            user, bot = str(line['initiatorWorkerId']), str(line['respondentWorkerId'])

            thread_list = []
            last_role = None
            user2entity = defaultdict(list)
            template_list = []

            for turn in line['messages']:
                role_id = str(turn['senderWorkerId'])
                # turn_id = turn['turn_id']

                template = turn['masked_text']
                for special_token in special_tokens:
                    template = template.replace(special_token, mask_token)
                template_list.append(template)

                thread = turn['thread']
                if len(thread) == 0:
                    continue

                user2entity[role_id].extend(thread)

                if role_id != last_role:
                    thread.insert(0, role_id)
                thread_list.extend(thread)
                last_role = role_id

            if user in user2entity and bot in user2entity:
                out_file.write(json.dumps({
                    'dialog_id': dialog_id,
                    # 'turn_id': turn_id,
                    'user': [user, bot],
                    'user2entity': user2entity,
                    'flow': thread_list,
                    'template': template_list,
                }, ensure_ascii=False) + '\n')

    out_file.close()


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        prepare(split)
