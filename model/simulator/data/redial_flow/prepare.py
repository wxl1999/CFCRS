import json
from collections import defaultdict

from tqdm import tqdm


def prepare(split):
    data_file_dir = "../../../../data/redial"
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')
    cnt = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            dialog_id = line['conversationId']
            user, bot = str(line['initiatorWorkerId']), str(line['respondentWorkerId'])

            thread_list = []
            last_role = None
            user2entity = defaultdict(list)

            for turn in line['messages']:
                thread = turn['thread']
                if len(thread) == 0:
                    continue

                role_id = str(turn['senderWorkerId'])
                # turn_id = turn['turn_id']

                user2entity[role_id].extend(thread)

                # if role_id != last_role:
                #     thread.insert(0, role_id)

                thread_list.extend(thread)
                last_role = role_id

            if user in user2entity and bot in user2entity:
                out_file.write(json.dumps({
                    # 'dialog_id': dialog_id,
                    # 'turn_id': turn_id,
                    'user': [user, bot],
                    'user2entity': user2entity,
                    'flow': thread_list,
                }, ensure_ascii=False) + '\n')
                cnt += 1

    out_file.close()
    print(cnt)


if __name__ == '__main__':
    for split in ['train', 'valid', 'test']:
        prepare(split)
