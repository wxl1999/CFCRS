import json
import random
from collections import defaultdict

from tqdm import tqdm

random.seed(42)

path_data_dir = '../../../../data/inspired'
with open(f'{path_data_dir}/entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)
    id2entity = {idx: ent for ent, idx in entity2id.items()}
with open(f'{path_data_dir}/entity2type.json', encoding='utf-8') as f:
    entity2type = json.load(f)

dialog_id = 0
data_file = open('train_data_processed.jsonl', 'w', encoding='utf-8')
# cnt = 0
# more_than_two_user_cnt = 0
users = ['user', 'bot']

with open(f'../../../../data/inspired-flow/path_num-10000.jsonl', encoding='utf-8') as f:
    for line in tqdm(f):
        flow = json.loads(line)
        flow = [id2entity[idx] for idx in flow]

        user_ent_num = random.randint(1, len(flow) - 1)
        user_ents = set(random.sample(flow, user_ent_num))

        user2entity = defaultdict(list)
        for ent in flow:
            if ent in user_ents:
                user2entity['user'].append(ent)
            else:
                user2entity['bot'].append(ent)

        # flow_dep = []
        # users = []
        # user2entity = defaultdict(list)
        # user_cnt = 0
        # last_user = None
        #
        # for ent in flow:
        #     if entity2type[ent] == 'user':
        #         if ent not in users:
        #             users.append(ent)
        #
        #         if ent == last_user:
        #             continue
        #         else:
        #             last_user = ent
        #             user_cnt += 1
        #     else:
        #         user2entity[last_user].append(ent)
        #     flow_dep.append(ent)

        # if user_cnt > 2:
        #     more_than_two_user_cnt += 1

        data_file.write(json.dumps({
            'user': users,
            'user2entity': user2entity,
            'flow': flow,
        }, ensure_ascii=False) + '\n')

        # cnt += 1

data_file.close()
# print(cnt, more_than_two_user_cnt)
