import json
from collections import defaultdict

from tqdm import tqdm

min_schema_cnt = 5

data_file_dir = "../redial"
with open(f'{data_file_dir}/entity2type.json', encoding='utf-8') as f:
    entity2type = json.load(f)
with open(f'{data_file_dir}/entity2id.json', encoding='utf-8') as f:
    entity2id = json.load(f)
    id2entity = {idx: ent for ent, idx in entity2id.items()}

metapath_cnt = defaultdict(int)

with open(f'../redial/train_data.jsonl', encoding='utf-8') as f:
    for line in tqdm(f):
        line = json.loads(line)
        meta_path = []
        # user_set = set()
        # last_role = None

        for turn in line['messages']:
            role_id = str(turn['senderWorkerId'])
            flow = [ent for ent in turn['thread'] if ent in entity2id]
            if len(flow) == 0:
                continue

            # if role_id != last_role:
            #     meta_path.append('user')
            #     user_set.add(role_id)
            # last_role = role_id

            for ent in flow:
                if ent in entity2type:
                    meta_path.append(entity2type[ent])

        # if len(user_set) == 2:
        metapath_cnt[tuple(meta_path)] += 1

# data_file = os.path.join(data_file_dir, 'path_user_len-15_num-1000.jsonl')
# with open(data_file, encoding='utf-8') as f:
#     for line in tqdm(f):
#         flow = json.loads(line)
#         flow = [id2entity[idx] for idx in flow]
#
#         meta_path = []
#         last_user = None
#         for ent in flow:
#             if entity2type[ent] == 'user':
#                 if ent == last_user:
#                     continue
#                 last_user = ent
#                 meta_path.append('user')
#             else:
#                 meta_path.append(entity2type[ent])
#
#         metapath_cnt[tuple(meta_path)] += 1

metapath_cnt = sorted(metapath_cnt.items(), key=lambda x: x[1], reverse=True)
metapath_cnt = dict(metapath_cnt)

with open('metapath_cnt.jsonl', 'w', encoding='utf-8') as f:
    for i, (k, v) in enumerate(metapath_cnt.items()):
        f.write(json.dumps([i, k, v], ensure_ascii=False) + '\n')

id2metapath = {}
min_length = 100
max_length = 0
for i, (meta_path, cnt) in enumerate(metapath_cnt.items()):
    if cnt < min_schema_cnt:
        break
    id2metapath[i] = meta_path + ('special_token',)
    max_length = max(max_length, len(id2metapath[i]))
    min_length = min(min_length, len(id2metapath[i]))

    # user_cnt = 0
    # for typ in meta_path:
    #     if typ == 'user':
    #         user_cnt += 1
    #
    # if user_cnt > 2:
    #     more_than_two_users += cnt

print(min_length, max_length)
# print(more_than_two_users)

with open('id2metapath.json', 'w', encoding='utf-8') as f:
    json.dump(id2metapath, f, ensure_ascii=False)
