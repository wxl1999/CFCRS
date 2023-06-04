import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from multiprocessing import Pool

from accelerate.utils import set_seed
from tqdm import tqdm


def check_path(path):
    # if entityid2type[path[-1]] == 'user':
    #     return False

    # user_set = set()
    # user_cnt = 0
    ent_cnt = defaultdict(int)
    meta_path = []
    for node in path:
        # if entityid2type[node] == 'user':
        #     user_set.add(node)
        #     if len(user_set) > 2:
        #         return False
        #     user_cnt += 1
        # else:
        #     if ent_cnt[node] == 1:
        #         return False
        #     ent_cnt[node] += 1

        if ent_cnt[node] == 1:
            return False
        ent_cnt[node] += 1
        meta_path.append(entityid2type[node])

    meta_path.append('special_token')
    if tuple(meta_path) not in metapath_set:
        return False
    return True

    # if len(user_set) == 2 and user_cnt >= 2:
    #     return True

    # return False


def generate_paths(meta_path):
    path_set = set()
    repeat_cnt = 0

    while len(path_set) < n_walks:
        path = []
        ent_set = set()

        start_typ = meta_path[0]
        start_ent = random.choice(type2entityids[start_typ])
        path.append(start_ent)
        ent_set.add(start_ent)

        fail_flag = False

        for i in range(1, len(meta_path) - 1):
            last_ent = path[i - 1]
            cur_typ = meta_path[i]
            candidate_ents = edge_dict[last_ent][cur_typ]
            if len(candidate_ents) == 0:
                fail_flag = True
                break

            cur_ent = random.choice(candidate_ents)

            fail_cnt = 0
            while cur_ent in ent_set:
                fail_cnt += 1
                if fail_cnt == 3:
                    break

                cur_ent = random.choice(candidate_ents)

            if fail_cnt == 3:
                fail_flag = True
                break

            path.append(cur_ent)
            ent_set.add(cur_ent)

        if fail_flag is False:
            path_tup = tuple(path)
            if path_tup in path_set:
                repeat_cnt += 1
                if repeat_cnt == 100:
                    break
            else:
                path_set.add(path_tup)
                repeat_cnt = 0

    return path_set


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--n_walks", type=int)
    args = parser.parse_args()

    n_walks = args.n_walks
    dataset = 'inspired'

    set_seed(42)

    with open(f'../{dataset}-meta/id2metapath.json', encoding='utf-8') as f:
        id2metapath = json.load(f)
        metapath_set = set(map(tuple, id2metapath.values()))

    with open(f'../{dataset}/entity2id.json', encoding='utf-8') as f:
        entity2id = json.load(f)

    with open(f'../{dataset}/entity2type.json', encoding='utf-8') as f:
        entity2type = json.load(f)

        type_set = set(entity2type.values())

        entityid2type = {}
        for ent, typ in entity2type.items():
            entityid2type[entity2id[ent]] = typ

        type2entityids = defaultdict(set)
        for ent, typ in entity2type.items():
            type2entityids[typ].add(entity2id[ent])
        for k, v in type2entityids.items():
            type2entityids[k] = list(v)

    with open(f'../{dataset}/kg.json', encoding='utf-8') as f:
        kg = json.load(f)

        edge_dict_1 = defaultdict(lambda: defaultdict(set))
        for item, attr_value_list in kg.items():
            item = int(item)
            for attr_value in attr_value_list:
                value = attr_value[1]
                edge_dict_1[item][entityid2type[value]].add(value)
                edge_dict_1[value][entityid2type[item]].add(item)

        edge_dict_2 = defaultdict(lambda: defaultdict(set))
        for head, edge_dict in edge_dict_1.items():
            missed_type_set = type_set - set(edge_dict.keys())

            for relation, tail_set in edge_dict.items():
                for tail in tail_set:
                    for missed_type in missed_type_set:
                        if missed_type in edge_dict_1[tail]:
                            edge_dict_2[head][missed_type] |= edge_dict_1[tail][missed_type]

        edge_dict = {}
        for k, v in edge_dict_1.items():
            v.update(edge_dict_2[k])
            for v_k, v_v in v.items():
                v[v_k] = list(v_v)
            edge_dict[k] = v

    all_path_set = set()
    with Pool(16) as p:
        for path_set in tqdm(p.imap_unordered(generate_paths, metapath_set), total=len(metapath_set)):
            all_path_set |= path_set
    # for meta_path in tqdm(metapath_set):
    #     all_path_set |= generate_paths(meta_path)
    print(len(all_path_set))

    with open(f'path_num-{n_walks}.jsonl', 'w', encoding='utf-8') as f:
        for path in all_path_set:
            f.write(json.dumps(path, ensure_ascii=False) + '\n')
