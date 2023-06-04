import json
from tqdm import tqdm


def process(split):
    data_file = f"../../save/redial_epoch-10_it-10_lr-0.5_l2-0_flow_bs-200_beam-5_kbrd-best_it-1_lr-1e-4_l2-0.01_aug.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    cnt = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['epoch'] < epoch:
                continue
            elif data['epoch'] > epoch:
                break

            context_list = data['context']
            entity_list = data['entity']
            resp = data['resp']

            out_file.write(json.dumps({
                'context': context_list,
                'entity': entity_list,
                'resp': resp,
            }, ensure_ascii=False) + '\n')
            cnt += 1

    print(cnt)
    out_file.close()


if __name__ == '__main__':
    epoch = 9
    process('train')
