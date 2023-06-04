import json
from tqdm import tqdm


def process(split):
    data_file = f"../../save/inspired_epoch-5_it-10_lr-0.5_l2-0.001_decay-0.9_flow_bs-80_beam-20_cf_bart-final_it-1_lr-5e-5_l2-0.01_aug-mix.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    cnt = 0

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['epoch'] != epoch:
                continue

            context_list = data['context']
            # entity_list = data['entity']
            resp = data['resp']

            out_file.write(json.dumps({
                'context': context_list,
                # 'entity': entity_list,
                'resp': resp,
            }, ensure_ascii=False) + '\n')
            cnt += 1

    out_file.close()
    print(cnt)


if __name__ == '__main__':
    epoch = 4
    process('train')
