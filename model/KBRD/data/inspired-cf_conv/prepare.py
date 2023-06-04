import json
from tqdm import tqdm


def process(split):
    data_file = f"../../save/inspired_epoch-10_it-10_lr-0.5_l2-0.1_decay-0.95_flow_bs-50_beam-20_kbrd-final_it-1_lr-1e-4_l2-0.01_aug.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            if data['epoch'] != epoch:
                continue

            context_list = data['context']
            entity_list = data['entity']
            resp = data['resp']

            out_file.write(json.dumps({
                'context': context_list,
                'entity': entity_list,
                'resp': resp,
            }, ensure_ascii=False) + '\n')

    out_file.close()


if __name__ == '__main__':
    epoch = 9
    process('train')
