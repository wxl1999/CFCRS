import json
from tqdm import tqdm


def process(split):
    data_file = f"../../save/redial_epoch-5_it-5_lr-0.5_l2-0.001_decay-0.9_flow_bs-150_beam-5_bart-final_it-1_lr-5e-5_l2-0.01_aug-pre.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

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

    out_file.close()


if __name__ == '__main__':
    epoch = 4
    process('train')
