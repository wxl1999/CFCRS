import json
from tqdm.auto import tqdm


def process(split):
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            messages = line['dialog']
            if len(messages) == 0:
                continue

            dialog_id = line['dialog_id']

            context_list = []
            entity_list = []
            turn_id = 0

            for i, turn in enumerate(messages):
                role = turn['role']
                text = turn['text']
                mask_text = turn['text_template']
                entity_turn = turn['entity']
                movie_turn = turn['movie']

                if len(movie_turn + entity_turn) > 0:
                    data = {
                        'dialog_id': dialog_id,
                        'turn_id': turn_id,
                        'context': context_list,
                        'entity': entity_list,
                        'rec': movie_turn + entity_turn,
                        'resp': text,
                        # 'mask_resp': masked_text
                    }
                    out_file.write(json.dumps(data, ensure_ascii=False) + '\n')

                turn_id += 1
                if i == 0 and role == 'RECOMMENDER':
                    context_list.append('')
                context_list.append(text)
                entity_list.extend(entity_turn)
                entity_list.extend(movie_turn)

    out_file.close()


if __name__ == '__main__':
    data_file_dir = '../../../../data/inspired'

    for split in ['train', 'valid', 'test']:
        process(split)
