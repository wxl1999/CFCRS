import json
import re
import html
from tqdm.auto import tqdm
import sys
sys.path.append('../../')
from config import mask_token

movie_pattern = re.compile(r'@\d+')

item_token = '<item>'
person_token = '<person>'
genre_token = '<genre>'
special_tokens = [item_token, person_token, genre_token]
mask_token = '<mask>'


def process_text(text, id2name, mask_movie=False):
    def convert(match):
        movie_id = match.group(0)[1:]
        if movie_id in id2name:
            movie_name = id2name[movie_id]
            movie_name = ' '.join(movie_name.strip().split())
            if mask_movie:
                return mask_token
            else:
                return movie_name
        else:
            return match.group(0)

    text = re.sub(movie_pattern, convert, text)
    text = ' '.join(text.split())
    text = html.unescape(text)

    return text


def process(split):
    data_file = f"{data_file_dir}/{split}_data.jsonl"
    out_file = f"{split}_data_processed.jsonl"
    out_file = open(out_file, 'w', encoding='utf-8')

    with open(data_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = json.loads(line)
            messages = line['messages']
            if len(messages) == 0:
                continue

            dialog_id = line['conversationId']
            id2movie = line['movieMentions']
            user_id, bot_id = line['initiatorWorkerId'], line['respondentWorkerId']

            context_list = []
            entity_list = []
            template_list = []
            turn_id = 0

            for i, turn in enumerate(messages):
                role_id = turn['senderWorkerId']
                raw_text = turn['text']
                text = process_text(raw_text, id2movie)
                # masked_text = process_text(raw_text, id2movie, mask_movie=True)
                entity_turn = turn['entity']
                movie_turn = turn['item']

                template = turn['masked_text']
                for special_token in special_tokens:
                    template = template.replace(special_token, mask_token)

                template_list.append(template)

                if role_id == bot_id and len(entity_list) > 0 and len(movie_turn) > 0:
                    data = {
                        'dialog_id': dialog_id,
                        'turn_id': turn_id,
                        'context': context_list,
                        'entity': entity_list,
                        'rec': movie_turn,
                        'resp': text,
                        'template': template_list,
                        # 'mask_resp': masked_text
                    }
                    out_file.write(json.dumps(data, ensure_ascii=False) + '\n')

                turn_id += 1
                if i == 0 and role_id == bot_id:
                    context_list.append('')
                context_list.append(text)
                entity_list.extend(entity_turn)
                entity_list.extend(movie_turn)

    out_file.close()


if __name__ == '__main__':
    data_file_dir = '../../../../data/redial'

    for split in ['train', 'valid', 'test']:
        process(split)
