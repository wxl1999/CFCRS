import random
import shutil

random.seed(42)

with open('train_data_link.jsonl', encoding='utf-8') as f:
    data = f.readlines()
all_data_len = len(data)
print(all_data_len)

random.shuffle(data)
valid_data = data[:int(all_data_len * 0.1)]
train_data = data[int(all_data_len * 0.1):]
print(len(train_data), len(valid_data))


def save_data(file_name, data):
    with open(file_name, 'w', encoding='utf-8') as f:
        f.writelines(data)


save_data('train_data.jsonl', train_data)
save_data('valid_data.jsonl', valid_data)

shutil.copyfile('test_data_link.jsonl', 'test_data.jsonl')
