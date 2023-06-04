from collections import defaultdict

import os
import torch


class BARTDataCollatorForRec:
    def __init__(
        self, tokenizer, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch):
        input_batch = defaultdict(list)
        label_batch = []
        weight_batch = []

        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            label_batch.append(data['rec'])
            if 'weight' not in data:
                weight_batch.append(1)
            else:
                weight_batch.append(data['weight'])

        batch = {'weight': torch.as_tensor(weight_batch, device=self.device)}

        input_batch = self.tokenizer.pad(
            input_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        input_batch['labels'] = label_batch

        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)

        # position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        # position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        # context_batch['position_ids'] = position_ids

        batch['input'] = input_batch

        return batch


class BARTDataCollatorForConv:
    def __init__(
        self, tokenizer, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, resp_max_length=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

    def __call__(self, data_batch):
        input_batch = defaultdict(list)
        label_batch = defaultdict(list)

        for data in data_batch:
            input_batch['input_ids'].append(data['context'])
            label_batch['input_ids'].append(data['resp'])

        input_batch = self.tokenizer.pad(
            input_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_ids = self.tokenizer.pad(
            label_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )['input_ids']
        input_batch['labels'] = label_ids

        for k, v in input_batch.items():
            if not isinstance(v, torch.Tensor):
                input_batch[k] = torch.as_tensor(v, device=self.device)

        return input_batch


if __name__ == '__main__':
    from kg_bart import KGForBART
    from pprint import pprint
    from utils import load_jsonl_data
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from dataset import DatasetForRec, DatasetForConv

    debug = True
    kg = 'redial'
    dataset = 'redial_conv'
    split = 'test'

    kg_info = KGForBART(kg, debug=debug).get_kg_info()

    model_name_or_path = '../../utils/tokenizer/bart-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    data_file = os.path.join('data', dataset, f'{split}_data_processed.jsonl')
    data_list = load_jsonl_data(data_file)
    dataset = DatasetForConv(
        data_list, entity2id=kg_info['entity2id'], tokenizer=tokenizer, debug=debug
    )

    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print()

    if debug is False:
        context_max_len, context_mean_len = 0, 0
        for data in dataset:
            context_len = len(data['context'])
            context_max_len = max(context_max_len, context_len)
            context_mean_len += context_len

        print(context_max_len, context_mean_len / len(dataset))
        # redial: (803, 43), (645, 43), (531, 35) -> (803, 43)

    data_collator = BARTDataCollatorForConv(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=data_collator,
    )

    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            for input_ids in batch['input_ids']:
                print(tokenizer.decode(input_ids))
            exit()
