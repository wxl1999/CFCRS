import os
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from utils import padded_tensor


class UniCRSDataCollatorForPreTraining:
    def __init__(
        self, tokenizer, prompt_tokenizer, entity_pad_id, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, entity_max_length=None,
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = min(self.tokenizer.model_max_length, self.prompt_tokenizer.model_max_length)

        self.entity_pad_id = entity_pad_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.context_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []

        for data in data_batch:
            context_batch['input_ids'].append(data['context'] + data['resp'])
            prompt_batch['input_ids'].append(data['prompt'] + data['prompt_resp'])
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])

        input_batch = {}
        context_batch = self.tokenizer.pad(
            context_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        context_batch['rec_labels'] = label_batch

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)

        position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        context_batch['position_ids'] = position_ids

        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, pad_id=self.entity_pad_id, pad_tail=True, device=self.device,
            debug=self.debug, max_length=self.entity_max_length
        )
        input_batch['entity'] = entity_batch

        return input_batch


class UniCRSDataCollatorForRec:
    def __init__(
        self, tokenizer, prompt_tokenizer, entity_pad_id, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, entity_max_length=None,
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = min(self.tokenizer.model_max_length, self.prompt_tokenizer.model_max_length)

        self.entity_pad_id = entity_pad_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.context_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []

        for data in data_batch:
            context_batch['input_ids'].append(data['context'])
            prompt_batch['input_ids'].append(data['prompt'])
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])

        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        context_batch['rec_labels'] = label_batch

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)

        position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
        context_batch['position_ids'] = position_ids

        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, pad_id=self.entity_pad_id, pad_tail=True, device=self.device,
            debug=self.debug, max_length=self.entity_max_length
        )
        input_batch['entity'] = entity_batch

        return input_batch


class UniCRSDataCollatorForConv:
    def __init__(
        self, tokenizer, prompt_tokenizer, entity_pad_id, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, resp_max_length=None, entity_max_length=100, gen=False
    ):
        self.gen = gen
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.prompt_tokenizer = prompt_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = min(self.tokenizer.model_max_length, self.prompt_tokenizer.model_max_length)

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.bot_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

        self.entity_pad_id = entity_pad_id
        self.entity_max_length = entity_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        context_len_batch = []
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = defaultdict(list)
        label_len_batch = []

        for data in data_batch:
            if self.gen is False:
                context = data['context'] + data['resp']
            else:
                context = data['context'] + self.bot_prompt
                context_len_batch.append(len(data['context']))
            context_batch['input_ids'].append(context)

            prompt_batch['input_ids'].append(data['prompt'])
            entity_batch.append(data['entity'])

            label_batch['input_ids'].append(data['resp'])
            label_len_batch.append(len(data['resp']))

        input_batch = {}

        if self.gen is False:
            context_max_length = self.context_max_length + self.resp_max_length
        else:
            context_max_length = self.context_max_length + len(self.bot_prompt)
        context_batch = self.tokenizer.pad(
            context_batch, max_length=context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)

        if self.gen is True:
            position_ids = context_batch['attention_mask'].long().cumsum(-1) - 1
            position_ids.masked_fill_(context_batch['attention_mask'] == 0, 1)
            context_batch['position_ids'] = position_ids

            input_batch['conv_labels'] = label_batch['input_ids']
            input_batch['context_len'] = context_len_batch
        else:
            label_ids = context_batch['input_ids'].detach().clone()
            # for i, length in enumerate(label_len_batch):
            #     label_ids[i, :-length] = -100
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
            context_batch['conv_labels'] = label_ids

        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(
            entity_batch, pad_id=self.entity_pad_id, pad_tail=True, device=self.device,
            debug=self.debug, max_length=self.entity_max_length
        )
        input_batch['entity'] = entity_batch

        return input_batch


if __name__ == '__main__':
    from kg_unicrs import KGForUniCRS
    from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
    from pprint import pprint
    from utils import load_jsonl_data
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from dataset import DatasetForConv

    debug = True
    gen = False
    kg = 'redial'
    dataset = 'redial_conv'
    split = 'test'

    kg_info = KGForUniCRS(kg, debug=debug).get_kg_info()

    model_name_or_path = '../../utils/tokenizer/dialogpt-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    prompt_tokenizer = AutoTokenizer.from_pretrained('../../utils/tokenizer/roberta-base')
    prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    data_file = os.path.join('data', dataset, f'{split}_data_processed.jsonl')
    data_list = load_jsonl_data(data_file)
    dataset = DatasetForConv(
        data_list, entity2id=kg_info['entity2id'], tokenizer=tokenizer, prompt_tokenizer=prompt_tokenizer, debug=debug,
    )
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(prompt_tokenizer.decode(data['prompt']))
        print(tokenizer.decode(data['resp']))
        # print(prompt_tokenizer.decode(data['prompt_resp']))
        print()

    if debug is False:
        context_max_len, context_mean_len = 0, 0
        resp_max_len, resp_mean_len = 0, 0
        entity_max_len, entity_mean_len = 0, 0
        for data in tqdm(dataset):
            context_len = len(data['context'])
            context_max_len = max(context_max_len, context_len)
            context_mean_len += context_len

            resp_len = len(data['resp'])
            resp_max_len = max(resp_max_len, resp_len)
            resp_mean_len += resp_len

            entity_len = len(data['entity'])
            entity_max_len = max(entity_max_len, entity_len)
            entity_mean_len += entity_len

        print(context_max_len, context_mean_len / len(dataset))
        print(resp_max_len, resp_mean_len / len(dataset))
        print(entity_max_len, entity_mean_len / len(dataset))

        # redial: (803, 43), (645, 43), (531, 35) -> (803, 43)
        # inspired: (1024, 134, 31), (831, 120, 23), (749, 128, 30) -> ()

        # redial: (1024, 31), (688, 29), (585, 19) -> (1024, 31)
        # inspired: (1024, 30), (902, 23), (945, 32) -> (1024, 32)

        # conv
        # redial: (1024, 187, 43), (669, 124, 44), (579, 130, 39) -> ()

    data_collator = UniCRSDataCollatorForConv(
        tokenizer=tokenizer, prompt_tokenizer=prompt_tokenizer, entity_pad_id=kg_info['pad_entity_id'], gen=gen,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=5,
        collate_fn=data_collator,
    )

    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            for i in range(len(batch['context']['input_ids'])):
                print(tokenizer.decode(batch['context']['input_ids'][i]))
                print(prompt_tokenizer.decode(batch['prompt']['input_ids'][i]))
            exit()
