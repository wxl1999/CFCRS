from collections import defaultdict

import torch
from utils import padded_tensor


class KBRDDataCollatorForRec:
    def __init__(self, pad_id, max_length=100, device=torch.device('cpu'), debug=False):
        self.debug = debug
        self.device = device

        self.pad_id = pad_id
        self.max_length = max_length

    def __call__(self, data_batch):
        dialog_id_batch = []
        turn_id_batch = []
        entity_ids = []
        label_batch = []
        weight_batch = []

        for data in data_batch:
            entity_ids.append(data['entity'])
            if 'item' in data:
                label_batch.append(data['item'])

            if 'weight' not in data:
                weight_batch.append(1)
            else:
                weight_batch.append(data['weight'])

            if 'dialog_id' in data:
                dialog_id_batch.append(data['dialog_id'])
            if 'turn_id' in data:
                turn_id_batch.append(data['turn_id'])

        input_batch = {
            'dialog_id': dialog_id_batch,
            'turn_id': turn_id_batch,
            'weight': torch.as_tensor(weight_batch, device=self.device)
        }

        entity_ids = padded_tensor(
            entity_ids, pad_id=self.pad_id, pad_tail=True, max_length=self.max_length,
            device=self.device, debug=self.debug,
        )
        entity_batch = {
            'entity_ids': entity_ids,
            'entity_mask': torch.ne(entity_ids, self.pad_id),
        }
        if len(label_batch) > 0:
            entity_batch['labels'] = torch.as_tensor(label_batch, device=self.device)
        input_batch['entity'] = entity_batch

        return input_batch


class KBRDDataCollatorForConv:
    def __init__(
        self, tokenizer, entity_pad_id, device=torch.device('cpu'), debug=False, use_amp=False,
        context_max_length=None, resp_max_length=None, entity_max_length=100
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

        self.entity_pad_id = entity_pad_id
        self.entity_max_length = entity_max_length

    def __call__(self, data_batch):
        context_batch = defaultdict(list)
        entity_batch = []
        label_batch = defaultdict(list)

        for data in data_batch:
            entity_batch.append(data['entity'])
            context_batch['input_ids'].append(data['context'])
            label_batch['input_ids'].append(data['resp'])

        context_batch = self.tokenizer.pad(
            context_batch, max_length=self.context_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_batch = self.tokenizer.pad(
            label_batch, max_length=self.resp_max_length,
            padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of
        )
        context_batch['labels'] = label_batch['input_ids']

        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)

        entity_ids = padded_tensor(
            entity_batch, pad_id=self.entity_pad_id, pad_tail=True, device=self.device,
            debug=self.debug, max_length=self.entity_max_length
        )
        entity_batch = {
            'entity_ids': entity_ids,
            'entity_mask': torch.ne(entity_ids, self.entity_pad_id),
        }

        batch = {
            'context': context_batch,
            'entity': entity_batch
        }
        return batch
