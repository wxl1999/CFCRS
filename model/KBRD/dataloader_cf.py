import re

import torch

from utils import padded_tensor


class UserDataCollator:
    def __init__(self, kg, max_length=100, device=torch.device('cpu'), debug=False):
        self.pad_id = kg['pad_id']

        self.max_length = max_length

        self.device = device
        self.debug = debug

    def __call__(self, data_batch):
        user_batch = []
        user_entity_batch = []

        for data in data_batch:
            user_batch.append(data['user_id'])
            user_entity_batch.extend(data['user_entity_id'])

        user_entity_ids = padded_tensor(
            user_entity_batch, pad_id=self.pad_id, pad_tail=True, device=self.device,
            max_length=self.max_length, debug=self.debug
        )
        user_entity_mask = torch.ne(user_entity_ids, self.pad_id)

        batch = {
            'user_id': user_batch,
            'entity': {
                'user_entity_ids': user_entity_ids,
                'user_entity_mask': user_entity_mask,
            }
        }
        return batch


class MetaPathDataCollator:
    def __call__(self, data_batch):
        meta_path_mask_batch = []

        for data in data_batch:
            meta_path_mask_batch.append(data['meta_path_mask'])

        batch = {'meta_path_mask': meta_path_mask_batch}
        return batch


class FlowDataCollator:
    def __init__(self, kg, max_length=100, device=torch.device('cpu'), debug=False):
        self.device = device
        self.debug = debug

        self.entity2id = kg['entity2id']
        self.pad_id = kg['pad_id']
        self.type_ids = kg['type_ids']

        self.max_length = max_length

    def __call__(self, data_batch, meta_paths):
        input_batch = []
        meta_path_batch = []
        type_mask_batch = []

        for data, meta_path in zip(data_batch, meta_paths):
            meta_path_batch.append(meta_path)
            input_batch.append(data['input'] + meta_path)

            meta_path_tensor = torch.as_tensor(meta_path).reshape(-1, 1)
            type_mask = torch.eq(self.type_ids, meta_path_tensor)
            type_mask_batch.append(type_mask)

        input_ids = padded_tensor(
            input_batch, pad_id=self.pad_id, pad_tail=True, max_length=self.max_length,
            device=self.device, debug=self.debug
        )
        attention_mask = input_ids.ne(self.pad_id)

        batch = {
            'inputs': {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        }

        logits_mask = torch.ones(
            (len(data_batch), max(map(len, type_mask_batch)), len(self.entity2id)),
            dtype=torch.bool, device=self.device
        )
        for i, type_mask in enumerate(type_mask_batch):
            logits_mask[i, :len(type_mask)] = type_mask
        batch['inputs']['decoder_logits_mask'] = logits_mask
        batch['meta_path'] = meta_path_batch

        return batch


class TemplateDataCollator:
    def __call__(self, data_batch):
        text_batch = []
        template_pos_batch = []

        for data in data_batch:
            text_batch.append(data['template'])

            template_pos_list = []
            for i, template in enumerate(data['template']):
                template_pos_list.extend([i] * len(re.findall(r'<mask>', template)))
            template_pos_batch.append(template_pos_list)

        batch = {
            'text': text_batch,
            'position': template_pos_batch
        }
        return batch
