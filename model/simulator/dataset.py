import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import padded_tensor, simple_collate


class FlowDataset(Dataset):
    def __init__(
        self, dataset, split, entity2id, entityid2typeid, id2metapathid, max_length=100, user_entity_max_length=100, debug=False,
        shot=1, compute_loss_for_meta=False
    ):
        super(FlowDataset, self).__init__()
        self.debug = debug

        self.entity2id = entity2id
        self.user_idx = self.entity2id['<user>']
        self.bot_idx = self.entity2id['<bot>']

        self.entityid2typeid = entityid2typeid

        self.id2metapathid = id2metapathid
        self.id2metapathlen = {int(idx): len(meta_path) for idx, meta_path in self.id2metapathid.items()}

        self.max_length = max_length
        self.max_length -= 4

        self.user_entity_max_length = user_entity_max_length

        self.prefix = [self.user_idx, self.bot_idx, self.entity2id['<sep>']]

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self._prepare_data(data_file, shot, compute_loss_for_meta)

    def _prepare_data(self, data_file, shot, compute_loss_for_meta=False):
        with open(data_file, encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:5120]
            if shot < 1:
                line_idx = random.sample(range(len(lines)), int(len(lines) * shot))
                lines = [lines[idx] for idx in line_idx]
            elif shot > 1:
                lines = random.sample(lines, min(int(shot), len(lines)))

            for line in tqdm(lines):
                line = json.loads(line)
                users = line['user']
                user, bot = users

                path = []
                for ent in line['flow']:
                    if ent == user:
                        # path.append(self.user_idx)
                        continue
                    elif ent == bot:
                        # path.append(self.bot_idx)
                        continue
                    else:
                        path.append(self.entity2id[ent])
                path = path[:self.max_length] + [self.entity2id['<eos>']]

                meta_path = [self.entityid2typeid[ent_id] for ent_id in path]

                user2entity = line['user2entity']
                user_entity_ids = [[self.entity2id[ent] for ent in user2entity[user] if self.entity2id[ent] in path] for
                                   user in users]

                user_entity_num = sum(map(len, user_entity_ids))
                meta_path_len_req = user_entity_num + 1
                meta_path_mask = [False] * len(self.id2metapathlen)
                flag = False
                for idx, meta_path_len in enumerate(self.id2metapathlen.values()):
                    if meta_path_len == meta_path_len_req:
                        flag = True
                        meta_path_mask[idx] = True
                if flag is False:
                    meta_path_mask = [True] * len(self.id2metapathlen)

                data = {
                    'user_id': users,
                    'user_entity_id': user_entity_ids,
                    'input': self.prefix,
                    'meta_path': meta_path,
                    'meta_path_mask': meta_path_mask,
                    'label': path,
                }

                if compute_loss_for_meta:
                    if 'meta_path_label' not in line:
                        continue
                    else:
                        data['meta_path_label'] = line['meta_path_label']

                self.data.append(data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class UserDataCollator:
    def __init__(self, entity2id, max_length=100, device=torch.device('cpu'), debug=False):
        self.entity2id = entity2id
        self.pad_id = self.entity2id['<pad>']

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
    def __init__(self, device=torch.device('cpu'), debug=False):
        self.debug = debug
        self.device = device

    def __call__(self, data_batch, infer=False):
        meta_path_mask_batch = []
        label_batch = []

        for data in data_batch:
            meta_path_mask_batch.append(data['meta_path_mask'])
            if infer is False:
                label_batch.append(data['meta_path_label'])

        batch = {'meta_path_mask': meta_path_mask_batch}
        if infer is False:
            labels = torch.as_tensor(label_batch, device=self.device)
            batch['labels'] = labels

        return batch


class FlowDataCollator:
    def __init__(self, entity2id, entityid2typeid, use_meta_path=True, max_length=100, device=torch.device('cpu'), debug=False):
        self.device = device
        self.debug = debug

        self.entity2id = entity2id
        self.pad_id = self.entity2id['<pad>']

        self.entityid2typeid = entityid2typeid
        self.type_ids = torch.tensor([self.entityid2typeid[idx] for idx in self.entity2id.values()]).unsqueeze(0)

        self.max_length = max_length

        self.use_meta_path = use_meta_path

    def __call__(self, data_batch, gen=False):
        input_batch = []
        meta_path_batch = []
        label_batch = []
        type_mask_batch = []

        for data in data_batch:
            meta_path = data['meta_path']
            meta_path_batch.append(meta_path)
            if self.use_meta_path:
                input_batch.append(data['input'] + meta_path)
            else:
                input_batch.append(data['input'])
            label_batch.append(data['label'])

            if gen:
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

        if gen:
            batch['meta_path'] = meta_path_batch
            batch['labels'] = label_batch
            if self.use_meta_path:
                logits_mask = torch.ones(
                    (len(data_batch), max(map(len, type_mask_batch)), len(self.entity2id)),
                    dtype=torch.bool, device=self.device
                )
                for i, type_mask in enumerate(type_mask_batch):
                    logits_mask[i, :len(type_mask)] = type_mask
                batch['inputs']['decoder_logits_mask'] = logits_mask
        else:
            label_ids = padded_tensor(
                label_batch, pad_id=-100, pad_tail=True, max_length=self.max_length,
                device=self.device, debug=self.debug
            )
            batch['inputs']['labels'] = label_ids

        return batch


if __name__ == '__main__':
    from dataset_kg import KG
    from pprint import pprint

    dataset = 'inspired_flow'
    split = 'train'
    debug = False
    gen = False

    kg = KG(dataset='inspired-meta', debug=debug).get_kg_info()
    entity2id = kg['entity2id']
    id2entity = kg['id2entity']
    entityid2typeid = kg['entityid2typeid']
    id2metapathid = kg['id2metapathid']
    user_idx, bot_idx = entity2id['<user>'], entity2id['<bot>']

    dataset = FlowDataset(
        dataset=dataset, split=split, debug=debug, compute_loss_for_meta=False,
        entity2id=entity2id, entityid2typeid=entityid2typeid, id2metapathid=id2metapathid,
        # max_length=10
    )
    for i in range(3):
        data = dataset[i]
        print(data)

        user, bot = data['user_id']
        for idx in data['input']:
            if idx == user_idx:
                print(user, end=' | ')
            elif idx == bot_idx:
                print(bot, end=' | ')
            else:
                print(id2entity[idx], end=' | ')
        print()
        for idx in data['label']:
            if idx == user_idx:
                print(user, end=' | ')
            elif idx == bot_idx:
                print(bot, end=' | ')
            else:
                print(id2entity[idx], end=' | ')
        print()
        print()

    meta_path_data_collator = MetaPathDataCollator(debug=debug)
    flow_data_collator = FlowDataCollator(entity2id=entity2id, entityid2typeid=entityid2typeid)
    user_data_collator = UserDataCollator(entity2id=entity2id)

    dataloader = DataLoader(dataset, batch_size=2, collate_fn=simple_collate)

    input_max_len, input_avg_len = 0, 0
    label_max_len, label_avg_len = 0, 0
    user_entity_max_len, user_entity_avg_len = 0, 0
    for batch in tqdm(dataloader):
        user_batch = user_data_collator(batch)
        flow_batch = flow_data_collator(batch, gen=gen)

        if debug:
            pprint(user_batch)
            pprint(flow_batch)
            break

        input_len = flow_batch['inputs']['input_ids'].shape[1]
        input_max_len = max(input_max_len, input_len)
        input_avg_len += input_len

        user_entity_len = user_batch['entity']['user_entity_ids'].shape[1]
        user_entity_max_len = max(user_entity_max_len, user_entity_len)
        user_entity_avg_len += user_entity_len

        # if gen:
        #     label_max_len = max(label_max_len, max(map(len, batch['labels'])))
        #     label_avg_len += sum(map(len, batch['labels'])) / len(batch['labels'])

    if debug is False:
        print(input_max_len, input_avg_len / len(dataloader))
        # print(label_max_len, label_avg_len / len(dataloader))
        print(user_entity_max_len, user_entity_avg_len / len(dataloader))
        # redial: (61, 30), (68, 25), (54, 32) -> (68, 32)
        # inspired: (48, 21), (45, 23), (52, 16) -> (52, 23)
