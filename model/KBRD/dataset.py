from torch.utils.data import Dataset
from utils import sample_data


class DatasetForRec(Dataset):
    def __init__(self, data_list, entity2id, max_length=100, debug=False, shot=1):
        super(DatasetForRec, self).__init__()

        self.entity2id = entity2id
        self.max_length = max_length

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self._prepare_data(data_list)

    def _prepare_data(self, data_list):
        for data in data_list:
            for rec in data['rec']:
                if rec in self.entity2id:
                    data_dict = {
                        'entity': [self.entity2id[ent] for ent in data['entity'][-self.max_length:] if ent in self.entity2id],
                        'item': self.entity2id[rec],
                    }
                    if 'dialog_id' in data:
                        data_dict['dialog_id'] = data['dialog_id']
                    if 'turn_id' in data:
                        data_dict['turn_id'] = data['turn_id']
                    if 'template' in data:
                        data_dict['template'] = data['template']
                    self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForConv(Dataset):
    def __init__(
        self, data_list, tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None, resp_max_length=None, entity_max_length=100
    ):
        super(DatasetForConv, self).__init__()
        self.tokenizer = tokenizer
        self.debug = debug

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.entity_max_length = entity_max_length

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self._prepare_data(data_list)

    def _prepare_data(self, data_list):
        for data in data_list:
            self.tokenizer.truncation_side = 'left'
            context_list = data['context']
            context = f'{self.tokenizer.sep_token}'.join(context_list)
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)

            self.tokenizer.truncation_side = 'right'
            resp = data['resp']
            resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)

            entity_list = [
                self.entity2id[ent] for ent in data['entity'][-self.entity_max_length:] if ent in self.entity2id
            ]

            data_dict = {
                # 'dialog_id': data['dialog_id'],
                # 'turn_id': data['turn_id'],
                'context': context_ids,
                'entity': entity_list,
                'resp': resp_ids
            }
            if 'dialog_id' in data:
                data_dict['dialog_id'] = data['dialog_id']
            if 'turn_id' in data:
                data_dict['turn_id'] = data['turn_id']

            self.data_list.append(data_dict)

    def __getitem__(self, item):
        return self.data_list[item]

    def __len__(self):
        return len(self.data_list)


class DatasetForFlow(Dataset):
    def __init__(
        self, data_list, kg, max_length=100, debug=False, shot=1,
    ):
        super(DatasetForFlow, self).__init__()

        self.entity2id = kg['entity2id']
        self.user_id = kg['user_id']
        self.bot_id = kg['bot_id']
        self.sep_id = kg['sep_id']
        self.eos_id = kg['eos_id']

        self.entityid2typeid = kg['entityid2typeid']
        self.type_ids = kg['type_ids']

        self.id2metapathlen = kg['id2metapathlen']

        # self.max_length = max_length - 4

        self.prefix = [self.user_id, self.bot_id, self.sep_id]

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self._prepare_data(data_list)

    def _prepare_data(self, data_list):
        for data in data_list:
            users = data['user']

            user2entity = data['user2entity']
            user_entity_ids = [[self.entity2id[ent] for ent in user2entity[user]] for
                               user in users]

            user_entity_num = sum(map(len, user_entity_ids))
            meta_path_len_req = user_entity_num + 1
            meta_path_mask = [False] * len(self.id2metapathlen)
            flag = False
            for idx, meta_path_len in self.id2metapathlen.items():
                if meta_path_len == meta_path_len_req:
                    flag = True
                    meta_path_mask[idx] = True
            if flag is False:
                meta_path_mask = [True] * len(self.id2metapathlen)

            data_dict = {
                'user_id': users,
                'user_entity_id': user_entity_ids,
                'input': self.prefix,
                'meta_path_mask': meta_path_mask,
                'template': data['template'],
            }
            if 'dialog_id' in data:
                data_dict['dialog_id'] = data['dialog_id']
            if 'turn_id' in data:
                data_dict['turn_id'] = data['turn_id']

            self.data_list.append(data_dict)

    def __getitem__(self, item):
        return self.data_list[item]

    def __len__(self):
        return len(self.data_list)
