from torch.utils.data import Dataset
from utils import sample_data


class DatasetForRec(Dataset):
    def __init__(
        self, data_list, tokenizer, prompt_tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
    ):
        super(DatasetForRec, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length

        self.entity2id = entity2id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
        for data in data_list:
            if len(data['rec']) == 0:
                continue
            # if len(dialog['context']) == 1 and dialog['context'][0] == '':
            #     continue

            text_list = []
            turn_idx = 0
            for utt in data['context']:
                if utt != '':
                    text = ''
                    if turn_idx % 2 == 0:
                        text += 'User: '
                    else:
                        text += 'System: '
                    text += utt
                    text_list.append(text)
                turn_idx += 1
            context = f'{self.tokenizer.eos_token}'.join(text_list)
            context += f'{self.tokenizer.eos_token}'
            prompt_context = f'{self.prompt_tokenizer.sep_token}'.join(text_list)

            # context = ''
            # prompt_context = ''
            # for i, utt in enumerate(data['context']):
            #     if utt == '':
            #         continue
            #     if i % 2 == 0:
            #         context += 'User: '
            #         prompt_context += 'User: '
            #     else:
            #         context += 'System: '
            #         prompt_context += 'System: '
            #     context += utt
            #     context += self.tokenizer.eos_token
            #     prompt_context += utt
            #     prompt_context += self.prompt_tokenizer.sep_token

            self.tokenizer.truncation_side = 'left'
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
            # context_ids = context_ids[-self.context_max_length:]
            # context_ids.append(self.tokenizer.eos_token_id)

            self.prompt_tokenizer.truncation_side = 'left'
            prompt_ids = self.prompt_tokenizer.encode(prompt_context, truncation=True,
                                                      max_length=self.context_max_length)
            # prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
            # prompt_ids = prompt_ids[-self.prompt_max_length:]
            # prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)
            # prompt_ids.append(self.prompt_tokenizer.sep_token_id)

            if turn_idx % 2 == 0:
                user_str = 'User: '
            else:
                user_str = 'System: '

            self.tokenizer.truncation_side = 'right'
            resp = user_str + data['resp'] + f'{self.tokenizer.eos_token}'
            resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)
            # resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))
            # resp_ids = resp_ids[:self.resp_max_length]
            # resp_ids.append(self.tokenizer.eos_token_id)

            self.prompt_tokenizer.truncation_side = 'right'
            prompt_resp = user_str + data['resp'] + f'{self.prompt_tokenizer.sep_token}'
            prompt_resp_ids = self.prompt_tokenizer.encode(
                prompt_resp, truncation=True, max_length=self.resp_max_length, add_special_tokens=False
            )
            # prompt_resp_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))
            # prompt_resp_ids = prompt_resp_ids[:self.resp_max_length]
            # prompt_resp_ids.append(self.prompt_tokenizer.sep_token_id)

            # context += resp + self.tokenizer.eos_token
            # prompt_context += resp + self.prompt_tokenizer.sep_token

            for rec in data['rec']:
                if rec in self.entity2id:
                    data_dict = {
                        'context': context_ids,
                        'prompt': prompt_ids,
                        'entity': [self.entity2id[ent] for ent in data['entity'][-self.entity_max_length:] if ent in self.entity2id],
                        'rec': self.entity2id[rec],
                        'resp': resp_ids,
                        'prompt_resp': prompt_resp_ids,
                    }
                    if 'template' in data:
                        data_dict['template'] = data['template']
                    self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForConv(Dataset):
    def __init__(
        self, data_list, tokenizer, prompt_tokenizer, entity2id, debug=False, shot=1,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
    ):
        super(DatasetForConv, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length
        self.resp_max_length -= 1

        self.entity2id = entity2id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self.prepare_data(data_list)

    def prepare_data(self, data_list):
        for data in data_list:
            text_list = []
            turn_idx = 0
            for utt in data['context']:
                if utt != '' and len(utt) > 0:
                    text = ''
                    if turn_idx % 2 == 0:
                        text += 'User: '
                    else:
                        text += 'System: '
                    text += utt
                    text_list.append(text)
                turn_idx += 1
            if len(text_list) == 0:
                continue

            context = f'{self.tokenizer.eos_token}'.join(text_list)
            context += f'{self.tokenizer.eos_token}'
            prompt_context = f'{self.prompt_tokenizer.sep_token}'.join(text_list)

            self.tokenizer.truncation_side = 'left'
            context_ids = self.tokenizer.encode(context, truncation=True, max_length=self.context_max_length)
            # context_ids.append(self.tokenizer.eos_token_id)

            self.prompt_tokenizer.truncation_side = 'left'
            prompt_ids = self.prompt_tokenizer.encode(prompt_context, truncation=True,
                                                      max_length=self.context_max_length)

            self.tokenizer.truncation_side = 'right'
            if turn_idx % 2 == 0:
                user_str = 'User: '
            else:
                user_str = 'System: '
            resp = user_str + data['resp']
            resp_ids = self.tokenizer.encode(resp, truncation=True, max_length=self.resp_max_length)
            resp_ids.append(self.tokenizer.eos_token_id)

            # self.prompt_tokenizer.truncation_side = 'right'
            # prompt_resp = user_str + data['resp'] + f'{self.prompt_tokenizer.sep_token}'
            # prompt_resp_ids = self.prompt_tokenizer.encode(
            #     prompt_resp, truncation=True, max_length=self.resp_max_length, add_special_tokens=False
            # )

            entity_list = [
                self.entity2id[ent] for ent in data['entity'][-self.entity_max_length:] if ent in self.entity2id
            ]

            data_dict = {
                'context': context_ids,
                'prompt': prompt_ids,
                'entity': entity_list,
                'resp': resp_ids,
                # 'prompt_resp': prompt_resp_ids,
            }
            # if 'template' in data:
            #     data_dict['template'] = data['template']
            self.data_list.append(data_dict)

    def __getitem__(self, ind):
        return self.data_list[ind]

    def __len__(self):
        return len(self.data_list)


class DatasetForFlow(Dataset):
    def __init__(
        self, data_list, kg, max_length=100, user_entity_max_length=100, debug=False, shot=1,
    ):
        super(DatasetForFlow, self).__init__()

        self.entity2id = kg['entity2id']
        self.user_id = kg['user_id']
        self.bot_id = kg['bot_id']
        self.sep_id = kg['sep_id']
        self.eos_id = kg['eos_id']

        self.entityid2typeid = kg['entityid2typeid']
        self.item_type_id = kg['type2id']['movie']

        self.id2metapathid = kg['id2metapathid']
        self.id2metapathlen = kg['id2metapathlen']

        # self.max_length = max_length
        # self.max_length -= 4

        # self.user_entity_max_length = user_entity_max_length

        self.prefix = [self.user_id, self.bot_id, self.sep_id]

        self.data_list = []
        data_list = sample_data(data_list, shot=shot, debug=debug)
        self._prepare_data(data_list)

    def _prepare_data(self, data_list):
        for data in data_list:
            users = data['user']
            user, bot = users

            # path = []
            # for ent in data['flow']:
            #     if ent == user:
            #         # path.append(self.user_idx)
            #         continue
            #     elif ent == bot:
            #         # path.append(self.bot_idx)
            #         continue
            #     else:
            #         path.append(self.entity2id[ent])
            # path = path[:self.max_length] + [self.entity2id['<eos>']]

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
