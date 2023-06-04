import os
import json
import torch
from loguru import logger


class KG:
    def __init__(self, dataset, debug=False):
        self.debug = debug

        dataset_dir = f"../../data/{dataset}"
        with open(os.path.join(dataset_dir, 'kg.json'), encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(dataset_dir, 'entity2id.json'), encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(dataset_dir, 'relation2id.json'), encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(dataset_dir, 'item_ids.json'), encoding='utf-8') as f:
            self.item_ids = json.load(f)
        with open(os.path.join(dataset_dir, 'entity2type.json'), encoding='utf-8') as f:
            self.entity2type = json.load(f)
        with open(os.path.join(dataset_dir, 'type2id.json'), encoding='utf-8') as f:
            self.type2id = json.load(f)
        with open(os.path.join(dataset_dir, 'id2metapath.json'), encoding='utf-8') as f:
            self.id2metapath = json.load(f)

        self._prepare_kg()

    def _prepare_kg(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) in self.entity_kg:
                for relation_and_tail in self.entity_kg[str(entity)]:
                    edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                    edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge = torch.tensor(list(edge_list), dtype=torch.long)

        self.edge_index = edge[:, :2].t()  # (2, n_edge)
        self.edge_type = edge[:, 2]  # (n_edge)
        self.num_relations = len(self.relation2id)
        self.num_entities = len(self.entity2id)

        self.entity2id['<pad>'] = len(self.entity2id)
        self.entity2id['<bos>'] = len(self.entity2id)
        self.entity2id['<sep>'] = len(self.entity2id)
        self.entity2id['<eos>'] = len(self.entity2id)
        self.entity2id['<user>'] = len(self.entity2id)
        self.entity2id['<bot>'] = len(self.entity2id)
        self.id2entity = {int(idx): entity for entity, idx in self.entity2id.items()}
        self.num_special_tokens = len(self.entity2id) - self.num_entities

        self.entityid2typeid = {}
        for ent, type in self.entity2type.items():
            self.entityid2typeid[self.entity2id[ent]] = self.type2id[type]
        for type, idx in self.type2id.items():
            self.entityid2typeid[self.entity2id[type]] = idx
        self.type2id['user'] = len(self.type2id)
        self.type2id['special_token'] = len(self.type2id)
        self.id2type = {idx: typ for typ, idx in self.type2id.items()}
        self.entityid2typeid[self.entity2id['<user>']] = self.type2id['user']
        self.entityid2typeid[self.entity2id['<bot>']] = self.type2id['user']
        for special_token in ['<pad>', '<bos>', '<sep>', '<eos>']:
            self.entityid2typeid[self.entity2id[special_token]] = self.type2id['special_token']

        self.id2metapathid = {}
        for idx, metapath in self.id2metapath.items():
            metapath_id = [self.type2id[typ] for typ in metapath]
            self.id2metapathid[int(idx)] = metapath_id

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {len(self.entity2id)}, #item: {len(self.item_ids)}'
                # f'#sim_mat: {self.sim_mat.shape}'
            )

    def get_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'num_special_tokens': self.num_special_tokens,
            'item_ids': self.item_ids,
            'entityid2typeid': self.entityid2typeid,
            'id2type': self.id2type,
            'id2metapathid': self.id2metapathid,
            'id2metapath': self.id2metapath
        }
        return kg_info
