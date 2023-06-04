import json
import os

import torch
from loguru import logger


class KGForUniCRS:
    def __init__(self, kg, debug=False):
        self.debug = debug

        self.dataset_dir = f'../../data/{kg}'
        with open(os.path.join(self.dataset_dir, 'kg.json'), 'r', encoding='utf-8') as f:
            self.kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'relation2id.json'), 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self._process()

    def _process(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) not in self.kg:
                continue
            for relation_and_tail in self.kg[str(entity)]:
                edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge_list = list(edge_list)

        edge = torch.as_tensor(edge_list, dtype=torch.long)
        self.edge_index = edge[:, :2].t()
        self.edge_type = edge[:, 2]
        self.num_relations = len(self.relation2id)
        self.pad_id = self.entity2id['<pad>'] = len(self.entity2id)
        self.id2entity = {idx: ent for ent, idx in self.entity2id.items()}
        self.num_entities = len(self.entity2id)

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {self.num_entities}, #item: {len(self.item_ids)}'
            )

    def get_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'pad_entity_id': self.pad_id,
            'item_ids': self.item_ids,
        }
        return kg_info
