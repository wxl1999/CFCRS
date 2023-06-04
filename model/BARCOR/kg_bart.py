import os
import json
import torch
from loguru import logger


class KGForBART:
    def __init__(self, kg_dataset, debug=False):
        self.debug = debug

        dataset_dir = f"../../data/{kg_dataset}"
        with open(os.path.join(dataset_dir, 'kg.json'), encoding='utf-8') as f:
            self.kg = json.load(f)
        with open(os.path.join(dataset_dir, 'entity2id.json'), encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(dataset_dir, 'item_ids.json'), encoding='utf-8') as f:
            self.item_ids = json.load(f)
        with open(os.path.join(dataset_dir, 'relation2id.json'), encoding='utf-8') as f:
            self.relation2id = json.load(f)

        self._prepare_kg()

    def _prepare_kg(self):
        edge_list = set()  # [(entity, entity, relation)]
        for entity in self.entity2id.values():
            if str(entity) in self.kg:
                for relation_and_tail in self.kg[str(entity)]:
                    edge_list.add((entity, relation_and_tail[1], relation_and_tail[0]))
                    edge_list.add((relation_and_tail[1], entity, relation_and_tail[0]))
        edge = torch.tensor(list(edge_list), dtype=torch.long)

        self.edge_index = edge[:, :2].T  # (2, n_edge)
        self.edge_type = edge[:, 2]  # (n_edge)
        self.num_relations = len(self.relation2id)
        self.num_entities = len(self.entity2id)

        self.pad_id = self.entity2id['<pad>'] = len(self.entity2id)

        self.id2entity = {idx: ent for ent, idx in self.entity2id.items()}

        if self.debug:
            logger.debug(
                f'#edge: {len(edge)}, #relation: {self.num_relations}, '
                f'#entity: {len(self.entity2id)}'
            )

    def get_kg_info(self):
        kg_info = {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'item_ids': self.item_ids,
            'pad_id': self.pad_id,
        }
        return kg_info
