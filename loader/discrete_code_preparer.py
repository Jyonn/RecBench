import hashlib
import os

import pandas as pd
from pigmento import pnt
from tqdm import tqdm

from loader.code_preparer import CodePreparer
from loader.discrete_code_dataset import DiscreteCodeDataset
from loader.code_map import CodeMap as Map
from loader.token_vocab import TV
from model.base_discrete_code_model import BaseDiscreteCodeModel
from utils.code import get_code_indices


class DiscreteCodePreparer(CodePreparer):
    DATASET_CLASS = DiscreteCodeDataset

    model: BaseDiscreteCodeModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        code_indices, _, _ = get_code_indices(self.conf.code_path)

        self.processor.load()
        self.code_indices = dict()
        self.code_tree = dict()
        self.code_map = []

        item_indices = self.processor.items[self.processor.IID_COL]
        for item_index in item_indices:
            current_indices = code_indices[item_index]
            self.code_indices[item_index] = current_indices
            current_node = self.code_tree
            for index in current_indices:
                if index not in current_node:
                    current_node[index] = dict()
                current_node = current_node[index]

            for i, index in enumerate(current_indices):
                if i == len(self.code_map):
                    self.code_map.append(set())
                self.code_map[i].add(index)

        self.test_datapath = os.path.join(self.store_dir, 'test.parquet')
        self.test_has_generated = os.path.exists(self.test_datapath)

        pnt(f'prepared data will be stored in {self.store_dir}')

    def get_secondary_meta(self):
        return dict(
            code_path=self.conf.code_path,
        )

    def get_secondary_signature(self):
        meta = self.get_secondary_meta()
        keys = sorted(meta.keys())
        key = '-'.join([f'{k}={meta[k]}' for k in keys])
        md5 = hashlib.md5(key.encode()).hexdigest()
        return md5[:6] + f'@{self.conf.valid_ratio}'

    def tokenize_items(self, source='finetune', item_attrs=None):
        return self.code_indices

    def load_datalist(self, source='finetune'):
        return self._process(source=source)

    def load_or_generate(self, mode='train'):
        if mode == 'test':
            if self.test_has_generated:
                pnt(f'loading prepared {mode} data on {self.processor.get_name()} dataset')
                return self._pack_datalist(pd.read_parquet(self.test_datapath))
            else:
                test_datalist = self._process(source='test')
                test_datalist = pd.DataFrame(test_datalist)
                test_datalist.to_parquet(self.test_datapath)
                return self._pack_datalist(test_datalist)

        return super().load_or_generate(mode)

    def generate_item_alignment_data(self):
        prefix, item_ = self.model.get_item_alignment_tokens()

        datalist = []
        max_sequence_len = 0

        for _, item in tqdm(self.processor.items.iterrows()):
            content = self.processor.organize_item(item, item_attrs=self.processor.default_attrs, item_self=True)
            content = self.model.generate_simple_input_ids(content)
            content = content[:self.model.max_len // 5]

            code = self.code_indices[item[self.processor.IID_COL]]

            input_ids = prefix + content + item_
            vocab_ids = [TV.LLM] * len(input_ids) + [TV.COD] * len(code)
            input_ids += code

            max_sequence_len = max(max_sequence_len, len(input_ids))
            datalist.append({
                Map.IPT_COL: input_ids,
                Map.VOC_COL: vocab_ids,
                Map.LBL_COL: 0,
                Map.UID_COL: -1,
                Map.IID_COL: item[self.processor.IID_COL],
                Map.LBW_COL: 0,
            })

        for data in datalist:
            data[Map.LEN_COL] = len(data[Map.IPT_COL])
            data[Map.IPT_COL] = data[Map.IPT_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.VOC_COL] = data[Map.VOC_COL] + [0] * (max_sequence_len - data[Map.LEN_COL])
            data[Map.UID_COL] = self.uid_vocab.append(data[Map.UID_COL])
            data[Map.IID_COL] = self.iid_vocab.append(data[Map.IID_COL])

        pnt(f'{self.processor.get_name()} dataset: additional item alignment data max_sequence_len: {max_sequence_len}')

        return pd.DataFrame(datalist)
