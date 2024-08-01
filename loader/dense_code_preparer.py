import os.path
from typing import cast

import numpy as np
import torch
from UniTok import Vocab
from torch import nn

from loader.code_preparer import CodePreparer
from loader.dense_code_dataset import DenseCodeDataset
from loader.code_map import CodeMap as Map
from model.base_dense_code_model import BaseDenseCodeModel
from utils.code import get_code_embeds


class DenseCodePreparer(CodePreparer):
    DATASET_CLASS = DenseCodeDataset

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.code_embeds = get_code_embeds(self.conf.code_path)
        assert self.code_embeds is not None, f'code embeddings for {self.processor.get_name()} not found'

        self.cod_vocab = Vocab(name=Map.COD_COL)
        if os.path.exists(self.cod_vocab.get_store_path(self.store_dir)):
            self.cod_vocab.load(self.store_dir)

    def tokenize_items(self, source='finetune', item_attrs=None):
        item_set = self.processor.get_item_subset(source, slicer=self.conf.slicer)

        item_dict = dict()
        for iid in item_set:
            code_embed = self.code_embeds[iid]  # type: np.ndarray
            num_embeds = len(code_embed)
            item_dict[iid] = []
            for i in range(num_embeds):
                item_dict[iid].append(self.cod_vocab.append(f'{iid}_{i}'))

        self.cod_vocab.save(self.store_dir)
        return item_dict

    def generate_cod_embeddings(self):
        self.model = cast(BaseDenseCodeModel, self.model)
        cod_embeddings = []
        for key in self.cod_vocab:
            iid, num = key.split('_')
            cod_embeddings.append(self.code_embeds[iid][int(num)])
        cod_embeddings = np.array(cod_embeddings)
        cod_embeddings = torch.tensor(cod_embeddings, dtype=self.model.get_dtype())
        torch.save(cod_embeddings, os.path.join(self.store_dir, 'cod_embeds.pth'))
        # return nn.Embedding.from_pretrained(cod_embeddings, freeze=True)

    def load_datalist(self):
        self.generate_cod_embeddings()
        return self._process()

    def load_or_generate(self, mode='train'):
        output = super().load_or_generate(mode)

        cod_embeddings = torch.load(os.path.join(self.store_dir, 'cod_embeds.pth'))
        self.model.set_cod_embeddings(nn.Embedding.from_pretrained(cod_embeddings, freeze=True))

        return output
