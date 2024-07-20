from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from model.base_model import BaseModel


class SentenceBertModel(BaseModel):
    KEY = 'efederici/sentence-bert-base'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model = SentenceTransformer(self.KEY)

    def embed(self, content) -> Optional[torch.Tensor]:
        return self.model.encode(content, convert_to_tensor=True)
