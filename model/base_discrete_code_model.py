from typing import Optional

import torch
from torch import nn

from loader.dense_code_map import DenseCodeMap as Map
from model.base_dense_code_model import DenseCodeEmbeddingLayer, BaseDenseCodeModel
from model.base_model import BaseModel


class DiscreteCodeEmbeddingLayer(DenseCodeEmbeddingLayer):
    def __init__(self, num_codes: int, dtype, **kwargs):
        super().__init__(**kwargs)

        self.cod_embeddings = nn.Embedding(num_codes, self.embedding_dim, dtype=dtype)
        self.cod_embeddings.weight.requires_grad = True

        self.cod_classifier = nn.Linear(self.embedding_dim, num_codes, bias=False, dtype=dtype)

    def classify(self, embeds):
        return self.cod_classifier(embeds)


class BaseDiscreteCodeModel(BaseDenseCodeModel):
    def __init__(self, num_codes, **kwargs):
        super().__init__(**kwargs)
        self.embedding_layer: Optional[DiscreteCodeEmbeddingLayer] = None
        self.embedding_dim = self.get_token_embeddings().weight.shape[1]
        self.num_codes = num_codes

    def post_init(self):
        BaseModel.post_init(self)

        self.embedding_layer = DiscreteCodeEmbeddingLayer(
            llm_embeddings=self.get_token_embeddings(),
            device=self.device,
            num_codes=self.num_codes,
            dtype=self.get_dtype(),
        )
        self.embedding_layer.to(self.device)

    def set_cod_embeddings(self, cod_embeddings):
        raise AttributeError('set_cod_embeddings is not supported in DiscreteCodeModel')

    def _get_scores(self, batch):
        # print(batch)

        output = self.embedding_layer(batch)
        input_embeddings = output['input_embeddings']
        attention_mask = output['attention_mask']
        cod_input = output['cod_input']  # [B, L]
        cod_mask = output['cod_mask']  # [B, L]

        # print(attention_mask)
        # print(cod_input)
        # print(cod_mask)

        output = self.model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = output.logits  # [B, L, V]
        indices = (batch[Map.LEN_COL] - 1).view(-1, 1, 1).expand(-1, 1, logits.size(-1)).to(self.device)
        logits = torch.gather(logits, 1, indices).squeeze(1)  # [B, V]
        logits = logits[:, self.label_tokens]  # [B, 2]
        scores = self.softmax_sft(logits)  # [B, 2]

        states = output.hidden_states[-1]  # [B, L, D]

        logits = self.embedding_layer.classify(states)  # [B, L, C]
        # left shift code input and mask to construct the target
        cod_input = torch.roll(cod_input, -1, 1)  # [B, L]
        cod_mask = torch.roll(cod_mask, -1, 1)  # [B, L]
        cod_labels = torch.ones(cod_input.shape, dtype=torch.long, device=self.device) * -100
        cod_labels[cod_mask] = cod_input[cod_mask]

        # print(cod_labels)
        # exit(0)

        # calculate the loss
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), cod_labels.view(-1))

        return scores[:, 1], loss

    def finetune(self, batch):
        scores, cod_loss = self._get_scores(batch)
        rec_loss = self.loss_fct(scores.float(), batch[Map.LBL_COL].to(self.device).float())
        return rec_loss + cod_loss

    def evaluate(self, batch):
        scores, cod_loss = self._get_scores(batch)  # [B, V=30522]
        return scores.detach().cpu().tolist()
