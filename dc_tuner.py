from typing import Type

import pigmento
from pigmento import pnt

from loader.class_hub import ClassHub
from loader.dense_code_preparer import DenseCodePreparer
from model.base_dense_code_model import BaseDenseCodeModel
from model.base_model import BaseModel
from tuner import Tuner
from utils.config_init import ConfigInit


class DenseCodeTuner(Tuner):
    PREPARER_CLASS = DenseCodePreparer

    def load_model(self):
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]  # type: Type[BaseModel]
            assert issubclass(model, BaseDenseCodeModel), f'{model} is not a subclass of BaseDenseCodeModel'
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device())
        raise ValueError(f'unknown model: {self.model}')


if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['model', 'train', 'valid', 'code_path'],
        default_args=dict(
            slicer=-20,
            gpu=None,
            valid_metric='GAUC',
            valid_ratio=0.1,
            batch_size=32,
            use_lora=True,
            lora_r=32,
            lora_alpha=128,
            lora_dropout=0.1,
            lr=0.00001,
            acc_batch=1,
            eval_interval=0,
            patience=2,
            tuner=None,
            init_eval=True,
        ),
        makedirs=[]
    ).parse()

    tuner = DenseCodeTuner(conf=configuration)
    tuner.run()
