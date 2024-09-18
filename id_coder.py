import json

import pigmento
from UniTok import Vocab
from pigmento import pnt

from process.base_processor import BaseProcessor
from seq_process.base_seqprocessor import BaseSeqProcessor
from utils.config_init import ConfigInit
from utils.function import load_seq_processor

if __name__ == '__main__':
    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    configuration = ConfigInit(
        required_args=['data'],
        default_args=dict(
            slicer=-20,
            source='original',
        ),
        makedirs=[]
    ).parse()

    processor = load_seq_processor(configuration.data.lower())  # type: BaseSeqProcessor
    processor.load()

    item_vocab = processor.item_vocab  # type: dict

    id_code = dict()
    for item, index in item_vocab.items():
        id_code[item] = [index]

    json.dump(id_code, open(f'code/{processor.get_name()}.id.code', 'w'), indent=2)
