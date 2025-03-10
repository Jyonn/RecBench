from model.base_dense_code_model import BaseDenseCodeModel
from model.opt_model import OPTModel
from utils.prompt import SIMPLE_SUFFIX, SIMPLE_SYSTEM


class OPTDenseCodeModel(BaseDenseCodeModel, OPTModel):
    PREFIX_PROMPT = SIMPLE_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16


class OPT1BDCModel(OPTDenseCodeModel):
    KEY = 'facebook/opt-1.3b'


class OPT350MDCModel(OPTDenseCodeModel):
    KEY = 'facebook/opt-350m'
