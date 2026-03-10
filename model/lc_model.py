import abc

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompt import CHAT_SYSTEM, SIMPLE_SUFFIX
from model.base_model import BaseModel
from utils.auth import HF_KEY


class LongContextModel(BaseModel, abc.ABC):
    PREFIX_PROMPT = CHAT_SYSTEM
    SUFFIX_PROMPT = SIMPLE_SUFFIX
    BIT = 16

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # use large size opt model
        # self.model = AutoModelForCausalLM.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY, torch_dtype=self.get_dtype())
        self.model = AutoModelForCausalLM.from_pretrained(
            self.key,
            **self.get_config(),
            trust_remote_code=True,
            token=HF_KEY
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.key, trust_remote_code=True, token=HF_KEY)
        self.max_len = 10_000

        self.yes_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('YES'))[0]
        self.no_token = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('NO'))[0]

    def _generate_input_ids(self, content):
        return self.tokenizer.encode(content, return_tensors='pt', add_special_tokens=False)


class GLM4TH9BModel(LongContextModel):
    KEY = 'THUDM/glm-4-9b-chat'
    NUM_LAYERS = 48


class Mistral7BModel(LongContextModel):
    KEY = 'mistralai/Mistral-7B-Instruct-v0.3'
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 1024


class Phi3TH7BModel(LongContextModel):
    KEY = 'microsoft/Phi-3-small-8k-instruct'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class Phi2TH3BModel(LongContextModel):
    KEY = 'microsoft/phi-2'
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.max_len = 2_000


class RecGPT7BModel(LongContextModel):
    KEY = 'vinai/RecGPT-7B'
    NUM_LAYERS = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = 2_000
