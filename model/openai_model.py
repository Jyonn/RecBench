from model.lc_model import LongContextModel


class GPTOSS20BModel(LongContextModel):
    KEY = 'openai/gpt-oss-20b'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_len = 2_000
