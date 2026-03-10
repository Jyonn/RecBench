import json
import os
from typing import Union, Optional, List, cast

import numpy as np
import pigmento
import torch
from oba import Obj
from pigmento import pnt

from loader.class_hub import ClassHub
from model.base_model import BaseModel
from process.base_processor import BaseProcessor
from service.base_service import BaseService
from service.claude_service import Claude21Service, Claude3Service
from service.gemini_service import GeminiService
from service.gpt_service import GPT4Service, GPT35Service
from utils import bars
from utils.auth import GPT_KEY, CLAUDE_KEY, GEMINI_KEY
from utils.config_init import ConfigInit
from utils.export import Exporter
from utils.function import load_processor, seeding
from utils.gpu import GPU
from utils.metrics import MetricPool
from utils.timer import Timer, StatusTimer


class Worker:
    def __init__(self, conf):
        self.conf = conf
        self.data = conf.data.lower()
        self.model = conf.model.replace('.', '').lower()

        self.type = conf.type
        assert self.type in ['prompt', 'embed'], f'Type {self.type} is not supported.'
        self.use_prompt = self.type == 'prompt'
        self.use_embed = self.type == 'embed'

        self.processor = load_processor(self.data)  # type: BaseProcessor
        self.processor.load()
        self.caller = self.load_model_or_service()  # type: Union[BaseService, BaseModel]
        self.use_service = isinstance(self.caller, BaseService)

        if self.conf.tuner is not None:
            self.conf.tuner = str(self.conf.tuner).replace('@', '')
            assert not self.use_service, 'Tuner is not supported for service.'
            self.sign = '@' + self.conf.tuner
            self.conf.tuner = os.path.join('tuning', self.model, self.conf.tuner + '.json')
            pnt(f'loading {self.sign} tuner')
            self.tuner_meta = Obj(json.load(open(self.conf.tuner)))
            required_args = ['use_lora', 'lora_r', 'lora_alpha', 'lora_dropout']
            for arg in required_args:
                assert arg in self.tuner_meta, f'{arg} is required in tuner configuration'
            tune_from = self.tuner_meta['tune_from'] or 0
            self.caller = cast(BaseModel, self.caller)
            self.caller.prepare_model_finetuning(self.tuner_meta, inference_mode=False, tune_from=tune_from)
            self.caller.load(self.conf.tuner.replace('.json', '.pt'))
        else:
            self.sign = ''

        self.log_dir = os.path.join('export', self.data)
        if self.use_embed:
            assert not self.use_service, 'Embedding is not supported for service.'
            assert self.conf.embed_func in ['last', 'pool']
            embed_suffix = '_embed'
            if self.conf.embed_func == 'pool':
                embed_suffix += '_pool'
            self.log_dir = os.path.join('export', self.data + embed_suffix)

        os.makedirs(self.log_dir, exist_ok=True)
        pigmento.add_log_plugin(os.path.join(self.log_dir, f'{self.model}{self.sign}.log'))
        self.exporter = Exporter(os.path.join(self.log_dir, f'{self.model}{self.sign}.dat'))

        if self.conf.rerun:
            self.exporter.reset()

        self.latency_timer = Timer(activate=False)

    def get_device(self):
        if self.conf.gpu is None:
            return GPU.auto_choose(torch_format=True)
        if self.conf.gpu == -1:
            pnt('manually choosing CPU device')
            return 'cpu'
        pnt(f'manually choosing {self.conf.gpu}-th GPU')
        return f'cuda:{self.conf.gpu}'

    def load_model_or_service(self):
        models = ClassHub.models()
        if self.model in models:
            model = models[self.model]
            pnt(f'loading {model.get_name()} model')
            return model(device=self.get_device()).post_init()

        services = [
            GPT35Service(auth=GPT_KEY), GPT4Service(auth=GPT_KEY),
            Claude21Service(auth=CLAUDE_KEY), Claude3Service(auth=CLAUDE_KEY),
            GeminiService(auth=GEMINI_KEY)
        ]
        for service in services:
            if service.get_name() == self.model:
                pnt(f'loading {service.get_name()} service')
                return service
        raise ValueError(f'unknown model/service: {self.model}')

    def test_prompt(self):
        input_template = """User behavior sequence: \n{0}\nCandidate item: {1}"""

        progress = 0
        if self.exporter.exist() and not self.conf.latency:
            responses = self.exporter.read(to_float=False)
            progress = len(responses)
            pnt(f'directly start from {progress}')

        for index, data in enumerate(bar := bars.TestBar()(self.processor.generate(
            slicer=self.conf.slicer,
            source=self.conf.source
        ), total=len(self.processor.get_source_set(self.conf.source)))):
            if index < progress:
                continue

            uid, iid, history, candidate, click = data

            response: Optional[str, float] = None

            self.latency_timer.run('test')
            for _ in range(5):
                for i in range(len(history)):
                    _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                    input_sequence = input_template.format('\n'.join(_history), candidate)
                    response = self.caller(input_sequence)
                    if response is not None:
                        break
                if response is not None:
                    break
                # find most long items in the history and select top 10% to cut off
                lengths = [len(history[i]) for i in range(len(history))]
                sorted_indices = np.argsort(lengths)[::-1].tolist()  # descending order
                for i in sorted_indices[:max(len(sorted_indices) // 10, 1)]:
                    history[i] = history[i][:max(len(history[i]) // 2, 10)]
                candidate = candidate[:max(len(candidate) // 2, 10)]

            if response is None:
                pnt(f'failed to get response for {index} ({uid}, {iid})', current=index + 1, count=len(self.processor.test_set))
                # self.exporter.save_progress(index)
                exit(0)

            self.latency_timer.run('test')

            if self.use_service:
                response = response.replace('\n', '').replace('\r', '')
            else:
                response = f'{response:.4f}'
            # pnt(f'click: {click}, response: {response}', current=index + 1, count=len(self.processor.test_set))
            bar.set_postfix_str(f'click: {click}, response: {response}')

            if not self.conf.latency:
                self.exporter.write(response)
            # self.exporter.save_progress(index + 1)
        # TqdmPrinter.deactivate()

    def test_embed(self):
        history_template = """User behavior sequence: \n{0}"""
        candidate_template = """Candidate item: {0}"""

        # progress = self.exporter.load_progress()
        # if progress > 0:
        #     pnt(f'directly start from {progress}')
        progress = 0
        if self.exporter.exist():
            responses = self.exporter.read()
            progress = len(responses)
            pnt(f'directly start from {progress}')

        item_dict = self.exporter.load_embed('item')
        user_dict = self.exporter.load_embed('user')

        # TqdmPrinter.activate()
        for index, data in enumerate(bar := bars.TestBar()(self.processor.generate(
                slicer=self.conf.slicer,
                source=self.conf.source,
                as_dict=self.caller.AS_DICT
        ), total=len(self.processor.get_source_set(self.conf.source)))):
            if index < progress:
                continue

            uid, iid, history, candidate, click = data

            user_embed: Optional[np.ndarray] = None
            if uid in user_dict:
                user_embed = user_dict[uid]
            else:
                if self.caller.AS_DICT:
                    user_embed = self.caller.embed(history)
                else:
                    for _ in range(5):
                        for i in range(len(history)):
                            _history = [f'({j + 1}) {history[i + j]}' for j in range(len(history) - i)]
                            history_sequence = history_template.format('\n'.join(_history), candidate)
                            user_embed = self.caller.embed(history_sequence)
                            if user_embed is not None:
                                break
                        if user_embed is not None:
                            break
                        # find most long items in the history and select top 10% to cut off
                        lengths = [len(history[i]) for i in range(len(history))]
                        sorted_indices = np.argsort(lengths)[::-1].tolist()  # descending order
                        for i in sorted_indices[:max(len(sorted_indices) // 10, 1)]:
                            history[i] = history[i][:max(len(history[i]) // 2, 10)]

                if user_embed is None:
                    pnt(f'failed to get user embeds for {index} ({uid}, {iid})')
                    # self.exporter.save_progress(index)
                    exit(0)
                user_dict[uid] = user_embed
                if index % 100 == 0:
                    self.exporter.save_embed('user', user_dict)

            item_embed: Optional[np.ndarray] = None
            if iid in item_dict:
                item_embed = item_dict[iid]
            else:
                if self.caller.AS_DICT:
                    item_embed = self.caller.embed([candidate])
                else:
                    for _ in range(5):
                        candidate_sequence = candidate_template.format(candidate)
                        item_embed = self.caller.embed(candidate_sequence)
                        if item_embed is None:
                            candidate = candidate[:len(candidate) // 2]
                            continue
                if item_embed is None:
                    pnt(f'failed to get item embeds for {index} ({uid}, {iid})')
                    # self.exporter.save_progress(index)
                    exit(0)
                item_dict[iid] = item_embed
                if index % 100 == 0:
                    self.exporter.save_embed('item', item_dict)

            th_item_embed = torch.tensor(item_embed)
            th_user_embed = torch.tensor(user_embed)

            # dot similarity
            score = torch.dot(th_item_embed, th_user_embed).item()
            bar.set_postfix_str(f'click: {click}, score: {score:.4f}')
            self.exporter.write(score)

    def evaluate(self):
        scores = self.exporter.read(from_convert=self.use_service)  # type: List[float]

        source_set = self.processor.get_source_set(self.conf.source)
        labels = source_set[self.processor.LBL_COL].values
        groups = source_set[self.processor.UID_COL].values

        pool = MetricPool.parse(self.conf.metrics.split('+'))
        results = pool.calculate(scores, labels, groups)
        for metric, value in results.items():
            pnt(f'{metric}: {value:.4f}')

        self.exporter.save_metrics(results)

    def evaluate_item(self):
        scores = self.exporter.read(from_convert=self.use_service)  # type: List[float]

        source_set = self.processor.get_source_set(self.conf.source)
        labels = source_set[self.processor.LBL_COL].values
        groups = source_set[self.processor.IID_COL].values

        item_groups = self.processor.group_item_by_frequency(num_groups=self.conf.num_groups)
        groups = [item_groups.get(g, 0) for g in groups]

        pool = MetricPool.parse(self.conf.metrics.split('+'))
        results = pool.calculate(scores, labels, groups)
        for metric, value in results.items():
            pnt(f'{metric}: {value:.4f}')

        self.exporter.save_metrics(results)

    def auto_convert(self):
        assert self.exporter.exist(), 'No response file found'
        responses = self.exporter.read(to_float=False)
        progress = len(responses)

        source_set = self.processor.get_source_set(self.conf.source)
        assert progress == len(source_set), f'{self.conf.source} is not fully tested'

        labels = self.exporter.read(to_float=False)  # type: List[str]
        # use upper string
        labels = [label.upper().strip() for label in labels]
        scores = []

        rule_counts = [0, 0, 0]
        for label in labels:
            # Rule 1: fully YES/NO detect
            if label in ['YES', 'NO']:
                scores.append(1 if label == 'YES' else 0)
                rule_counts[0] += 1
                continue
            # Rule 2: partially YES/NO detect
            if ('YES' in label) ^ ('NO' in label):
                scores.append(1 if 'YES' in label else 0)
                rule_counts[1] += 1
                continue
            # Rule 3: manually detect
            pnt(f'please manually convert: {label}')
            scores.append(int(input('Score (0/1): ')))
            rule_counts[2] += 1

        pnt(f'rule 1: {rule_counts[0]}, rule 2: {rule_counts[1]}, rule 3: {rule_counts[2]}')
        self.exporter.save_convert(scores)

    def latency(self):
        self.latency_timer.activate()
        self.latency_timer.clear()
        try:
            self.test_prompt()
        except KeyboardInterrupt:
            st = self.latency_timer.status_dict["test"]  # type: StatusTimer
            pnt(f'Total {st.count} steps, avg ms {st.avgms():.4f}')

    def run(self):
        if self.conf.latency:
            self.latency()
            return

        if self.use_prompt:
            self.test_prompt()
        else:
            self.test_embed()
        if self.use_service:
            self.auto_convert()
        self.evaluate()


if __name__ == '__main__':
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    pigmento.add_time_prefix()
    pnt.set_display_mode(
        use_instance_class=True,
        display_method_name=False
    )

    seeding(2024)

    configuration = ConfigInit(
        required_args=['data', 'model'],
        default_args=dict(
            slicer=-20,
            gpu=None,
            source='test',
            metrics='+'.join(['GAUC', 'NDCG@1', 'NDCG@5', 'MRR', 'F1', 'Recall@1', 'Recall@5']),
            type='prompt',
            tuner=None,
            rerun=False,
            embed_func='last',
            latency=False,
        ),
        makedirs=[]
    ).parse()

    worker = Worker(configuration)
    worker.run()
