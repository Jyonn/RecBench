import os

import pandas as pd

from process.base_uict_processor import UICTProcessor

import gzip
import json


class AmazonProcessor(UICTProcessor):
    UID_COL = 'reviewerID'
    IID_COL = 'asin'
    HIS_COL = 'history'
    LBL_COL = 'click'
    DAT_COL = 'reviewTime'
    RAT_COL = 'overall'

    POS_COUNT = 2
    MAX_LINES = 1e7

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def __init__(self, subset, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset

    @staticmethod
    def parse(path):
        f = gzip.open(path, 'rb')
        for line in f:
            yield json.loads(line)

    @classmethod
    def _load_data(cls, path):
        data = {}
        for index, d in enumerate(cls.parse(path)):
            data[index] = d
            if cls.MAX_LINES and index >= cls.MAX_LINES:
                break
        return pd.DataFrame.from_dict(data, orient='index')

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, f"meta_{self.subset}.json.gz")
        items = self._load_data(path)

        items = items[['asin', 'title']]
        # filter special symbol
        items['title'] = items['title'].str.replace(r'&#[0-9]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'&[a-zA-Z]+;', '', regex=True)
        items['title'] = items['title'].str.replace(r'[^\w\s]', '', regex=True)

        items = items.drop_duplicates(subset=self.IID_COL).reset_index(drop=True)
        return items

    def load_users(self) -> pd.DataFrame:
        item_set = set(self.items[self.IID_COL].unique())

        path = os.path.join(self.data_dir, f"{self.subset}.json.gz")
        interactions = self._load_data(path)

        interactions = interactions[[self.UID_COL, self.IID_COL, self.RAT_COL, self.DAT_COL]]

        interactions[self.DAT_COL] = pd.to_datetime(interactions[self.DAT_COL], format='%m %d, %Y')
        interactions = interactions[interactions[self.IID_COL].isin(item_set)]

        interactions[self.RAT_COL] = interactions[self.RAT_COL].astype(int)
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL].apply(lambda x: int(x >= 4))

        interactions = interactions.drop(columns=[self.RAT_COL])
        return self._load_users(interactions)
