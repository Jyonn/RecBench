import os
import pandas as pd
from process.base_uict_processor import UICTProcessor


class MovieLensProcessor(UICTProcessor):
    IID_COL = 'movieId'
    UID_COL = 'userId'
    HIS_COL = 'his'
    LBL_COL = 'click'
    DAT_COL = 'timestamp'
    RAT_COL = 'rating'

    POS_COUNT = 2

    NUM_TEST = 20_000
    NUM_FINETUNE = 0

    REQUIRE_STRINGIFY = False

    @property
    def default_attrs(self):
        return ['title']

    def load_items(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, 'movies.csv')
        movies = pd.read_csv(
            filepath_or_buffer=path,
            sep=',',
            engine='python',
            encoding="ISO-8859-1",
        )
        movies['title'] = movies['title'].str.replace(r'[^A-Za-z0-9 ]+', '')
        return movies

    def load_users(self) -> pd.DataFrame:
        interactions = pd.read_csv(
            filepath_or_buffer=os.path.join(self.data_dir, 'ratings.csv'),
            sep=',',
            engine='python'
        )

        # filter out rating = 3
        interactions = interactions[interactions[self.RAT_COL] != 3]
        interactions[self.LBL_COL] = interactions[self.RAT_COL] > 3
        interactions.drop(columns=[self.RAT_COL], inplace=True)

        return self._load_users(interactions)
