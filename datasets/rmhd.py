import wget
import os
import pandas as pd
from glob import glob
from datasets.dataset import UnlabeledDataset
from preprocess import clean


def _get_name_from_url(url):
    return url.split('/').pop()


class RedditMentalHealthDataset(UnlabeledDataset):
    urls_filename = "depr-selftrain/datasets/rmhd.txt"
    urls_small_filename = "depr-selftrain/datasets/rmhd-small.txt"

    def __init__(self, data_dir, small=False):
        super().__init__(data_dir)
        self.small = small

    def download(self):
        filename = self.urls_filename
        if self.small:
            filename = self.urls_small_filename

        with open(filename, 'r') as f:
            rmhd_urls = f.readlines()

        for url in rmhd_urls:
            url = url.strip()
            csv_name = _get_name_from_url(url)
            path = os.path.join(self.data_dir, csv_name)
            if not os.path.isfile(path):
                wget.download(url, out=path)

    def get_texts(self):
        texts = []
        for filename in glob(os.path.join(self.data_dir, "*.csv")):

            # use single col for fast loading
            posts = pd.read_csv(filename, usecols=['post'])['post']

            # clean strings
            posts = posts.map(clean)

            texts += posts.tolist()

        return texts
