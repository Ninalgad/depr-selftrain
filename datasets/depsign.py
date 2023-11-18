import wget
import os
import pandas as pd

from preprocess import clean
from datasets.dataset import Dataset


class DepSign2023(Dataset):
    dataset_urls = {
      'train.csv': "https://drive.google.com/u/0/uc?id=10qLuBGZaAMCxQysOFQs3rVW6O44nEavk&export=download",
      'dev.csv': "https://drive.google.com/u/0/uc?id=10qd0pSIGnwqWVSeAFfnCGTiXVd48VLKf&export=download",
      'test.csv': "https://docs.google.com/spreadsheets/d/1R_V5b0AhKn6T5ZJdVJXrk5hcS6UD6_mYC_lyHWJwwg0/export?format=csv&gid=552642328"
    }
    labels = ['moderate', 'not depression', 'severe']
    num_labels = 3

    def download(self):
        for name, url in self.dataset_urls.items():
            path = os.path.join(self.data_dir, name)
            if not os.path.isfile(path):
                wget.download(url, out=path)

    def get_data_splits(self):
        self.download()

        train_df = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        dev_df = pd.read_csv(os.path.join(self.data_dir, 'dev.csv'))
        test_df = pd.read_csv(os.path.join(self.data_dir, 'test.csv'))

        # align column names
        dev_df = dev_df.rename(columns={"Pid": "PID", "text data": "Text data", "Class labels": "Label"})
        test_df = test_df.rename(columns={"Pid": "PID"})

        # drop duplicates
        train_df['split'] = 'train'
        dev_df['split'] = 'dev'
        test_df['split'] = 'test'
        df = pd.concat([train_df, dev_df, test_df])
        df = df.drop_duplicates(subset=['Text data', 'Label'])

        # encode labels
        df["Label"] = df["Label"].map(lambda y: self.labels.index(y))

        # clean text
        df_clean = df.copy()
        df_clean["Text data"] = train_df["Text data"].map(clean)

        # drop na
        df_clean = df_clean.dropna(how='any')

        train_df = df_clean[df_clean.split == 'train']
        dev_df = df_clean[df_clean.split == 'dev']
        test_df = df_clean[df_clean.split == 'test']

        return train_df, dev_df, test_df
