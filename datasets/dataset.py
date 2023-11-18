from utils import make_directories


class Dataset(object):
    def __init__(self, data_dir):
        make_directories(data_dir)
        self.data_dir = data_dir

    def download(self):
        raise NotImplementedError


class UnlabeledDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__(data_dir)

    def download(self):
        raise NotImplementedError

    def get_texts(self):
        raise NotImplementedError
