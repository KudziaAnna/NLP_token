from lib2to3.pgen2.tokenize import tokenize
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset

SEED = 1234
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_key(word_dict, val):
    for key, value in word_dict.items():
         if val == value:
             return key
 
    raise Exception("Key doesn't exists.")

def prepare_dict(data_dir):
    word_dict = {
        "<pad>" : 0,
        "<sos>" : 1,
        "<eos>" : 2,
        "<sot>" : 3,
        "<eot>" : 4,
    }
    longest_sentence = 2 # include added sos i eos later
    i = 2
    with open(data_dir) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if line != "\n":
                i += 3 # include added sot i eot
                if tmp[1] not in word_dict:
                    word_dict[tmp[1]] = len(word_dict)
            else:
                if longest_sentence < i:
                    longest_sentence = i
                i = 0
    return word_dict, longest_sentence

def read_data(data_dir):
    word_dict, longest_sentence = prepare_dict(data_dir)
    group_sent = []
    group_tok_sent = []
    tokenized_single_sent = [word_dict["<sos>"]]
    single_sent = [word_dict["<pad>"]]
    input_data = []
    target_data = []

    with open(data_dir) as f:
        lines = f.readlines()
        i = 0
        for line in lines:
            tmp = line.split()

            if line == "\n":
                tokenized_single_sent.append(word_dict["<eos>"])
                while len(single_sent) < longest_sentence:
                    single_sent.append(word_dict["<pad>"])

                while len(tokenized_single_sent) < longest_sentence:
                    tokenized_single_sent.append(word_dict["<pad>"])

                group_sent += single_sent
                group_tok_sent += tokenized_single_sent
                tokenized_single_sent = [word_dict["<sos>"]]
                single_sent = [word_dict["<pad>"]]
                i  += 1

                if i % 2 == 0:

                    if len(group_sent) != 684 or len(group_tok_sent) != 684:
                        print(len(group_sent))
                    else:
                        input_data.append(np.array(group_sent, dtype=np.float).reshape(1, len(group_sent)))
                        target_data.append(np.array(group_tok_sent, dtype=np.float).reshape(1, len(group_tok_sent)))

                    group_sent = []
                    group_tok_sent = []  
            else:
                single_sent.append(word_dict[tmp[1]])

                tokenized_single_sent.append(word_dict["<sot>"])
                tokenized_single_sent.append(word_dict[tmp[1]])
                tokenized_single_sent.append(word_dict["<eot>"])

    return np.array(input_data), np.array(target_data)


class EuroparlDataSet(Dataset):
    def __init__(self,
                 data_dir: str = None,
                 ) -> None:
        """
        Args:
            data_dir:
        """
        self.input_data, self.output_data = read_data(data_dir)


    def __getitem__(self, x):
        """
        Args:
            x:
        Returns:
        """
        return {'text':torch.LongTensor(self.input_data[x]), 'label':torch.LongTensor(self.output_data[x])}

    def __len__(self) -> int:
        return len(self.input_data)

class EuroparlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = None
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self):
        self.dataset = EuroparlDataSet(self.data_dir)
        

    def setup(self, stage = None):

        length = len(self.dataset)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=self.dataset,
            lengths=[int(length * 0.6) + 1, int(length * 0.2), int(length * 0.2)]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )