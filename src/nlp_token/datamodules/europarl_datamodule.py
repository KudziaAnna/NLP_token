import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import transforms

from ..configs import Config


def prepare_dict(file_path, sep="\t", encoding=None):
    word_dict = {
        0: "<PAD>"
    }
    longest_sentence_size = 0
    i = 0
    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if line != "\n":
                i += 0
                if tmp[1] not in word_dict:
                    word_dict[tmp[1]] = len(word_dict) + 1
            else:
                if i > longest_sentence_size:
                    longest_sentence_size = i
                    i = 0
    
    return word_dict, longest_sentence_size


def read_conll_file(file_path, sep="\t", encoding=None):
    """
    Reads a data file in CoNLL format and returns word and label lists.

    Args:
        file_path (str): Data file path.
        sep (str, optional): Column separator. Defaults to "\t".
        encoding (str): File encoding used when reading the file.
            Defaults to None.

    Returns:
        (list, list): A tuple of word and label lists (list of lists).
    """

    word_dict, longest_sentence_size = prepare_dict(file_path)

    tokenized_single_sent = []
    single_sent = []
    group_of_tok_sent = []
    group_of_sent = []
    input_data = []
    output_data = []

    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()

            if line == "\n":
                #group_of_tok_sent.append(tokenized_single_sent)
                group_of_sent += single_sent
                #if len(group_of_tok_sent) == 3:
                input_data.append(group_of_sent)
                output_data.append(tokenized_single_sent)
                group_of_sent = []
                tokenized_single_sent = []
                single_sent = []
            else:
                lookup_tensor = torch.tensor(word_dict[tmp[1]], dtype=torch.long)
                single_sent.append(lookup_tensor)
                tokenized_single_sent.append([lookup_tensor])
    
    return (input_data, output_data)



def pad_collate_fn(batch):
    
    sequences_vectors, vec_lengths = zip(*[
        (torch.LongTensor(np.stack(seq_vectors)), len(seq_vectors))
        for (seq_vectors, _) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
    ])

    sequences_labels, lab_lengths = zip(*[
        (torch.LongTensor(np.stack(labels)), len(labels))
        for (_, labels) in sorted(batch, key=lambda x: len(x[0]), reverse=True)
    ])

    vec_lengths = torch.LongTensor(vec_lengths)
    lab_lengths = torch.LongTensor(lab_lengths)

    padded_sequences_vectors = pad_sequence(sequences_vectors, batch_first=True, padding_value=0)
    padded_sequences_labels = pad_sequence(sequences_labels, batch_first=True, padding_value=-100)

    pack_padded_sequences_vectors = pack_padded_sequence(
        padded_sequences_vectors, vec_lengths.cpu(), batch_first=True
    )  # We pack the padded sequence to improve the computational speed during training

    return (pack_padded_sequences_vectors, padded_sequences_labels)

class EuroparlDataSet(Dataset):
    def __init__(self,
                 data_dir: str = None,
                 ) -> None:
        """
        Args:
            data_dir:
        """
        self.input_data, self.output_data = read_conll_file(data_dir)
        assert len(self.input_data) == len(self.output_data)

    def __getitem__(self, x):
        """
        Args:
            x:
        Returns:
        """
        data_dict = dict({"input": self.input_data[x],
                        "output": self.output_data[x]})

        return data_dict["input"], data_dict["output"]

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

        print(data_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.ToTensor()

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
            collate_fn=pad_collate_fn,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pad_collate_fn,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=pad_collate_fn,
            shuffle=False,
        )