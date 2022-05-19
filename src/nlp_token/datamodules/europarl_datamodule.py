import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision.transforms import transforms

from ..configs import Config



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
    tokenized_single_sent = []
    group_of_tok_sent = []
    group_of_sent = []
    input_data = []
    output_data = []

    with open(file_path, encoding=encoding) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()

            if line == "\n":
                group_of_tok_sent.append(tokenized_single_sent)
                group_of_sent.append(" ".join(tokenized_single_sent))
                tokenized_single_sent = []
                if len(group_of_tok_sent) == 3:
                    input_data.append(" ".join(group_of_sent))
                    output_data.append(group_of_tok_sent)
                    group_of_sent = []
                    group_of_tok_sent = []
            else:
                tokenized_single_sent.append(tmp[1])

    return (input_data, output_data)


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
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
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