from lib2to3.pgen2.tokenize import tokenize
from tokenize import group
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchtext.legacy.data import Field, BucketIterator
import numpy as np

import random
import math
import time

SEED = 1234
FILE_PATH = "/home/ania/Documents/NLP_token/data/Europarl_En.conl"
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def prepare_dict():
    word_dict = {
        "<pad>" : 0,
        "<sos>" : 1,
        "<eos>" : 2,
    }
    with open(FILE_PATH) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()
            if line != "\n":
                if tmp[1] not in word_dict:
                    word_dict[tmp[1]] = len(word_dict) + 1
    return word_dict

def read_data():
    word_dict = prepare_dict()
    group_sent = []
    group_tok_sent = []
    tokenized_single_sent = [[word_dict["<sos>"]]]
    single_sent = [word_dict["<sos>"]]
    input_data = []
    target_data = []

    with open(FILE_PATH) as f:
        lines = f.readlines()
        for line in lines:
            tmp = line.split()

            if line == "\n":
                tokenized_single_sent.append([word_dict["<eos>"]])
                single_sent.append(word_dict["<eos>"])
                group_sent.append(single_sent)
                group_tok_sent.append(tokenized_single_sent)
                tokenized_single_sent = [[word_dict["<sos>"]]]
                single_sent = [word_dict["<sos>"]]
                if len(group_tok_sent) == 3:
                    input_data.append(np.array(group_sent, dtype=object).reshape(3,1))
                    target_data.append(np.array(group_tok_sent, dtype=object).reshape(3,1))
                    group_sent = []
                    group_tok_sent = []    
            else:
                single_sent.append(word_dict[tmp[1]])
                tokenized_single_sent.append([word_dict[tmp[1]]])
    
    return np.stack(input_data), np.stack(target_data)



def prepare_data():
    input_data, target_data = read_data()
    input_train, input_test, target_train, target_test = train_test_split(np.array(input_data), np.array(target_data), test_size = 0.4)
    input_val, input_test, target_val, target_test = train_test_split(input_test, target_test, test_size = 0.5)
    train_data = [{"input": input_train[x], "target": target_train[x]} for x in range(len(input_train))]
    val_data = [{"input": input_val[x], "target": target_val[x]} for x in range(len(input_val))]
    test_data = [{"input": input_test[x], "target": target_test[x]} for x in range(len(input_test))]

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data), 
     batch_size = BATCH_SIZE,
     sort_within_batch = True,
     sort_key = lambda x : len(x.src),
     device = device)
    return train_iterator, valid_iterator, test_iterator
       

if __name__ == '__main__':
    prepare_data()