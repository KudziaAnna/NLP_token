import sys
import random
from unicodedata import bidirectional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plfunc

SEED = 42
random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from ..configs import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(pl.LightningModule):
    """RNN encoder module
    """

    def __init__(
        self, cfg: Config):
        super().__init__()

        self.embedding = nn.Embedding(cfg.experiment.dict_size, cfg.experiment.embedding_dim)
        self.hid_dim = cfg.experiment.hidden_size

        self.rnn = nn.LSTM(
            cfg.experiment.embedding_dim, 
            cfg.experiment.hidden_size, 
            num_layers=cfg.experiment.n_layers, 
            bidirectional = True,
            dropout = cfg.experiment.dropout[0]
        )
        self.fc_hidden = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.fc_cell = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.dropout = nn.Dropout(cfg.experiment.dropout[0])

    def forward(self, src):

        #src = src.reshape(src.shape[0], src.shape[1], -1)
        #src = torch.permute(src, (1, 0, 2))

        embedded = self.dropout(self.embedding(src))
        output, (hidden, cell) = self.rnn(embedded)

        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return output, hidden, cell

class Attention(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = nn.Linear((cfg.experiment.hidden_size * 2) + cfg.experiment.hidden_size, cfg.experiment.hidden_size)
        self.v = nn.Linear(cfg.experiment.hidden_size, 1, bias = False)

    def forward(self, encoder_outputs, hidden, cell):
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)

class Decoder(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.output_dim = cfg.experiment.output_size
        self.attention = Attention(cfg)
        self.hid_dim = cfg.experiment.hidden_size

        self.embedding = nn.Embedding(cfg.experiment.dict_size, cfg.experiment.embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(cfg.experiment.embedding_dim + self.hid_dim * 2, self.hid_dim, cfg.experiment.n_layers, dropout = cfg.experiment.dropout[1])
        #self.fc_out = nn.Linear(cfg.experiment.embedding_dim + self.hid_dim * 3, self.output_dim)

        self.energy = nn.Linear(self.hid_dim * 3, 1)
        self.fc = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(cfg.experiment.dropout[1])
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()



    def forward(self, input,  encoder_output, hidden, cell):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        sequence_length = encoder_output.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_output), dim=2)))

        a = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", a, encoder_output)

        rnn_input = torch.cat((context_vector, embedded), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)

        return predictions, hidden, cell


class LSTMBasedSeq2Seq(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
    
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_outputs, hidden, cell= self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, encoder_outputs, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs