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

        self.embedding = nn.Embedding(cfg.experiment.dict_size, cfg.experiment.embedding_dim, padding_idx=0)
        self.hid_dim = cfg.experiment.hidden_size

        self.rnn = nn.LSTM(
            cfg.experiment.embedding_dim, 
            cfg.experiment.hidden_size, 
            num_layers=cfg.experiment.n_layers, 
            bidirectional = True,
            dropout = cfg.experiment.dropout[0]
        )
        self.fc = nn.Linear(self.hid_dim * 2, self.hid_dim)
        self.dropout = nn.Dropout(cfg.experiment.dropout[0])

    def forward(self, src):

        #src = src.reshape(src.shape[0], src.shape[1], -1)
        #src = torch.permute(src, (1, 0, 2))

        embedded = self.dropout(self.embedding(src))
        output, hidden = self.rnn(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        return output, hidden

class Attention(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = nn.Linear((cfg.experiment.hidden_size * 2) + cfg.experiment.hidden_size, cfg.experiment.hidden_size)
        self.v = nn.Linear(cfg.experiment.hidden_size, 1, bias = False)

    def forward(self, hidden, encoder_outputs):
        
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
        self.fc_out = nn.Linear(cfg.experiment.embedding_dim + self.hid_dim * 3, self.output_dim)
        self.dropout = nn.Dropout(cfg.experiment.dropout[1])

    def forward(self, input, hidden, encoder_output):
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        a = self.attention(hidden, encoder_output)
        a = a.unsqueeze(1)

        encoder_output = encoder_output.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_output)
        weighted = weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        return prediction, hidden.squeeze(0)

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
        encoder_outputs, hidden= self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            
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