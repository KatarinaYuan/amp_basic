from tqdm import tqdm

import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from utils import train_transformer_MCdropout, PAD_TOKEN_ID

class AMPCLS(nn.Module):
    def __init__(self, **kwargs):
        super(AMPCLS, self).__init__()
        assert 'pos_weight' in kwargs
        if kwargs.get('pos_weight') is not None: # for unbalanced data 
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.loss_function = nn.BCEWithLogitsLoss()

    def forward(self, inp_seq):
        raise NotImplementedError
    
    def predict(self, logits, threshold=0.5):
        # input: logits of shape (bsz, 1)
        # output: predicted probability and hard labels according to threshold
        # threshold: for assining hard labels, i.e., label is 1 if proba > threshold

        proba = torch.sigmoid(logits).detach().cpu().numpy()
        proba = np.hstack((1-proba, proba))
        label = np.array(proba[:, 1] > threshold, dtype=np.int32)

        return proba, label
    
    def compute_loss(self, logits, y):
        return self.loss_function(logits, y.float())

class LSTM_CLS(AMPCLS):
    # An LSTM classifier to predict whether a sequence belongs to AMP or not.
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size,
                num_layers, max_len, dropout, bidirectional=True,
                word_embedder=None, **kwargs):
        print(kwargs)
        super(LSTM_CLS, self).__init__(**kwargs)

        self.embedding_size = embedding_size
        self.max_len = max_len
        
        self.word_embedder = nn.Embedding(vocab_size, embedding_size)
        if word_embedder is not None:
            self.word_embedder.weight.data = word_embedder._embedding.data.detach().clone()
        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*2, output_size) if bidirectional else nn.Linear(hidden_size, output_size)

    def forward(self, seqs):
        # seqs is in shape [batch_size, max_len]
        seq_lens = torch.count_nonzero(seqs-PAD_TOKEN_ID, dim=1).detach().cpu().numpy()
        x = self.word_embedder(seqs)
        pack = pack_padded_sequence(x, seq_lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(pack)
        x, unpacked_len = pad_packed_sequence(x) # x here should be of shape: [max_len, batch_size, hidden_size * #direction]
        #x = torch.squeeze(x[-1, : , :]) # here we use the output on the last position for prediction
        #x = torch.squeeze(x[0, : , :]) # here we use the output on the first position (i.e., BOS token) for prediction
        x = torch.squeeze(x[1, : , :]) # here we use the output on the first meaningful position for prediction
        x = self.fc(x)

        return x


# the definition of this class is from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerCLS(AMPCLS):
    # An transformer-based classifier to predict whether a sequence belongs to AMP or not.
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, 
                num_layers, max_len, dropout=0., nhead=1, word_embedder=None, **kwargs):
        super(TransformerCLS, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.max_len = max_len

        self.word_embedder = nn.Embedding(vocab_size, embedding_size)
        if word_embedder is not None:
            self.word_embedder.weight.data = word_embedder._embedding.data.detach().clone()
        
        self.pos_encoder = PositionalEncoding(embedding_size, dropout, max_len=max_len+2)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # self.fc = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(embedding_size, output_size)
    
    def forward(self, seqs):
        # seqs is of shape [batch_size, max_len]
        # In transformer, we should first exchange first two dimensions.
        x = torch.transpose(seqs, 0, 1)
        src_key_padding_mask = (x == PAD_TOKEN_ID)

        x = self.word_embedder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=torch.transpose(src_key_padding_mask, 0, 1))
        # x = self.transformer_encoder(x) # no mask applied
        x = x[1, :, :]
        x = self.fc(x)

        return x 
