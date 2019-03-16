"""
Runs word embeddings through bidirectional LSTM, and then CNN to classify
"""

import torch
import torch.nn as nn
from models.cnn import CNN
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

class MultitaskBiLSTMAttention(nn.Module): 
    
    def __init__(self, hidden_att, out_att, hidden_lstm, input_size):
        super(MultitaskBiLSTMAttention, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_lstm, bidirectional=True)
        self.att_1 = nn.Linear(2*hidden_lstm, hidden_att, bias=False)
        self.att_2 = nn.Linear(hidden_att, out_att, bias=False)
        
        self.v_inner = nn.Linear(out_att, 1)
        self.v_final = nn.Linear(2*hidden_lstm, 1)
        
        self.a_inner = nn.Linear(out_att, 1)
        self.a_final = nn.Linear(2*hidden_lstm, 1)
        


    def forward(self, input, lengths):
        xhidden, _ = self.lstm(pack_padded_sequence(input, lengths))
        xhidden = pad_packed_sequence(xhidden)[0]
        xhidden = xhidden.permute(1, 0, 2)
        att_in = torch.tanh(self.att_1(xhidden))
        att_out = F.softmax(self.att_2(att_in), dim=1)
        #M = torch.mm(xhidden.permute(0, 2, 1), att_out)
        M = xhidden.permute(0, 2, 1).matmul(att_out)
        
        inner_v = F.relu(torch.squeeze(self.v_inner(M)))
        output_v = torch.squeeze(self.v_final(inner_v))
        
        inner_a = F.relu(torch.squeeze(self.a_inner(M)))
        output_a = torch.squeeze(self.a_final(inner_a))

        output = (torch.squeeze(output_v), torch.squeeze(output_a))
        return output


