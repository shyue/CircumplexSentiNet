"""
Runs word embeddings through bidirectional LSTM, and then CNN to classify
"""

import torch
import torch.nn as nn
from models.cnn import CNN
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class MultitaskBiLSTMCNN(nn.Module): 
    
    def __init__(self, embed_size, kernel, hidden, input_size):
        """
        Init the CNN
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (dict): Vocab dictionary
        @param kernel (int): kernel size for CNN
        """
        super(MultitaskBiLSTMCNN, self).__init__()
        
        dropout_rate = 0.3
        self.linear_v = nn.Linear(embed_size, 1)
        self.linear_a = nn.Linear(embed_size, 1)
        self.lstm = nn.LSTM(input_size, hidden, bidirectional=True)
        self.cnn = CNN(2*hidden, embed_size, kernel)
        #self.conv = nn.Conv1d(2*hidden, embed_size, kernel)
        #self.dropout = nn.Dropout(dropout_rate)
        ### END YOUR CODE

    def forward(self, input, lengths):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of length len

        @param output: Tensor of length embed_size
        """
        xhidden, _ = self.lstm(pack_padded_sequence(input, lengths))
        xhidden = pad_packed_sequence(xhidden)[0]
        xhidden = xhidden.permute(1, 2, 0)
        xconv_out = self.cnn(xhidden)
        #xconv = self.conv(xhidden)
        #xconv_out = torch.max(xconv, dim=2)[0]
        #output = xconv_out #self.dropout(xconv_out)
        output_v = self.linear_v(xconv_out)
        output_a = self.linear_a(xconv_out)
        output = (torch.squeeze(output_v), torch.squeeze(output_a))
        return output


