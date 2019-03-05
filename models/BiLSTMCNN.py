"""
Runs word embeddings through bidirectional LSTM, and then CNN to classify
"""

import torch.nn as nn
from models.cnn import CNN
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTMCNN(nn.Module): 
    
    def __init__(self, embed_size, kernel, hidden, input_size):
        """
        Init the CNN
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (dict): Vocab dictionary
        @param kernel (int): kernel size for CNN
        """
        super(BiLSTMCNN, self).__init__()
        
        dropout_rate = 0.3
        
        self.lstm = nn.LSTM(input_size, hidden, bidirectional=True)
        self.cnn = CNN(2*hidden, embed_size, kernel)
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
        output = xconv_out #self.dropout(xconv_out)
        return output


