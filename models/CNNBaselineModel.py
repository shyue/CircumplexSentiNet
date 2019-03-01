"""
CNNBaselineModel.py: baseline model that runs characters through embeddings,
and then through CNN to classify
"""

import torch.nn as nn
from models.cnn import CNN

class CNNBaselineModel(nn.Module): 
    
    def __init__(self, embed_size, vocab, kernel):
        """
        Init the CNN
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (dict): Vocab dictionary
        @param kernel (int): kernel size for CNN
        """
        super(CNNBaselineModel, self).__init__()
        
        echar = 50
        dropout_rate = 0.3
        self.embed_size = embed_size
        
        self.embeddings = nn.Embedding(len(vocab), echar, padding_idx=vocab['<pad>'])
        self.cnn = CNN(echar, embed_size, kernel)
        self.dropout = nn.Dropout(dropout_rate)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of length len

        @param output: Tensor of length embed_size
        """
        xemb = self.embeddings(input)
        xreshaped = xemb.permute(0, 2, 1)
        xconv_out = self.cnn(xreshaped)
        output = self.dropout(xconv_out)
        return output


