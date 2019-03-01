import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, echar, embed_size, k):
        """Initialize CNN layer
        
        @param echar (int): Embedding size of the characters
        @param embed_size (int): Output embedding size of CNN
        @param k (int): kernel size
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(echar, embed_size, k)
    
    def forward(self, xin):
        """Go forward in CNN
        
        @param xin (Tensor): input tensor of character embeddings, size echar * len
        
        @return xconv_out (Tensor): output tensor after going through convolution
        """
        xconv = self.conv(xin)
        xconv_out = torch.max(F.relu(xconv), dim=2)[0]
        return xconv_out
