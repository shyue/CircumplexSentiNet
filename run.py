"""
run.py: trains the model and tests it

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import pandas as pd
from models.BiLSTMCNN import BiLSTMCNN
from vocab import *
import random
import time

EMBED_SIZE = 17
KERNEL_SIZE = 5
VALENCE_MODEL = "model_weights/valence_model"
AROUSAL_MODEL = "model_weights/arousal_model"
BATCH_SIZE = 32
INPUT_SIZE = 100
HIDDEN_SIZE = 1024
LEARNING_RATE = 0.001

def train(file):
    sentences = pd.read_csv(file).dropna()
    valences, arousals = extract_labels(sentences)
    sentences = read_sents(sentences)
    #print(valences)
    print("Loading glove vectors")
    glove = load_glove_vectors()
    pad = [0.0 for i in range(INPUT_SIZE)]
    print("Finished loading vectors")
    
    random.seed(42)
    
    
    model_valence = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    model_arousal = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    
    criterion_valence = nn.CrossEntropyLoss()
    criterion_arousal = nn.CrossEntropyLoss()
    optimizer_valence = optim.Adam(model_valence.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    optimizer_arousal = optim.Adam(model_arousal.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    
    model_valence.train()
    model_arousal.train()
    
    indexes = [i for i in range(len(sentences))]
    t0 = time.time()
    #based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for epoch in range(100):  # loop over the dataset multiple times
        random.shuffle(indexes)
        
        running_loss = 0.0
        for i in range(len(sentences)//BATCH_SIZE):
            
            # get the inputs
            
            inputs = [sentences[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            inputs, lengths = pad_sents(inputs)
            inputs = [[glove[word] if word in glove else pad for word in sent] for sent in inputs]
            inputs = torch.tensor(inputs)
            inputs = inputs.permute(1, 0, 2)

            labels_valence = [valences[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            labels_valence = torch.tensor(labels_valence, dtype=torch.long)
            
            labels_arousal = [arousals[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            labels_arousal = torch.tensor(labels_arousal, dtype=torch.long)
            
            # zero the parameter gradients
            optimizer_valence.zero_grad()
            optimizer_arousal.zero_grad()

            # forward + backward + optimize
            outputs_valence = model_valence(inputs, lengths)
            outputs_arousal = model_arousal(inputs, lengths)
            
            loss_valence = criterion_valence(outputs_valence, labels_valence)
            loss_arousal = criterion_arousal(outputs_arousal, labels_arousal)
            loss = loss_arousal+loss_valence
            loss.backward()

            optimizer_valence.step()
            optimizer_arousal.step()
            
            # print statistics
            running_loss += loss.item()
            #print(loss.item())
            #if i % 400 == 399:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / (len(sentences)//BATCH_SIZE)))
        print('time: '+str(time.time()-t0))
        running_loss = 0.0
            
    print('Finished Training')
    torch.save(model_valence.state_dict(), VALENCE_MODEL)
    torch.save(model_arousal.state_dict(), AROUSAL_MODEL)
    

def test(file):
    sentences = pd.read_csv(file).dropna()
    valences, arousals = extract_labels(sentences)
    sentences = read_sents(sentences)
    glove = load_glove_vectors()

    """
    model_valence = 
    model_valence.load_state_dict(torch.load(VALENCE_MODEL))
    model_valence.eval()
    
    model_arousal = 
    model_arousal.load_state_dict(torch.load(AROUSAL_MODEL))
    model_arousal.eval()

    SSE_V = 0
    SSE_A = 0
    
    with torch.no_grad():
        for i, data in enumerate(id_sents):
            inputs = data.view(1, data.shape[0])
            outputs_valence = model_valence(inputs)
            outputs_arousal = model_arousal(inputs)
            labels_arousal = torch.tensor([arousals[i]], dtype=torch.long)
            labels_valence = torch.tensor([valences[i]], dtype=torch.long)
            SSE_V += (torch.max(outputs_valence, dim=1)[1]+1-labels_valence)*(torch.max(outputs_valence, dim=1)[1]+1-labels_valence)
            SSE_A += (torch.max(outputs_arousal, dim=1)[1]+1-labels_arousal)*(torch.max(outputs_arousal, dim=1)[1]+1-labels_arousal)
    SSE_V = SSE_V/len(id_sents)
    SSE_A = SSE_A/len(id_sents)
    print(SSE_V, SSE_A)
"""

if __name__ == '__main__':
    parser =ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-f", dest="fname")
    parser.add_argument("-d", dest="data_split")
    args = parser.parse_args()
    
    if (args.fname == None):
        print("Please select a file")
    
    if (args.mode == 'train'):
        train(args.fname)
    if (args.mode == 'test'):
        test(args.fname)
