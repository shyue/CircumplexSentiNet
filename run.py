"""
run.py: trains the model and tests it

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import pandas as pd
from models.CNNBaselineModel import CNNBaselineModel

EMBED_SIZE = 9
KERNEL_SIZE = 5
VALENCE_MODEL = "model_weights/valence_model"
AROUSAL_MODEL = "model_weights/arousal_model"

def train(file):
    sentences = pd.read_csv(file).dropna()
    vocab, id_sents, valences, arousals = char2id(sentences)
    id_sents = pad_sents(id_sents, KERNEL_SIZE, vocab)
    
    model_valence = CNNBaselineModel(EMBED_SIZE, vocab, KERNEL_SIZE)
    model_arousal = CNNBaselineModel(EMBED_SIZE, vocab, KERNEL_SIZE)
    
    criterion_valence = nn.CrossEntropyLoss()
    criterion_arousal = nn.CrossEntropyLoss()
    optimizer_valence = optim.SGD(model_valence.parameters(), lr=0.001, momentum=0.9)
    optimizer_arousal = optim.SGD(model_arousal.parameters(), lr=0.001, momentum=0.9)
    
    model_valence.train()
    model_arousal.train()
    
    #based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for epoch in range(30):  # loop over the dataset multiple times
 
        running_loss = 0.0
        for i, data in enumerate(id_sents):
            # get the inputs
            inputs = data.view(1, data.shape[0])
            labels_arousal = torch.tensor([arousals[i]-1], dtype=torch.long)
            labels_valence = torch.tensor([valences[i]-1], dtype=torch.long)
            # zero the parameter gradients
            optimizer_valence.zero_grad()
            optimizer_arousal.zero_grad()

            # forward + backward + optimize
            outputs_valence = model_valence(inputs)
            outputs_arousal = model_arousal(inputs)
            
            loss_valence = criterion_valence(outputs_valence, labels_valence)
            loss_arousal = criterion_arousal(outputs_arousal, labels_arousal)
            loss = loss_arousal+loss_valence
            loss.backward()

            optimizer_valence.step()
            optimizer_arousal.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0

    print('Finished Training')
    torch.save(model_valence.state_dict(), VALENCE_MODEL)
    torch.save(model_arousal.state_dict(), AROUSAL_MODEL)


def test(file):
    sentences = pd.read_csv(file).dropna()
    vocab, id_sents, valences, arousals = char2id(sentences)
    id_sents = pad_sents(id_sents, KERNEL_SIZE, vocab)
    
    model_valence = CNNBaselineModel(EMBED_SIZE, vocab, KERNEL_SIZE)
    model_valence.load_state_dict(torch.load(VALENCE_MODEL))
    model_valence.eval()
    
    model_arousal = CNNBaselineModel(EMBED_SIZE, vocab, KERNEL_SIZE)
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

def pad_sents(id_sents, ksize, vocab):
    for i, sent in enumerate(id_sents):
        if sent.shape[0] < ksize:
            new_sent = sent.tolist()
            for j in range(ksize-len(new_sent)):
                new_sent.append(vocab['<pad>'])
            id_sents[i] = torch.tensor(new_sent)
    return id_sents
    
def char2id(df):
    char_list = list("""ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")
    vocab = dict()
    vocab['<pad>'] = 0   
    for c in char_list:
        vocab[c] = len(vocab)
    
    id_sents = []
    valences = []
    arousals = []
    for idx, row in df.iterrows():
        msg = row['Anonymized Message']
        valences.append( (int(row['Valence1'])+int(row['Valence2']))/2 )
        arousals.append( (int(row['Arousal1'])+int(row['Arousal2']))/2 )
        msg_id = []
        for char in msg:
            if char in vocab:
                msg_id.append(vocab[char])
            else:
                msg_id.append(vocab['<pad>'])
        id_sents.append(torch.tensor(msg_id))

    return vocab, id_sents, valences, arousals
    
if __name__ == '__main__':
    parser =ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-f", dest="fname")
    args = parser.parse_args()
    
    if (args.fname == None):
        print("Please select a file")
    
    if (args.mode == 'train'):
        train(args.fname)
    if (args.mode == 'test'):
        test(args.fname)