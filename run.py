"""
run.py: trains the model and tests it

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import pandas as pd
from models.MultitaskBiLSTMCNN import MultitaskBiLSTMCNN
from vocab import *
import random
import time

EMBED_SIZE = 256
KERNEL_SIZE = 5
VALENCE_MODEL = "model_weights/valence_model"
AROUSAL_MODEL = "model_weights/arousal_model"
MULTITASK_MODEL = "model_weights/multitask_model"
BATCH_SIZE = 1
INPUT_SIZE = 100
HIDDEN_SIZE = 1024
LEARNING_RATE = 0.001
PRINT_SIZE = 200
PREDICTION_OUTPUT = "data/test_output.csv"
NUM_EPOCHS = 30

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
    
    
    #model_valence = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    #model_arousal = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    model = MultitaskBiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    
    criterion_valence = nn.MSELoss()
    criterion_arousal = nn.MSELoss()
    #optimizer_valence = optim.Adam(model_valence.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    #optimizer_arousal = optim.Adam(model_arousal.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#, momentum=0.9)
    
    """
    model_valence.train()
    model_arousal.train()
    """
    model.train()
    
    indexes = [i for i in range(len(sentences))]
    t0 = time.time()
    #based on PyTorch tutorial: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
        random.shuffle(indexes)
        
        running_loss = 0.0
        for i in range(len(sentences)//BATCH_SIZE):
            
            # get the inputs
            
            inputs = [sentences[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            inputs, lengths = pad_sents(inputs, KERNEL_SIZE)
            inputs = [[glove[word] if word in glove else pad for word in sent] for sent in inputs]
            inputs = torch.tensor(inputs)
            inputs = inputs.permute(1, 0, 2)

            labels_valence = [valences[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            labels_valence = torch.tensor(labels_valence)
            
            labels_arousal = [arousals[i] for i in indexes[i*BATCH_SIZE:(i+1)*BATCH_SIZE]]
            labels_arousal = torch.tensor(labels_arousal)
            
            # zero the parameter gradients
            #optimizer_valence.zero_grad()
            #optimizer_arousal.zero_grad()
            optimizer.zero_grad()
            
            # forward + backward + optimize
            #outputs_valence = model_valence(inputs, lengths)
            #outputs_arousal = model_arousal(inputs, lengths)
            outputs = model(inputs, lengths)
            
            loss_valence = criterion_valence(outputs[0], labels_valence)
            loss_arousal = criterion_arousal(outputs[1], labels_arousal)
            loss = loss_arousal+loss_valence
            loss.backward()

            #optimizer_valence.step()
            #optimizer_arousal.step()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            #print(outputs_valence)
            #print(labels_valence)
            """print(loss.item())
            print(loss.item())
            print(running_loss)
            print('---')"""
            if i % PRINT_SIZE == PRINT_SIZE-1:    # print every 400 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / PRINT_SIZE))
                running_loss = 0.0
            
        print('epoch '+str(epoch)+' finished! time: '+str(time.time()-t0))
        if (epoch % 5 == 0):
            #torch.save(model_valence.state_dict(), VALENCE_MODEL+'_epoch'+str(epoch))
            #torch.save(model_arousal.state_dict(), AROUSAL_MODEL+'_epoch'+str(epoch))
            torch.save(model.state_dict(), MULTITASK_MODEL+'_epoch'+str(epoch))
            
    print('Finished Training')
    torch.save(model.state_dict(), MULTITASK_MODEL+'_epoch'+str(epoch))
    #torch.save(model_valence.state_dict(), VALENCE_MODEL)
    #torch.save(model_arousal.state_dict(), AROUSAL_MODEL)
    print('Finished Saving Weights')

def test(file):
    sentences = pd.read_csv(file).dropna()
    valences, arousals = extract_labels(sentences)
    sentences = read_sents(sentences)
    #print(valences)
    print("Loading glove vectors")
    glove = load_glove_vectors()
    pad = [0.0 for i in range(INPUT_SIZE)]
    print("Finished loading vectors")
    
    """
    model_valence = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    model_valence.load_state_dict(torch.load(VALENCE_MODEL))
    model_valence.eval()
    
    model_arousal = BiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    model_arousal.load_state_dict(torch.load(AROUSAL_MODEL))
    model_arousal.eval()
    """
    model = MultitaskBiLSTMCNN(EMBED_SIZE, KERNEL_SIZE, HIDDEN_SIZE, INPUT_SIZE)
    model.load_state_dict(torch.load(MULTITASK_MODEL))
    model.eval()
    
    pred_valence = []
    pred_arousal = []
    
    t0 = time.time()
    with torch.no_grad():
        
        for i, data in enumerate(sentences):
            
            # get the inputs
            inputs = [data]
            inputs, lengths = pad_sents(inputs, KERNEL_SIZE)
            inputs = [[glove[word] if word in glove else pad for word in sent] for sent in inputs]
            inputs = torch.tensor(inputs)
            inputs = inputs.permute(1, 0, 2)
            
            """
            labels_valence = [valences[i]]
            labels_valence = torch.tensor(labels_valence)
            
            labels_arousal = [arousals[i]]
            labels_arousal = torch.tensor(labels_arousal)
            """
            
            #outputs_valence = model_valence(inputs, lengths)
            #outputs_arousal = model_arousal(inputs, lengths)
            outputs = model(inputs, lengths)
            
            pred_valence.append(outputs[0].tolist())
            pred_arousal.append(outputs[1].tolist())
        
            
    print('Finished Testing: '+str(time.time()-t0))
        
    SSE_V = 0
    SSE_A = 0
    pred_totals = []
    
    for i in range(len(sentences)):
        curr_line = [sentences[i], valences[i]+5, pred_valence[i]+5, arousals[i]+5, pred_arousal[i]+5]
        pred_totals.append(curr_line)
        SSE_V += (valences[i]-pred_valence[i])**2
        SSE_A += (arousals[i]-pred_arousal[i])**2
    SSE_V = SSE_V/len(sentences)
    SSE_A = SSE_A/len(sentences)
    print("Valence MSE: "+str(SSE_V)+", Arousal MSE: "+str(SSE_A))
    
    df = pd.DataFrame(pred_totals, columns = ['Message', 'Valence Label', 'Valence Prediction', 'Arousal Label', 'Arousal Prediction'])
    df.to_csv(PREDICTION_OUTPUT)
    
    print('Finished Writing Data')

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
