"""

methods for extracting word vectors, padding, turning into character ids

"""

from nltk.tokenize import TweetTokenizer
import torch
import pandas as pd
import copy

def pad_chars(id_sents, ksize, vocab):
    for i, sent in enumerate(id_sents):
        if sent.shape[0] < ksize:
            new_sent = sent.tolist()
            for j in range(ksize-len(new_sent)):
                new_sent.append(vocab['<pad>'])
            id_sents[i] = torch.tensor(new_sent)
    return id_sents

def extract_labels(df):
    """
    extract labels for each sentence, taken as the average of the specific columns relating to valence and arousal
    """
    valences = []
    arousals = []
    for idx, row in df.iterrows():
        valences.append( (int(row['Valence1'])+int(row['Valence2'])) / 2 - 5)#(int(row['Valence1'])+int(row['Valence2'])) -2)
        arousals.append( (int(row['Arousal1'])+int(row['Arousal2'])) / 2 - 5)#(int(row['Arousal1'])+int(row['Arousal2'])) -2)
    return valences, arousals

def read_sents(df):
    """
    reads each row of dataframe for message, and tokenizes it
    
    returns list of a list of lowercase words
    """
    sents = []
    for idx, row in df.iterrows():
        msg = row['Anonymized Message']
        tknzr = TweetTokenizer()
        curr_sent = []
        for word in tknzr.tokenize(msg):
            curr_sent.append(word.lower())
        sents.append(curr_sent)
    return sents
    
def pad_sents(sents, minLen):
    """
    reads in list of a list of lowercase words
    
    returns list of a list of lowercase words, where each sentence is the same length
    list is sorted by length, and original lengths are returned as well
    """
    new_sents = copy.deepcopy(sents)
    new_sents.sort(reverse=True, key=len)
    lengths = [max(len(s), minLen) for s in new_sents]
    max_len = max(lengths)

    for i in range(len(new_sents)):
        while (len(new_sents[i]) < max_len):
            new_sents[i].append('')

    return new_sents, lengths
    
def load_glove_vectors(fname="data/glove.twitter.27B/glove.twitter.27B.100d.txt"):
    """
    loads glove vectors
    
    returns dictionary of word:Tensor embedding
    """
    vocab = dict()
    with open(fname, 'rb') as file:
        for line in file:
            split_line = line.decode().split()
            word = split_line[0]
            vocab[word] = [float(i) for i in split_line[1:]]
    return vocab
        
    
    
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
    