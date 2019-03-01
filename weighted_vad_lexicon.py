"""
weighted_vad_lexicon.py: Doing basic weighting using VAD lexicon

NRC VAD Lexicon found here: http://saifmohammad.com/WebPages/nrc-vad.html
"""

import numpy as np
import pandas as pd
import os
from nltk.tokenize import TweetTokenizer

def weight_sentences(file="http://wwbp.org/downloads/public_data/dataset-fb-valence-arousal-anon.csv", sentOnly=False):
    """
    download data and weight the sentences for the valence/arousal scores
    
    @param file (string): location of datafile to download
    @param sentOnly (boolean): whether to divide only by sentiment words, rather than all words in the sentence
    """
    print("Loading files...")
    sentences = pd.read_csv(file).dropna()
    lexicon = pd.read_csv('NRC-VAD-Lexicon.txt', sep="\t")
    lexicon = lexicon.set_index('Word').T.to_dict('list')
    print("Lexicon and sentences loaded")
    
    sentences["Predicted Valence"] = np.vectorize(weight_sentence)(sentences['Anonymized Message'], lexicon, sentOnly, 0) 
    sentences["Predicted Arousal"] = np.vectorize(weight_sentence)(sentences['Anonymized Message'], lexicon, sentOnly, 1)
    
    sentences.to_csv("output/weighted_vad_lexicon.csv", index = None, header=True)
    
    
def weight_sentence(sentence, lexicon, sentOnly, index):
    """
    weight each individual sentence
    
    @param sentence (dataframe): current sentence to weight
    @param lexicon (dict): dictionary of lexicon words
    @param sentOnly (boolean): whether to divide only by sentiment words, rather than all words in the sentence
    @param index (int): index of the lexicon to use to calculate weights
    """

    numWords = 0
    sentWords = 0
    weight = 0
    tknzr = TweetTokenizer()
    
    for word in tknzr.tokenize(sentence):
        word = word.lower()
        numWords += 1
        if word in lexicon:
            sentWords += 1
            weight += lexicon[word][index]
            
    if (sentOnly):
        if sentWords == 0: 
            sentWords = 1
        weight = weight / sentWords
    else:
        weight = weight / numWords
        
    return weight*8+1
        
    
    
if __name__ == '__main__':
    weight_sentences(sentOnly=True)