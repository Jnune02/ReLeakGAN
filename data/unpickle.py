#!/usr/bin/python3

# Test to unpickle the LeakGAN data corpuses, and repickle my own into
# drop in replacements.

import sys
import pickle
import numpy
import json

def checkTrainCorpus():
    gen_corpus = open("train_corpus.npy", "rb")
    gen_data = numpy.load(gen_corpus)

    char_corpus = open("chars.pkl", "rb")
    char_data = pickle.load(char_corpus)

    conversionData = []

    for i in range(len(gen_data)):
        for j in range(len(gen_data[i])):
            conversionData.append(char_data[gen_data[i][j]])

    conversionFd = open("train.txt", "w")
    print(conversionData, file=conversionFd)
    
def checkGenCorpus():
    gen_corpus = open("gen_corpus.npy", "rb")
    gen_data = numpy.load(gen_corpus)

    char_corpus = open("chars.pkl", "rb")
    char_data = pickle.load(char_corpus)

    conversionData = []

    for i in range(len(gen_data)):
        for j in range(len(gen_data[i])):
            conversionData.append(char_data[gen_data[i][j]])

    conversionFd = open("conversion.txt", "w")
    print(conversionData, file=conversionFd)

def encodeToTensor():
    corpus = open("sentenceCorpus.pkl", "rb")
    corpus_data = pickle.load(corpus)
    corpus.close()

    corpus_data_np = numpy.array(corpus_data, dtype="object")
    numpy.save("corpus", corpus_data_np)

def extractEvalCorpus():
    corpus = numpy.load("eval_corpus.npy")
    fd_out = open("eval_corpus.dump", "w")
    print(corpus.tolist(), file=fd_out)

def extractGenCorpus():
    corpus = numpy.load("gen_corpus.npy")
    fd_out = open("gen_corpus.dump", "w")
    print(corpus.tolist(), file=fd_out)

def extractTestCorpus():
    corpus = numpy.load("test_corpus.npy")
    fd_out = open("test_corpus.dump", "w")
    print(corpus.tolist(), file=fd_out)

def extractTrainCorpus():
    corpus = numpy.load("train_corpus.npy", allow_pickle=True)
    fd_out = open("train_corpus.dump", "w")
    print(corpus.tolist(), file=fd_out)
    
def extractCorpus(data):
    corpus = numpy.load(data, allow_pickle=True)
    print(corpus.tolist())

def extractChars():
    depickle = open("chars.pkl", "rb")
    fd_out = open("chars.dump", "w")
    chars = pickle.load(depickle)

    print("vocab_size: %s" % len(chars))
    print(chars, file=fd_out)

def addSpaceToken():
    data = numpy.load("train_corpus_unpadded.npy", allow_pickle=True)
    rawData = data.tolist()
    
def padTrainCorpus():
    data = numpy.load("train_corpus_unpadded.npy", allow_pickle=True)
    rawData = data.tolist()

    # Remove anything longer than 60 items.
    for datum in list(rawData):
        if len(datum) >= 56:
            rawData.remove(datum)
    
    # Get max item length:
    padLength = max([len(rawData[i]) for i in range(len(rawData))])
    print("Max Array Length: %d " % padLength)

    if padLength >= 60:
        print("Max Array Length must be less than 60! Abort!")
        return
    
    for i in range(len(rawData)):        
        s = len(rawData[i])
        for _ in range(padLength - s):
            rawData[i].append(0)

        print("padded length of element at index %d: %d" % (i, len(rawData[i])))

    padData = numpy.array(rawData)
    numpy.save("train_corpus_padded", padData)
    
def main():
    #extractChars()
    #extractCorpus("train_corpus_padded.npy")
    #extractTestCorpus()
    #extractTrainCorpus()
    #extractGenCorpus()
    #checkGenCorpus()
    #extractEvalCorpus()
    #checkTrainCorpus()
    padTrainCorpus()

if __name__ == "__main__":
    main()
    
