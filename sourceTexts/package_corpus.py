import numpy
import pickle

def process_numpy_corpus():
    vocabFile = open('lawrence-vocabulary.txt', 'r')
    trainFile = open('lawrence-train-corpus.txt', 'r')

    vocabLines = vocabFile.readlines()
    trainLines = trainFile.readlines()

    vocabList = []
    trainList = []

    for line in vocabLines:
        tmp = line.replace("\n", "")
        vocabList.append(tmp)
        
    for line in trainLines:
        tmp = line.replace("\n", "")
        trainList.append(tmp.split())

    for i in range(len(trainList)):
        print(trainList[i])
    
        for j in range(len(trainList[i])):
            trainList[i][j] = vocabList.index(trainList[i][j])
            print(trainList[i][j])

    trainData = []

    for i in range(len(trainList)):
        if len(trainList[i]) > 30:
            print("input sentence longer that 30 words. Skipping")
        else:
            if len(trainList[i]) < 30:
                print("padded input sentence at index %d with %d null-space tokens.\n"
                      "original input length: %d" % (i, 30-len(trainList[i]), len(trainList[i])))
            
                for j in range(30 - len(trainList[i])):
                    trainList[i].append(0)
                
            trainData.append(trainList[i])

    trainNumpy = numpy.array(trainData)
    numpy.save("./data/train_corpus_padded.npy", trainNumpy);

    with open("./data/train_corpus_padded.txt", 'w') as fd:
        print(trainNumpy.tolist(), file=fd)

    print("Processing Complete.")

    vocabFile.close()
    trainFile.close()

def process_chars_pickle():
    vocabFile = open('./lawrence-vocabulary.txt', 'r')
    vocabLines = vocabFile.readlines()
    vocabList = []

    for line in vocabLines:
        tmp = line.replace("\n", "")
        vocabList.append(tmp)

    with open('./data/chars.pkl', 'wb') as fd:
        pickle.dump(vocabList, fd)

    with open('./data/chars.txt', 'w') as fd:
        print(vocabList, file=fd)

def gen_parlengths():
    fd = open("lawrence-first-novel.txt", 'r')

    file_data = fd.readlines()
    par_data = []
    
    par_flag = 0
    par_length = 0

    for line in file_data:
        if line == '\n':
            par_data.append(par_length)
            par_length = 0
        else:
            for i in range(len(line)):
                if line[i].startswith(".") or \
                   line[i].startswith("?") or \
                   line[i].startswith("!"):
                    par_length += 1

    par_data = numpy.array(par_data)
    numpy.save("./data/par_lengths.npy", par_data)

    with open('./data/par_lengths.txt', 'w') as fd:
        print(par_data.tolist(), file=fd)
        print("%s, MaxParLength: %s " % (par_data, max(par_data)))
        print("Number of Paragraphs: %s " % (len(par_data.tolist())))
        
def main():
    process_chars_pickle()
    process_numpy_corpus()
    gen_parlengths()

main()
