from __future__ import print_function
import fileinput
from glob import glob
import sys
import evalNER
import re

from seqlearn.datasets import load_conll
from seqlearn.evaluation import bio_f_score
from seqlearn.perceptron import StructuredPerceptron
from sklearn.metrics import accuracy_score

#features to tag for
def features(sentence, i):
    #split off line number; not needed
    word = sentence[i].split()[1]
    print(word)

    yield "word:{}" + word.lower()
    
    #if word starts with capital letter
    #also excludes first word of sentence
    if (word[0].isupper() == True):
        if (i != 0):
            yield "CAPITAL"

    #if word is more than 1 capital letter
    if (re.search((r'([A-Z])+'),word)) == True:
        yield "ALLCAPS"

    #if word is Cap, lowercase, then another cap presetnt
    #i.e. IxB
    if (re.search((r'([A-Z]+[a-z]+[A-Z]+[a-zA-Z]*)'),word)) == True:
        yield "CAPLOWCAP"
    
    #if word is human:
    if (word == 'human'):
        yield "HUMAN"

    #if word is c:
    if (word == 'c'):
        yield "C"

    #if word has/is a number
    if (any(char.isdigit() for char in word) == True):
        yield "NUMBER"

    #if acronym exists, using Regex
    if (re.search((r'(?:[a-zA-Z]\.)+'), word) == True):
        yield "ACRONYM"

    #if we get hypen, yield hypen, previous word, next word
    if (word == '-'):
        yield "HYPHEN"
    #yield word before hyphen
    if (i+1 < len(sentence)):
        if (sentence[i+1].split()[1] == '-'):
            yield "BEFOREHYPHEN"
    #yield word after hyphen
    #ensure we arent reference previous word as null
    if (i > 0):
        if (sentence[i-1].split()[1] == '-'):
            yield "AFTERHYPHEN"

    if i > 0:
        yield "word-1:{}" + sentence[i - 1].split("\t")[1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].split("\t")[1].lower()
            if i > 2:
                yield "word-3:{}" + sentence[i - 3].split("\t")[1].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].split("\t")[1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].split("\t")[1].lower()
            if i + 3 < len(sentence):
                yield "word+3:{}" + sentence[i + 3].split("\t")[1].lower()


def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))

def load_data(trainingPath, testPath):
    #files = glob('nerdata/*.bio')

    #load training file and run through conll sequencer
    print("Training data loaded from {0}".format(trainingPath))
    #only doing glob because the example did above the print statement for glob
    trainFiles = glob(trainingPath)
    train_files = [f for i, f in enumerate(trainFiles)]
    train = load_conll(fileinput.input(train_files), features)
    #training data and description
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)

    #Filler third column of testing data with fillers because load_conll wants 3 columns
    #We are only given 2 columns in test data, will not work without
    newTest = "newTest.txt"
    postTest = open(newTest, 'w+')
    with open(testPath) as file:
        for line in file:
            if (line != '\n'):
                #strip \n from end of file
                line = line.rstrip()
                #add filler to end of line
                postTest.write(line + '\t NonApplicable\n')
    postTest.close()

    #load test data
    print("Test data loaded from {0}".format(testPath))
    #again, only doing glob because the example did
    testFiles = glob(newTest)
    test_files = [f for i, f in enumerate(testFiles)]
    test = load_conll(fileinput.input(test_files), features)
    #test data and description
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    return train, test

if __name__ == "__main__":
    #print(__doc__)

    #print("Loading training data...", end=" ")
    #X_train, y_train, lengths_train = load_conll(sys.argv[1], features)
    #describe(X_train, lengths_train)

    #get all data
    training = "gene-trainF18.txt"
    testing = "test-run-test.txt"
    outputFile = "assign4-output.txt"
    train, test = load_data(training, testing)
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test

    #print("Loading test data...", end=" ")
    #X_test, y_test, lengths_test = load_conll(sys.argv[2], features)
    #describe(X_test, lengths_test)

    #actually train data sequence with perceptron in example
    clf = StructuredPerceptron(verbose=True, max_iter=10)
    print("Training %s" % clf)
    clf.fit(X_train, y_train, lengths_train)

    #predicted IOB tags sequence from test data
    y_pred = clf.predict(X_test, lengths_test)

    outputCopy = open(outputFile, 'w+')
    counter = 0
    with open(testing) as file:
        for line in file:
            if (line != '\n'):
                line = line.rstrip()
                outputCopy.write(line + '\t' + y_pred[counter] + '\n')
                counter += 1
            else:
                outputCopy.write('\n')
    outputCopy.close()

    #Martin's evalNER score script
    outputReal = open(outputFile, 'r')
    correctOutput = open("test-run-test-with-keys.txt", 'r')

    evalNER.eval(outputReal, correctOutput)
    outputReal.close()
    correctOutput.close()
    #print("Accuracy: %.3f" % (100 * accuracy_score(y_test, y_pred)))
    #print("CoNLL F1: %.3f" % (100 * bio_f_score(y_test, y_pred)))