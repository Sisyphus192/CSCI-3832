import re
import fileinput
from seqlearn.perceptron import StructuredPerceptron
from seqlearn.datasets import load_conll
import evalNER



def split_file():
    # used once to create training and testing sets
    with open("gene-trainF18.txt") as in_file:
        data = in_file.read().splitlines()
    for i in range(0, len(data)):
        if i <= len(data) * 0.8:
            with open("gene-trainF18_train.txt", "a+") as out_file:
                out_file.write(data[i] + "\n")
        if i > len(data) * 0.8:
            with open("gene-trainF18_test.txt", "a+") as out_file:
                out_file.write(data[i] + "\n")


def features(sentence, i):
    # split off line number; not needed
    word = sentence[i].split("\t")[1]

    yield "word:{}" + word.lower()

    #if word[0].isupper():
    #    yield "CAP"
    
    # if word starts with capital letter
    # also excludes first word of sentence
    if word[0].isupper():
        if i != 0:
            yield "CAPITAL"

    # if word has more than 1 capital letter
    if (re.search((r"([A-Z][A-Z])+"), word)):
        yield "ALLCAPS"

    # checks for technical abreviations
    if (re.search((r"[A-Z][A-Z]+[0-9]+"), word)):
        yield "SMALLTECHNICAL"

    # checks for big technical names, this is not perfect but close
    if (re.search((r"[A-Za-z]{5,}(ase|gen|lin)\b"), word)):
        yield "BIGTECHICAL"
    # if word is Cap, lowercase, then another cap presetnt
    # i.e. IxB
    if (re.search((r"([A-Z]+[a-z]+[A-Z]+[a-zA-Z]*)"), word)):
        yield "CAPLOWCAP"

    # if word is human:
    if word == "human":
        yield "HUMAN"

    # if word is c:
    if word == "c":
        yield "C"

    # if word has/is a number
    if any(char.isdigit() for char in word):
        yield "NUMBER"

    # if acronym exists, using Regex
    if re.search((r"(?:[a-zA-Z]\.)+"), word):
        yield "ACRONYM"

    # if we get hypen
    if word == "-":
        yield "HYPHEN"

    # yield word before hyphen
    if i + 1 < len(sentence):
        if sentence[i + 1].split()[1] == "-":
            yield "BEFOREHYPHEN"
    # yield word after hyphen
    # ensure we arent reference previous word as null
    if i > 0:
        if sentence[i - 1].split()[1] == "-":
            yield "AFTERHYPHEN"
    
    if i > 0:
        yield "word-1:{}" + sentence[i - 1].split("\t")[1].lower()
        if i > 1:
            yield "word-2:{}" + sentence[i - 2].split("\t")[1].lower()
            # didnt improve performance
            #if i > 2:
            #    yield "word-3:{}" + sentence[i - 3].split("\t")[1].lower()
                # didnt improve perfomance
                #if i > 3:
                #    yield "word-4:{}" + sentence[i - 4].split("\t")[1].lower()
    if i + 1 < len(sentence):
        yield "word+1:{}" + sentence[i + 1].split("\t")[1].lower()
        if i + 2 < len(sentence):
            yield "word+2:{}" + sentence[i + 2].split("\t")[1].lower()
            # didnt improve performance
            #if i + 3 < len(sentence):
            #    yield "word+3:{}" + sentence[i + 3].split("\t")[1].lower()
                # didnt improve performance
                #if i + 4 < len(sentence):
                #    yield "word+4:{}" + sentence[i + 4].split("\t")[1].lower()
    

def describe(X, lengths):
    print("{0} sequences, {1} tokens.".format(len(lengths), X.shape[0]))


def load_data(trainingPath, testPath):
    print("Loading training data...", end=" ")
    train = load_conll(fileinput.input(trainingPath), features)
    X_train, _, lengths_train = train
    describe(X_train, lengths_train)

    # Filler third column of testing data with fillers because load_conll wants 3 columns
    # We are only given 2 columns in test data, will not work without
    newTest = "newTest.txt"
    postTest = open(newTest, "w+")
    with open(testPath) as file:
        for line in file:
            if line != "\n":
                # strip \n from end of file
                line = line.rstrip()
                # add filler to end of line
                postTest.write(line + "\t NonApplicable\n")
    postTest.close()

    print("Loading test data...", end=" ")
    test = load_conll(fileinput.input(newTest), features)
    X_test, _, lengths_test = test
    describe(X_test, lengths_test)

    return train, test


if __name__ == "__main__":
    training = "gene-trainF18.txt"
    testing = "test-run-test.txt"
    outputFile = "assign4-output.txt"

    train, test = load_data(training, testing)
    X_train, y_train, lengths_train = train
    X_test, y_test, lengths_test = test
    score = 0
    for i in range(30):

        # train data sequence with perceptron in example
        clf = StructuredPerceptron(verbose=True, max_iter=10)
        print("Training %s" % clf)
        clf.fit(X_train, y_train, lengths_train)

        # predicted IOB tags sequence from test data
        y_pred = clf.predict(X_test, lengths_test)

        outputCopy = open(outputFile, "w+")
        counter = 0
        with open(testing) as file:
            for line in file:
                if line != "\n":
                    line = line.rstrip()
                    outputCopy.write(line + "\t" + y_pred[counter] + "\n")
                    counter += 1
                else:
                    outputCopy.write("\n")
        outputCopy.close()

        # Martin's evalNER score script
        outputReal = open(outputFile, "r")
        correctOutput = open("test-run-test-with-keys.txt", "r")

        score += evalNER.eval(outputReal, correctOutput)
        outputReal.close()
        correctOutput.close()

    print("score: ", score/30)
