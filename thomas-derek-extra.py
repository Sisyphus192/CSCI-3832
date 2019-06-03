import re
import string
from collections import Counter
from math import log, ceil
import operator
from itertools import islice


classes = ["T", "F"]

sent_classes = ['POS', 'NEG']

def load_documents_sent():
	with open('hotelPosT-train.txt', 'r') as infile:
		posT = infile.read()
	with open('hotelNegT-train.txt', 'r') as infile:
		negT = infile.read()
	posT = dict(zip(re.findall(r'ID-[0-9]{4,}', posT),
					re.findall(r'(?<=\t).*', posT)))
	negT = dict(zip(re.findall(r'ID-[0-9]{4,}', negT),
					re.findall(r'(?<=\t).*', negT)))

	return posT, negT

def parse_documents_sent(documents):
	parsed_docs = {}
	for doc in documents:
		d = documents[doc].lower()
		d = re.sub(r'[\(\)]', '', d) # parentheses are not needed and will screw up the regex
		d = re.sub('do not disturb', 'DND', d) # do not disturb is a phrase specific to hotels, I replace it with the dnd token so that the 'not' doesnt trigger the negation
		d = re.sub('wouldnt', "wouldn't", d) # fixes edge case
		negations = re.findall(r"(?<=never\s)[^?.;,!]*[?.;,!]|(?<=not\s)[^?.;,!]*[?.;,!]|(?<=no\s)[^?.;,!]*[?.;,!]|(?<=n't\s)[^?,;.!]*[?.;,!]", d)
		for n in negations:
			negated_words = ''.join(['NOT_'+x+' ' for x in n.split(' ')])
			d = re.sub(n, negated_words, d)
		d = list(x.strip(string.punctuation) for x in d.split(' ') if x != '')
		if '' in d: # some empty strings result from striping punctuation
			d.remove('')
		parsed_docs[doc] = d

	return parsed_docs

def train_naive_bayes_sent(posT, negT):
	logprior = {}
	loglikelihood = {}
	V = Counter([item for sublist in list(posT.values()) for item in sublist]+[item for sublist in list(negT.values()) for item in sublist])
	Vpos = Counter([item for sublist in list(posT.values()) for item in sublist])
	Vneg = Counter([item for sublist in list(negT.values()) for item in sublist])
	num_doc = len(posT) + len(negT)
	for class_ in sent_classes:
		if class_ == 'POS':
			num_c = len(posT)
		elif class_ == 'NEG':
			num_c = len(negT)
		logprior[class_] = log(num_c / num_doc)
		loglikelihood[class_] = {}
		for word in V:
			if class_ == 'POS':
				if word in Vpos:
					count = Vpos[word]
				else:
					count = 0
			elif class_ == 'NEG':
				if word in Vneg:
					count = Vneg[word]
				else:
					count = 0
			loglikelihood[class_][word] = log((count + 1) / (sum([V[w] for w in V]) + len(V)))
	return logprior, loglikelihood, V

def test_naive_bayes_sent(testdoc, logprior, loglikelihood, V):
	sum_ = {}
	for class_ in sent_classes:
		sum_[class_] = logprior[class_]
		for word in testdoc:
			if word in V:
				sum_[class_] += loglikelihood[class_][word]
	return max(sum_.items(), key=operator.itemgetter(1))[0]

def load_documents():
    with open("hotelT-train.txt", "r") as infile:
        hotelT = infile.read()
    with open("hotelF-train.txt", "r") as infile:
        hotelF = infile.read()
    with open("hotelDeceptionTest.txt", "r") as infile:
        hotel_test = infile.read()
    hotelT = dict(
        zip(re.findall(r"ID-[0-9]{4,}", hotelT), re.findall(r"(?<=\t).*", hotelT))
    )
    hotelF = dict(
        zip(re.findall(r"ID-[0-9]{4,}", hotelF), re.findall(r"(?<=\t).*", hotelF))
    )
    hotel_test = dict(
        zip(re.findall(r"ID-[0-9]{4,}", hotel_test), re.findall(r"(?<=\t).*", hotel_test))
    )

    # the below takes advantage of the fact that in python3.6+ dicts are ordered
    # this may not work in earlier versions of python3
    #hotelT_train = dict(islice(hotelT.items(), 0, ceil(len(hotelT) * 0.8)))
    #hotelT_test = dict(islice(hotelT.items(), ceil(len(hotelT) * 0.8), len(hotelT) + 1))
    #hotelF_train = dict(islice(hotelF.items(), 0, ceil(len(hotelF) * 0.8)))
    #hotelF_test = dict(islice(hotelF.items(), ceil(len(hotelF) * 0.8), len(hotelF) + 1))

    #print("# of hotelT_train, # of hotelT_test")
    #print(len(hotelT_train), len(hotelT_test))
    #print("# of hotelF_train, # of hotelF_test")
    #print(len(hotelF_train), len(hotelF_test))
    #print(
    #    "Is there an overlap between the hotelT train and test sets? (if true, use python3.6+)"
    #)
    #print(bool(set(hotelT_train.keys()) & set(hotelT_test.keys())))
    #print(
    #    "Is there an overlap between the hotelF train and test sets? (if true, use python3.6+)"
    #)
    #print(bool(set(hotelF_train.keys()) & set(hotelF_test.keys())))

    return hotelT, hotelF, hotel_test #hotelT_train, hotelT_test, hotelF_train, hotelF_test


def parse_documents(documents, posT, negT, logprior_sent, loglikelihood_sent, V_sent):
    parsed_docs = {}
    for doc in documents:
        d = documents[doc].lower()
        d = re.sub(
            r"[\(\)]", "", d
        )  # parentheses are not needed and will screw up the regex
        d = re.sub(r"[\[\]']+",'', d) # brackets are screwing with our regex
        d = re.sub(
            "do not disturb", "DND", d
        )  # do not disturb is a phrase specific to hotels, I replace it with the dnd token so that the 'not' doesnt trigger the negation
        d = re.sub("wouldnt", "wouldn't", d)  # fixes edge case
        negations = re.findall(
            r"(?<=never\s)[^?.;,!]*[?.;,!]|(?<=not\s)[^?.;,!]*[?.;,!]|(?<=no\s)[^?.;,!]*[?.;,!]|(?<=n't\s)[^?,;.!]*[?.;,!]",
            d,
        )
        for n in negations:
            negated_words = "".join(["NOT_" + x + " " for x in n.split(" ")])
            d = re.sub(n, negated_words, d)
        d = list(x.strip(string.punctuation) for x in d.split(" ") if x != "")
        if "" in d:  # some empty strings result from striping punctuation
            d.remove("")
        if doc in posT:
        	d.append("POS")
        elif doc in negT:
        	d.append("NEG")
        else:
        	d.append(test_naive_bayes_sent(d, logprior_sent, loglikelihood_sent, V_sent))
        parsed_docs[doc] = d

    return parsed_docs


def train_naive_bayes(hotelT, hotelF):
    logprior = {}
    loglikelihood = {}

    # implement binary NB
    #hotelT = {k : set(v) for k,v in hotelT.items()}
    #hotelF = {k : set(v) for k,v in hotelF.items()}
    V = Counter(
        [item for sublist in list(hotelT.values()) for item in sublist]
        + [item for sublist in list(hotelF.values()) for item in sublist]
    )
    Vpos = Counter([item for sublist in list(hotelT.values()) for item in sublist])
    Vneg = Counter([item for sublist in list(hotelF.values()) for item in sublist])
    num_doc = len(hotelT) + len(hotelF)
    for class_ in classes:
        if class_ == "T":
            num_c = len(hotelT)
        elif class_ == "F":
            num_c = len(hotelF)
        logprior[class_] = log(num_c / num_doc)
        loglikelihood[class_] = {}
        for word in V:
            if class_ == "T":
                if word in Vpos:
                    count = Vpos[word]
                else:
                    count = 0
            elif class_ == "F":
                if word in Vneg:
                    count = Vneg[word]
                else:
                    count = 0
            loglikelihood[class_][word] = log(
                (count + 1) / (sum([V[w] for w in V]) + len(V))
            )
    return logprior, loglikelihood, V


def test_naive_bayes(testdoc, logprior, loglikelihood, V):
    sum_ = {}
    for class_ in classes:
        sum_[class_] = logprior[class_]
        for word in testdoc:
            if word in V:
                sum_[class_] += loglikelihood[class_][word]
    return max(sum_.items(), key=operator.itemgetter(1))[0]

posT, negT = load_documents_sent()
posT, negT = parse_documents_sent(posT), parse_documents_sent(negT)
logprior_sent, loglikelihood_sent, V_sent = train_naive_bayes_sent(posT, negT)


hotelT_train, hotelF_train, hotel_test = load_documents()
hotelT_train, hotelF_train, hotel_test = (
    parse_documents(hotelT_train, posT, negT, logprior_sent, loglikelihood_sent, V_sent),
    parse_documents(hotelF_train, posT, negT, logprior_sent, loglikelihood_sent, V_sent),
    parse_documents(hotel_test, posT, negT, logprior_sent, loglikelihood_sent, V_sent),
)

logprior, loglikelihood, V = train_naive_bayes(hotelT_train, hotelF_train)
with open("thomas-derek-extra-out.txt", "w") as out_file:
    for t in hotel_test:
        out_file.write(
            t
            + " "
            + test_naive_bayes(hotel_test[t], logprior, loglikelihood, V)
            + "\n"
        )


"""
t_correct = 0
f_correct = 0
for t in hotelT_test:
	if test_naive_bayes(hotelT_test[t], logprior, loglikelihood, V) == "T":
		t_correct += 1
print("hotelT Accuracy: {}%".format((t_correct/len(hotelT_test))*100))

for t in hotelF_test:
	if test_naive_bayes(hotelF_test[t], logprior, loglikelihood, V) == "F":
		f_correct += 1
print("hotelF Accuracy: {}%".format((f_correct/len(hotelF_test))*100))

print("Total Accuracy: {}%".format(((t_correct + f_correct) / (len(hotelT_test) + len(hotelF_test)))*100))
"""

