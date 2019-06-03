
import re
import string
from collections import Counter
from math import log
import operator

sent_classes = ['POS', 'NEG']

def load_documents_sent():
	with open('hotelPosT-train.txt', 'r') as infile:
		posT = infile.read()
	with open('hotelNegT-train.txt', 'r') as infile:
		negT = infile.read()
	with open('HW3-testset.txt', 'r') as infile:
		test = infile.read()
	posT = dict(zip(re.findall(r'ID-[0-9]{4,}', posT),
					re.findall(r'(?<=\t).*', posT)))
	negT = dict(zip(re.findall(r'ID-[0-9]{4,}', negT),
					re.findall(r'(?<=\t).*', negT)))
	test = dict(zip(re.findall(r'ID-[0-9]{4,}', test),
					re.findall(r'(?<=\t).*', test)))

	return posT, negT, test

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
		d = set(x.strip(string.punctuation) for x in d.split(' ') if x != '')
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


#posT, negT, test = load_documents()
#posT, negT, test = parse_documents(posT), parse_documents(negT), parse_documents(test)
#logprior, loglikelihood, V = train_naive_bayes(posT, negT)
#with open('thomas-derek-assgn3-out.txt', 'w') as out_file:
#	for t in test:
#		out_file.write(t + ' ' + test_naive_bayes(test[t], logprior, loglikelihood, V) + '\n')

