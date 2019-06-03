
from collections import Counter
import random

# tested bigram probabilites with sentences from lecture slide
test_corpus_1 = '''
<s> I am Sam </s>
<s> Sam I am </s>
<s> I do not like green eggs and ham </s>
'''

class LanguageModel:
	def __init__(self, file_path=None, smoothing=True,):
		if file_path:
			self.load_corpus(file_path)
		else:
			self.corpus = None
			self.words = None
			self.freqs = None
			self.numTokens = None
			self.prod = None
			self.bigrams = None

		self.smoothing = smoothing

	def load_corpus(self, file_path):
		if file_path == 'test_corpus_1':
			self.corpus = test_corpus_1
		else:
			with open(file_path, "r") as infile:
				self.corpus = infile.read()
		self.words = self.corpus.split() # split corpus into words
		self.freqs = Counter(self.words) # count frequencies of words
		self.freqs['<unk>'] = 0 # add <unk> to the dictionary

		for k, v in list(self.freqs.items()):
			if v == 1:
				self.freqs['<unk>'] += 1	# increment the frequency of <unk> delete all tokens with a frequency of 1
				del self.freqs[k]

		self.numTokens = sum(self.freqs.values())
	
	def uniProb(self, word):
		if self.freqs and self.numTokens:
			return self.freqs[word] / self.numTokens

	def product(self, nums):
		prod = 1
		for num in nums: 
			prod = prod * num
		return prod

	def unigramModel(self, sentence):
		sentence = sentence.split()
		for i in range(len(sentence)):
			if sentence[i] not in self.freqs:
				sentence[i] = '<unk>'

		return self.product(self.uniProb(word) for word in sentence)

	def bigramProb(self, bigram):
		if self.smoothing:
			return (self.corpus.count('{} {}'.format(bigram[0], bigram[1])) + 1)  / (self.corpus.count(bigram[0]) + len(self.freqs))
		else:
			return self.corpus.count('{} {}'.format(bigram[0], bigram[1]))  / self.corpus.count(bigram[0])

	def bigramModel(self, sentence):
		sentence = sentence.split()
		for i in range(len(sentence)-1):
			sentence[i] = (sentence[i], sentence[i+1])
		sentence.pop(-1)
		return self.product(self.bigramProb(bigram) for bigram in sentence)

	def shannon(self):
		start = '<s>'
		sentence = ''
		while start != '</s>':
			bigram_freqs = {}
			for i in list(self.freqs.keys()):
				if i != '<s>':
					bigram_freqs[(start,i)] = self.bigramProb((start,i)) # generate dict of all possible bigrams
			
			total = sum(bigram_freqs.values())
			for i in bigram_freqs:
				bigram_freqs[i] /= total # normalize frequencies

			bigram = random.choices(list(bigram_freqs.keys()), list(bigram_freqs.values()))
			
			sentence += bigram[0][0] + ' '
			
			start = bigram[0][1]

		sentence += start
		return sentence



def main():
	LM = LanguageModel('berp-training.txt')

	with open('berp-100-test.txt', "r") as infile:
				test_corpus = infile.read()

	with open("Thomas-Derek-assgn2-unigram-out.txt", "wt") as outfile:
				for i in test_corpus.split('\n'):
					outfile.write(str(LM.unigramModel(i))+'\n')
	with open("Thomas-Derek-assgn2-bigram-out.txt", "wt") as outfile:
				for i in test_corpus.split('\n'):
					outfile.write(str(LM.bigramModel(i))+'\n')
	with open("Thomas-Derek-assgn2-bigram-rand-corpus.txt", "a") as outfile:
				for i in range(89):
					outfile.write(LM.shannon()+'\n')

	

if __name__ == '__main__':
	main()