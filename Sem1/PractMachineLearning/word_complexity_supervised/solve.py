import os
import sys

import string
import re

import numpy as np
import pandas as pd


import gensim.models
import gensim.downloader

from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *


class Sample:

	def __init__(self, Id, sentence, start_pos, end_pos, word, native, foreign, native_complex=-1, foreign_complex=-1, percentage=-1.0):

		self.id = Id
		self.sentence = sentence
		self.start_pos = start_pos
		self.end_pos = end_pos
		self.word = word
		self.native = native
		self.foreign = foreign
		self.native_complex = native_complex
		self.foreign_complex = foreign_complex
		self.percentage = percentage


# def read_sample(line):
# 	parts = line.split('\t')

# 	Id = int(parts[0])

# def get_features(X, model):

# 	for x in X:




if __name__ == "__main__":

	filename = "train_full.txt"


	df = pd.read_csv(filename, sep='\t', header=None)

	X = df[df.columns[0:7]]

	Y = df[df.columns[9]]

	
	# print(df)

	# all_data = df.to_numpy()
	# print(all_data.shape)


	X = X.to_numpy()
	Y = Y.to_numpy()

	# print(X)
	# print(Y)

	X_train, X_rest, y_train, Y_rest = train_test_split(X, Y, test_size=0.2, random_state=1337)

	X_valid, X_test, y_valid, y_test = train_test_split(X_rest, Y_rest, test_size=0.4, random_state=1337)

	# print(X_train.shape)
	# print(y_train.shape)
	# print(X_valid.shape)
	# print(y_valid.shape)
	# print(X_test.shape)
	# print(y_test.shape)

	# vectorizer = CountVectorizer()
	# vectorized_text = vectorizer.fit_transform(X_train[:,1])

	# print(vectorized_text)
	# print(X_train[:,1])


	# print(set(X_train[:,1]))

	sentences = list(set(X_train[:,1]))

	# new_sentences = []

	# for s in sentences:
	# 	new_sentences.append(re.sub("'", " ", s))

	# sentences = new_sentences

	# sentences = [x.translate(str.maketrans('', '', string.punctuation)).split() for x in sentences]

	print(word_tokenize(sentences))

	# sentences = [x.split() for x in sentences]


	# new_sentences = []

	# for sentece in sentences:
	# 	s = []
	# 	for word in sentence:
	# 		if re.search(".*'.*") is not None:



	# x.translate(str.maketrans('', '', string.punctuation))

	# new_string = a_string.translate(str.maketrans('', '', string.punctuation))

	# print(clean_sentences)


	# model = gensim.models.Word2Vec(sentences=np.array(sentences, dtype=object), vector_size=100, window=5, min_count=1, workers=4)
	
	# model = gensim.downloader.load('word2vec-google-news-300')

	# print()

	# print(X_train[2])
	# start_pos = X_train[0][2]
	# end_pos = X_train[0][3]


	# word = X_train[1][4]

		
	# print(feature)


	features = []

	for x in X_train:
		word = x[4]
		word = re.sub("'", " ", word)
		word = word.translate(str.maketrans('', '', string.punctuation))

		word_parts = word.split()

		print(str(x[0]) + " - " + str(word_parts))

		feature = 0

		if len(word_parts) > 1:
			feature = np.sum(np.array([model.wv[w] for w in word_parts]), axis=0)
		else:
			feature = model.wv[word_parts]
	
		features.append(feature)	

	features = np.array(features)

	# print(features.shape)



	# print(word)
	# print(model.wv.get_vector(word))

	# print(get_features(X_train[0], lambda x: np.zeros(2)))
















