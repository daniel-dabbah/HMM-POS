#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:18:12 2024

@author: danieldabbah
"""
import nltk
from nltk.corpus import brown
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import collections
import random

np.seterr(divide='ignore')

Token = collections.namedtuple('Token', ['word', 'tag'])
EmissionPair = collections.namedtuple('EmissionPair', ['word', 'tag'])
LabelsPair = collections.namedtuple('LabelsPair', ['label1', 'label2'])

special_tag_pattern = re.compile(r'[-+*$]')
hour_pattern = re.compile(r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$')

nltk.download('brown')


class Mle():
    """ this class represent most likely tag model. 
    """

    def __init__(self):

        self.counter = collections.Counter()
        self.known_words = set()

    def fit(self, train):

        # update the counter:
        for row in train:

            for token in row:

                self.known_words.add(token.word)

            self.counter.update(row)

    def infer(self, word):
        if word in self.known_words:
            first_word_pairs = {k: v for k,
                                v in self.counter.items() if k.word == word}

            return (max(first_word_pairs, key=first_word_pairs.get).tag)
        else:
            return 'NN'

    def get_test_result(self, test):

        known_words = 0
        unknown_words = 0
        correct_tag_known_words = 0
        correct_tag_unknown_words = 0

        for row in test:

            # we dont have to check the START_TOKEN and the END_TOKEN:
            for word, tag in row[1:-1]:

                if word in self.known_words:
                    known_words += 1

                    if tag == self.infer(word):
                        correct_tag_known_words += 1
                else:
                    unknown_words += 1
                    if tag == self.infer(word):
                        correct_tag_unknown_words += 1

        known_accuracy = correct_tag_known_words/known_words
        unknown_accuracy = correct_tag_unknown_words/unknown_words
        total_accuracy = (correct_tag_known_words +
                          correct_tag_unknown_words) / (known_words + unknown_words)

        return known_accuracy, unknown_accuracy, total_accuracy


class BigramHmm():
    """ represent the bigram hmm model. the fit mehod can
    receive a threshold paramater, if we want to use Pseudo Words.
    In addition, the 'get_test_result' function receive a boolean paramater
    that decide if to add laplace smoothing to the emission probabilites."""

    def __init__(self):

        self.counter = collections.Counter()
        self.label_pairs_counter = collections.Counter()
        self.labels_counter = collections.Counter()
        self.words_counter = collections.Counter()
        self.known_words = set()

        self.vocabulary = set()

        self.emissions_probs = {}
        self.transition_probs = {}

        self.labels = []

        self.ps_words_counter = collections.Counter()

    def fit(self, train, pseudo_threshold=0):

        if pseudo_threshold:
            # we have to count the frequency of each word:
            for row in train:
                for token in row:

                    if token.word != None:
                        self.words_counter[token.word] += 1

            train = self.text2pseudo(train, pseudo_threshold)

        # update the counter:

        labels = set()

        for row in train:

            for token in row:

                if token.word != None:
                    # None represent START_TOKEN and END_TOKEN
                    self.known_words.add(token.word)
                    # we add also the word to the vocabulary as well,
                    # becuse we will add words that appers in the test set to the vocabulary as well.

                    self.vocabulary.add(token.word)

                    self.ps_words_counter[token.word] += 1

                self.labels_counter[token.tag] += 1
                labels.add(token.tag)
            labels_pairs = [LabelsPair(row[i].tag, row[i+1].tag)
                            for i in range(len(row)-1)]

            self.label_pairs_counter.update(labels_pairs)

            self.counter.update(row)

        self.labels = list(labels)

    def get_emission_prob(self, word, tag, add_one_smoothing=False):

        emission_pair = EmissionPair(word, tag)

        if add_one_smoothing:

            if emission_pair in self.emissions_probs:
                return self.emissions_probs[emission_pair]

            # The smoothing:
            emission_count = self.counter[emission_pair] + 1

            tag_count = self.labels_counter[emission_pair.tag] + \
                len(self.vocabulary)
            prob = emission_count/tag_count

            self.emissions_probs[emission_pair] = np.log(prob)

            return self.emissions_probs[emission_pair]

        # if we didnt saw the word, we give the max probabilty:
        if emission_pair.word not in self.known_words:
            return np.log(1)

        if emission_pair in self.emissions_probs:
            return self.emissions_probs[emission_pair]
        else:
            # calculate the log probability of the emission pair:
            emission_count = self.counter[emission_pair]

            tag_count = self.labels_counter[emission_pair.tag]

            prob = 0 if tag_count == 0 else emission_count/tag_count

            self.emissions_probs[emission_pair] = np.log(prob)

            return self.emissions_probs[emission_pair]

    def get_transition_prob(self, label1, label2):

        labels_pair = LabelsPair(label1, label2)

        if labels_pair in self.transition_probs:
            return self.transition_probs[labels_pair]
        else:
            labels_count = self.label_pairs_counter[labels_pair]

            first_tag_count = self.labels_counter[labels_pair.label1]

            prob = 0 if first_tag_count == 0 else labels_count/first_tag_count

            self.transition_probs[labels_pair] = np.log(prob)

            return self.transition_probs[labels_pair]

    def viterbi(self, words, add_one_smoothing=False):

        prob_df = pd.DataFrame(index=self.labels, columns=words)
        trace_df = pd.DataFrame(index=self.labels, columns=words)

        for word_id, word in enumerate(words):

            if word_id == 0:

                # fill the first column:
                for i, label in enumerate(self.labels):

                    # For each label, there is only one possible previus tag(START_TAG)
                    score = self.get_transition_prob(
                        "START_TAG", label) + self.get_emission_prob(word, label, add_one_smoothing)

                    prob_df.iloc[i, word_id] = score
            # fill the others columns:
            else:

               # for each word, we have to fill the column:
                for i, curr_label in enumerate(self.labels):
                    # for each cell, we have to check all previus labels:
                    scores = [(prob_df.iloc[j, word_id-1]+self.get_transition_prob(prev_label, curr_label) +
                              self.get_emission_prob(word, curr_label, add_one_smoothing)) for j, prev_label in enumerate(self.labels)]

                    # we take the min becuase we are working with log probability:

                    prob_df.iloc[i, word_id] = max(scores)
                    trace_df.iloc[i, word_id] = np.argmax(scores)

        # get the labels:
        predicted_labels = []

        best_label_id = np.argmax(prob_df.iloc[:, -1])

        back_pointer = trace_df.iloc[best_label_id, -1]
        predicted_labels.append(self.labels[back_pointer])

        for word_id in reversed(range(1, len(words)-1)):

            back_pointer = trace_df.iloc[back_pointer, word_id]
            predicted_labels.append(self.labels[back_pointer])

        return predicted_labels[::-1]

    def get_test_result(self, test, add_one_smoothing=False, pseudo_threshold=0):

        confusion_matrix = pd.DataFrame(
            0, index=self.labels, columns=self.labels)

        if pseudo_threshold:
            test = self.text2pseudo(test, pseudo_threshold)

        # add unseen words to the vocabulary. (we will need this if we do add-one smoothing)

        for row in test:

            for token in row[1:-1]:

                self.vocabulary.add(token.word)

        known_words = 0
        unknown_words = 0
        correct_tag_known_words = 0
        correct_tag_unknown_words = 0

        for row in test:

            words = [token.word for token in row][1:]

            # we dont have to check the START_TOKEN and the END_TOKEN:
            predicted_labels = self.viterbi(words, add_one_smoothing)
            tokens_to_eval = [token for token in row[1:-1]]

            for test_token, predicted_label in zip(tokens_to_eval, predicted_labels):

                if test_token.word in self.known_words:
                    known_words += 1

                    if test_token.tag in self.labels:

                        if test_token.tag == predicted_label:
                            correct_tag_known_words += 1
                        else:
                            confusion_matrix.loc[test_token.tag,
                                                 predicted_label] += 1
                else:
                    unknown_words += 1
                    if test_token.tag in self.labels:
                        if test_token.tag == predicted_label:
                            correct_tag_unknown_words += 1
                        else:
                            confusion_matrix.loc[test_token.tag,
                                                 predicted_label] += 1

        known_accuracy = correct_tag_known_words/known_words
        unknown_accuracy = correct_tag_unknown_words/unknown_words
        total_accuracy = (correct_tag_known_words +
                          correct_tag_unknown_words) / (known_words + unknown_words)

        return known_accuracy, unknown_accuracy, total_accuracy, confusion_matrix

    def text2pseudo(self, sents, threshold):
        """ change the given sentances to new sentencess with the pseudo words.
        """

        for i in range(len(sents)):
            new_row = [Token(None, "START_TAG")]

            # we dont have to chnage START_TOKEN and END_TOKEN:
            for word, tag in sents[i][1:-1]:

                if self.words_counter[word] < threshold:
                    new_row.append(Token(word2pseudo(word), tag))
                else:
                    new_row.append(Token(word, tag))

            new_row.append(Token(None, "END_TAG"))
            sents[i] = new_row

        return sents


def get_train_test(train_size=.9):

    sents = brown.tagged_sents(categories='news')
    data = np.array(sents, dtype="object")

    data = change_special_tags(data)

    split_index = int(len(data) * train_size)

    return data[:split_index], data[split_index:]


def deal_with_special_tag(tag):

    return re.split(pattern=special_tag_pattern, string=tag, maxsplit=1)[0]


def change_special_tags(sents):
    for i in range(len(sents)):
        sents[i] = [Token(None, "START_TAG")] + [Token(word, deal_with_special_tag(tag))
                                                 for word, tag in sents[i]]+[Token(None, "END_TAG")]
    return sents


def word2pseudo(word):
    """ change the words to a pseudo word.
    """

    if '$' in word:
        return "MONEY_CLASS"

    if '%' in word:
        return "PERCENRAGE_CLASS"

    if word.replace(',', '').isdigit():
        if len(word) == 1:
            return "ONE_DIGIT_NUM_CLASS"
        if len(word) == 2:
            return "TWO_DIGITS_NUM_CLASS"
        if len(word) == 3:
            return "THREE_DIGITS_NUM_CLASS"
        if len(word) == 4:
            return "FOUR_DIGITS_NUM_CLASS"
        else:
            return "NUM_CLASS"

    if bool(re.match(hour_pattern, word)):
        return "HOUR_CLASS"

    if word.endswith('th'):
        return "ENDS_WITH_TH_CLAAS"

    if word.endswith('ing'):
        return "ENDS_WITH_ING_CLAAS"

    if word.isupper():
        return "ALL_CAPS_CLASS"

    # try to catch plural nouns:
    if word.endswith('s'):
        return "ENDS_WITH_S_CLASS"

    else:
        return word


if __name__ == '__main__':

    train, test = get_train_test()

    # Q2: BaseLine:
    mle = Mle()
    mle.fit(train)

    known_accuracy, unknown_accuracy, total_accuracy = mle.get_test_result(
        test)

    print("MLE Model:")
    print(f"Error rate for known words is: {1-known_accuracy}")
    print(f"Error rate for unknown words is: {1-unknown_accuracy}")
    print(f"Total Eroor rate is: {1-total_accuracy}")

    # Q3: bigram HMM tagger:

    bigram_hmm = BigramHmm()
    bigram_hmm.fit(train)

    known_accuracy, unknown_accuracy, total_accuracy, confusion_matrix = bigram_hmm.get_test_result(
        test)
    print("Without Smoothing:")
    print(f"Error rate for known words is: {1-known_accuracy}")
    print(f"Error rate for unknown words is: {1-unknown_accuracy}")
    print(f"Total Eroor rate is: {1-total_accuracy}")

    # Q4: bigram HMM tagger using Add-one smoothing:

    bigram_hmm = BigramHmm()
    bigram_hmm.fit(train)

    known_accuracy, unknown_accuracy, total_accuracy, confusion_matrix = bigram_hmm.get_test_result(
        test, add_one_smoothing=True)
    print("With Smoothing:")
    print(f"Error rate for known words is: {1-known_accuracy}")
    print(f"Error rate for unknown words is: {1-unknown_accuracy}")
    print(f"Total Eroor rate is: {1-total_accuracy}")

    # Q5: bigram HMM tagger using Pseudo Words:

    threshold = 5

    pseudo_bigram_hmm = BigramHmm()
    pseudo_bigram_hmm.fit(train, pseudo_threshold=threshold)

    known_accuracy, unknown_accuracy, total_accuracy, confusion_matrix = pseudo_bigram_hmm.get_test_result(
        test,  pseudo_threshold=threshold)
    print("With Pseudo Words:")
    print(f"Error rate for known words is: {1-known_accuracy}")
    print(f"Error rate for unknown words is: {1-unknown_accuracy}")
    print(f"Total Eroor rate is: {1-total_accuracy}")

    # with add_one_smoothing:

    pseudo_bigram_hmm = BigramHmm()
    pseudo_bigram_hmm.fit(train, pseudo_threshold=threshold)

    known_accuracy, unknown_accuracy, total_accuracy, confusion_matrix = pseudo_bigram_hmm.get_test_result(
        test, add_one_smoothing=True, pseudo_threshold=threshold)
    print("With Pseudo Words and Add One smoothing:")
    print(f"Error rate for known words is: {1-known_accuracy}")
    print(f"Error rate for unknown words is: {1-unknown_accuracy}")
    print(f"Total Eroor rate is: {1-total_accuracy}")

    plt.figure(figsize=(60, 50))
    heatmap = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
    heatmap.set_xlabel('Predicted Tag', fontsize=50)
    heatmap.set_ylabel('True Tag', fontsize=50)

    font_properties = {'fontsize': 13, 'rotation': 45, 'ha': 'right'}
    heatmap.set_xticklabels(heatmap.get_xticklabels(),
                            fontdict=font_properties)
    heatmap.set_yticklabels(heatmap.get_xticklabels(),
                            fontdict=font_properties)
    plt.savefig('confusion_matrix_plot.png')
