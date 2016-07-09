# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


class BatchGenerator(object):
    def __init__(self, text, vocabulary, batch_size=64, num_unrollings=10):

        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        self.first_letter = ord(string.ascii_lowercase[0])
        self.batch_size = batch_size
        self.num_unrollings = num_unrollings-1

        self._text = text
        self._text_size = len(text)
        self._batch_size = self.batch_size
        self._num_unrollings = self.num_unrollings
        segment = self._text_size // self.batch_size
        self._cursor = [offset * segment for offset in range(self.batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self.vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, self.char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


    def characters(self, probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation."""
        return [self.id2char(c) for c in np.argmax(probabilities, 1)]


    def batches2string(self, batches):
        """Convert a sequence of batches back into their (most likely) string
        representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, self.characters(b))]
        #for s_i in s:
            #s_i.decode('utf-8')
        return s


    def char2id(self, char):
        if char in self.vocabulary:
            return self.vocabulary.index(char)
        else:
            print('Unexpected character: %s' % char)
            return 0

    def id2char(self, dictid):
        return self.vocabulary[dictid]