# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
imikolov's simple dataset: http://www.fit.vutbr.cz/~imikolov/rnnlm/

Complete comments.
"""
from common import download
import collections
import tarfile
import functools

__all__ = ['train', 'test', 'build_dict']

URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def fetch():
    return download(URL, "imikolov", MD5)


def open_file(filename):
    tf = tarfile.open(fetch())
    f = tf.extractfile(filename)
    return f


__train_file__ = './simple-examples/data/ptb.train.txt'
__test_file__ = './simple-examples/data/ptb.valid.txt'


def build_dict():
    trainf = open_file(__train_file__)
    testf = open_file(__test_file__)
    word_freq = word_count(testf, word_count(trainf))
    if '<unk>' in word_freq:
        # remove <unk> for now, since we will set it as last index
        del word_freq['<unk>']

    TYPO_FREQ = 50
    word_freq = filter(lambda x: x[1] > TYPO_FREQ, word_freq.items())
    word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_freq_sorted))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)

    trainf.close()
    testf.close()

    return word_idx


def reader_creator(word_idx, n, filename):
    def reader():
        f = open_file(filename)
        UNK = word_idx['<unk>']
        for l in f:
            l = ['<s>'] + l.strip().split() + ['<e>']
            if len(l) >= n:
                l = [word_idx.get(w, UNK) for w in l]
                for i in xrange(n, len(l) + 1):
                    yield l[i - n:i]
        f.close()

    return reader


train = functools.partial(reader_creator, filename=__train_file__)
test = functools.partial(reader_creator, filename=__test_file__)
