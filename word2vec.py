import gensim
import os
from slither.slither import Slither
from slither.core.cfg.node import NodeType, Node
from split_words import split_words
import solidity
import json
import time
import gensim
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn import manifold
import numpy as np
from sklearn.manifold import TSNE
from slither.slithir.variables import Constant
from slither.core.solidity_types import ElementaryType, UserDefinedType, Type
from slither.core.declarations import solidity_variables
from slither.core.declarations import Contract
from slither.slithir.operations import TypeConversion
import re
import string

RESERVED = ["&&", "||", '!=', '==']


class Word2vec():
    def __init__(self, corpus_path:str):
        self.corpus_path = corpus_path
        if not os.path.exists(corpus_path):
            print('corpus path does not exist')
            return
        self.files = []
        for root, dirs, files in os.walk(self.corpus_path):
            for file in files:
                if file.endswith('.sol'):
                    full_path = os.path.join(root, file)
                    self.files.append(full_path)

    @staticmethod
    def replace_var_name(txt, old_var_name, new_var_name):
        # words = split_words(txt, RESERVED, lower=False)
        punctuation = string.punctuation.replace('_', '') + " "
        start = 0
        str_len = len(txt)
        for i in range(str_len):
            if txt[start] not in punctuation:
                if txt[i] in punctuation:
                    end = i
                    var_name = txt[start:end]
                    if var_name == old_var_name:
                        return txt[0:start] + new_var_name + txt[end:]
                    start = end
            else:
                if txt[i] not in punctuation:
                    end = i
                    start = end
        return txt

    def extract_sentences(self):
        sentences = list()
        i = 0
        success = 0
        failed = 0
        start = time.time()
        for file in self.files:
            i += 1
            if i != 1 and (i - 1) % 100 == 0:
                current = time.time()
                print(f'>>>>>>>> [duration: {current - start}] {success} successfully extracted, {failed} failed.')
            print(f'processing {i}-th file ({file})')
            try:
                solc_compiler = solidity.get_solc(file)
                sl = Slither(file, solc=solc_compiler)
                success += 1
            except Exception as e:
                print(f'compling failed for {file}')
                failed += 1
                continue
            for contract in sl.contracts:
                for function in contract.functions_and_modifiers:
                    for node in function.nodes:
                        if node.type == NodeType.OTHER_ENTRYPOINT:
                            sentence = self.normalize(node)
                            if node.variables_written:
                                v = node.variables_written[0]
                                sentence = sentence[0:16] + ' ' + str(v.type) + ' ' + sentence[16:]
                        elif node.type == NodeType.ENTRYPOINT:
                            sentence = f'ENTRY {node.function.signature}'
                        elif node.type == NodeType.VARIABLE:
                            sentence = self.normalize(node)
                            sentence = sentence.replace(str(NodeType.VARIABLE), f'{node.variable_declaration.type}')
                        else:
                            sentence = self.normalize(node)
                            sentence = sentence.replace('EXPRESSION', '')
                        sentence = split_words(sentence, RESERVED)
                        sentence = self.convert_number(sentence)
                        sentences.append(sentence)
                sentence = f'{contract.name}.START_CONTRACT'
                sentence = split_words(sentence, RESERVED)
                sentences.append(sentence)
        return sentences, success, failed

    @staticmethod
    def convert_number(sentence: list):
        new_sentence = []
        for s in sentence:
            if s.isdigit():
                new_sentence.append('constant')
            elif '0x' in s or s[2:].isdigit():
                new_sentence.append('constant')
            else:
                new_sentence.append(s)
        return new_sentence

    @staticmethod
    def extract_constant_strs(node: Node):
        constants = []
        for ir in node.irs:
            for r in ir.read:
                if isinstance(r, Constant) and isinstance(r.value, str) and len(r.value) > 1:
                    constants.append(r)
        return constants

    @staticmethod
    def extract_external_address(node: Node):
        addrs = []
        convert_addrs = []
        RESERVED = ['msg.sender', 'tx.origin']
        for ir in node.irs:
            if isinstance(ir, TypeConversion) and isinstance(ir.type, ElementaryType) and ir.type.name == 'address':
                pattern = "address\s*\(\s*(\w*)\s*\)"
                matchObj = re.search(pattern, str(node))
                if matchObj:
                    if matchObj.group() not in convert_addrs and matchObj.group(1) not in RESERVED:
                        convert_addrs.append(matchObj.group())
            if isinstance(ir, TypeConversion) and isinstance(ir.type, UserDefinedType) and isinstance(ir.type.type, Contract):
                pattern = f"{ir.type}\s*\(\s*(\w*)\s*\)"
                matchObj = re.search(pattern, str(node))
                if matchObj:
                    if matchObj.group() not in convert_addrs and matchObj.group(1) not in RESERVED:
                        convert_addrs.append(matchObj.group())
        for c in node.variables_read:
            if str(c) in RESERVED:
                continue
            if isinstance(c, Type):
                continue
            if hasattr(c, 'type') and isinstance(c.type, ElementaryType) and c.type.name == 'address':
                if str(c) not in addrs:
                    addrs.append(str(c))
                elif hasattr(c, 'type') and isinstance(c.type, UserDefinedType) and isinstance(c.type.type, Contract):
                    if c not in addrs:
                        addrs.append(str(c))
        return convert_addrs, addrs

    def normalize(self, node:Node):
        expression = ""
        if node.expression:
            expression += " " + str(node.expression)
        elif node.variable_declaration:
            expression += " " + str(node.variable_declaration)
        txt = str(node.type) + expression
        for c in self.extract_constant_strs(node):
            txt = txt.replace(c.value, 'string_type')
        convert_addrs, addrs = self.extract_external_address(node)
        for addr in convert_addrs:
            txt = txt.replace(addr, 'addr_var')
        for addr in addrs:
            txt = self.replace_var_name(txt, addr, 'addr_var')
        return txt

    def train_word_vector(self, sentence_path):
        if not os.path.exists(sentence_path):
            print('Target file does not exist')
            return
        add_unknown_word(sentence_path)
        with open(sentence_path, 'r') as f:
            sentences = json.load(f)
        if sentences:
            model = gensim.models.Word2Vec(sentences=sentences, vector_size=160, window=20, sg=1)
        model.save('word2vec.model')
        return model

    @staticmethod
    def similarity(vec1, vec2):
        return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))


def add_unknown_word(path):
    with open(path, 'r') as f:
        sentences = json.load(f)
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    with open(path, 'w') as f:
        json.dump(sentences, f)


def test_model(path):
    model = gensim.models.Word2Vec.load(path)
    # print(model.wv['_msgsender'].shape)
    X_tsne = TSNE(learning_rate=200, perplexity=30).fit_transform(model.wv.vectors)
    fig = px.scatter(X_tsne, x=0, y=1, hover_name=model.wv.index_to_key)
    fig.add_annotation(x=X_tsne[0, 0], y=X_tsne[0, 1], text=model.wv.index_to_key[0])
    fig.show()


def test_tfidf(sentences):
    with open('sentences.json', 'r') as f:
        sentences = json.load(f)
    gensim_dictionary = gensim.corpora.Dictionary()
    gensim_corpus = [gensim_dictionary.doc2bow(sent, allow_update=True) for sent in sentences]
    tfidf = gensim.models.TfidfModel(gensim_corpus, smartirs='ntc')
    for sent in tfidf[gensim_corpus]:
        print([[gensim_dictionary[id], np.around(frequency, decimals=2)] for id, frequency in sent])


def extract_sentences(path, target_sentences):
    word2vec = Word2vec(path)
    start = time.time()
    sentences, success, failed = word2vec.extract_sentences()
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    sentences.append(['_unknown_word_'])
    end = time.time()
    with open(target_sentences, 'w') as f:
        json.dump(sentences, f)
    print(f'[Done] extract {len(sentences)} sentences in {end-start} seconds')

def train_word_vec(path):
    word2vec = Word2vec(path)
    start = time.time()
    model = word2vec.train_word_vector('sentences.json')
    end = time.time()
    print(f'duration: {end - start}')
    return model

def show(model):
    X_tsne = TSNE(learning_rate=200, perplexity=30).fit_transform(model.wv.vectors)
    fig = px.scatter(X_tsne, x=0, y=1, hover_name=model.wv.index_to_key)
    nonreentrant = model.wv.key_to_index['nonreentrant']
    callerisuser = model.wv.key_to_index['callerisuser']
    onlyowner = model.wv.key_to_index['onlyowner']
    locktheswap = model.wv.key_to_index['locktheswap']
    fig.add_annotation(x=X_tsne[nonreentrant, 0], y=X_tsne[nonreentrant, 1], text=model.wv.index_to_key[nonreentrant])
    fig.add_annotation(x=X_tsne[callerisuser, 0], y=X_tsne[callerisuser, 1], text=model.wv.index_to_key[callerisuser])
    fig.add_annotation(x=X_tsne[onlyowner, 0], y=X_tsne[onlyowner, 1], text=model.wv.index_to_key[onlyowner])
    fig.add_annotation(x=X_tsne[locktheswap, 0], y=X_tsne[locktheswap, 1], text=model.wv.index_to_key[locktheswap])
    fig.show()


if __name__ == '__main__':
    # sentences = []
    extract_sentences('etherscan/', 'sentences.json')
    # train_word_vec('sentences.json')
    # add_unknown_word('sentences.json')
    # extract_sentences('etherscan/')
    model = train_word_vec('sentences.json')
    model.save('word2vec.model')
    show(model)
