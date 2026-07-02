# coding: utf-8
import sys
sys.path.append('..')
from 人工智能.Deep_Learning.Natural_Language_Processing.res.common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]  #you的单词向量
c1 = C[word_to_id['i']]  #iの单词向量
print(cos_similarity(c0, c1))
