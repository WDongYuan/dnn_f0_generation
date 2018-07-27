# -*- coding: utf-8 -*-
import os
from stanfordcorenlp import StanfordCoreNLP
from sets import Set
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/Users/weidong/Downloads/stanford-corenlp-full-2017-06-09')

sentence = 'A student studies in Hunt Library.'
print 'Tokenize:', nlp.word_tokenize(sentence)
print 'Part of Speech:', nlp.pos_tag(sentence)
print 'Named Entities:', nlp.ner(sentence)
print 'Constituency Parsing:', nlp.parse(sentence)
print 'Dependency Parsing:', nlp.dependency_parse(sentence)








