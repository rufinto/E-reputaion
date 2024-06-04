import pandas as pd
import nltk
from nltk.tag import StanfordPOSTagger
import test

jar = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
model = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/models/french-ud.tagger'
import os
java_path = "C:/Program Files (x86)/Java/jre-1.8/bin/java.exe"
os.environ['JAVAHOME'] = java_path
# Importing the dataset
chemin = 'FR-L-MIGR-TWIT-2011-2022.csv'
dataset = pd.read_csv(chemin, sep=';')

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
res = pos_tagger.tag(test.filtered_tweets[100])
print (res)

grammar = "NP: {<DT>?<ADJ>*<VERB>?<NOUN>}"
chunk_parser = nltk.RegexpParser(grammar)
tree = chunk_parser.parse(res)
tree.draw()