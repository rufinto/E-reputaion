import pandas as pd
import nltk
from nltk.tag import StanfordPOSTagger
import stanza 
import test
import re

#nlp = stanza.Pipeline('fr', processors='tokenize, pos, lemma, depparse')

jar = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
model = 'C:/Users/paull/Documents/EI DataWeb/stanford-postagger-full-2020-11-17/models/french-ud.tagger'
import os
java_path = "C:/Program Files (x86)/Java/jre-1.8/bin/java.exe"
os.environ['JAVAHOME'] = java_path
# Importing the dataset
chemin = 'FR-L-MIGR-TWIT-2011-2022.csv'
dataset = pd.read_csv(chemin, sep=';')

pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8')
res = pos_tagger.tag(test.filtered_tweets[90])
#print (res)


#grammar = "NP: {<DT>?<ADJ>*<VERB>?<NOUN>}"

grammar = r"""
  NP: {<DET>?<ADJ>*<NOUN>+}         # Groupe nominal: déterminant optionnel, adjectifs optionnels, un ou plusieurs noms
  VP: {<VERB>+<NP|PP|CLAUSE>+}      # Groupe verbal: un ou plusieurs verbes suivis de groupes nominaux, prépositionnels ou de clauses
  PP: {<ADP><NP>}                    # Groupe prépositionnel: préposition suivie d'un groupe nominal
  ADJP: {<ADJ>+<CC>?<ADJ>*}          # Groupe adjectival: un ou plusieurs adjectifs, optionnellement coordonnés par une conjonction de coordination
  CLAUSE: {<NP><VP>}                 # Clause: groupe nominal suivi d'un groupe verbal
"""
chunk_parser = nltk.RegexpParser(grammar)


tree = chunk_parser.parse(res)
#print(type(tree))



def chunking(tweet):
    res = pos_tagger.tag(tweet)
    tree = chunk_parser.parse(res)
    list = extract_chunks(tree)
    print(list)

#Transforme l'arbre en liste de branches
def extract_chunks(tree):
    chunks = []
    for i in range(len(tree)):
        chunks_inter = []
        for j in range (len(tree[i])):
            chunks_inter.append(tree[i][j])
        chunks.append(chunks_inter)
    return chunks
            
