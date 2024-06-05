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



grammar = r"""
  NP: {<DET>?<ADJ>*<NOUN>+}         # Groupe nominal: déterminant optionnel, adjectifs optionnels, un ou plusieurs noms
  VP: {<VERB>+<NP|PP|CLAUSE>+}      # Groupe verbal: un ou plusieurs verbes suivis de groupes nominaux, prépositionnels ou de clauses
  PP: {<ADP><NP>}                    # Groupe prépositionnel: préposition suivie d'un groupe nominal
  ADJP: {<ADJ>+<CC>?<ADJ>*}          # Groupe adjectival: un ou plusieurs adjectifs, optionnellement coordonnés par une conjonction de coordination
  CLAUSE: {<NP><VP>}                 # Clause: groupe nominal suivi d'un groupe verbal
"""
chunk_parser = nltk.RegexpParser(grammar)

# Test sur un tweet

#res = pos_tagger.tag(test.filtered_tweets[90])
#tree = chunk_parser.parse(res)



#Fonction qui permet de chunker un tweet
def chunking(tweet):
    res = pos_tagger.tag(tweet)
    tree = chunk_parser.parse(res)
    list = extract_chunks(tree)
    list_1= extract_words_from_chunks(list)
    
    print(list_1)
    tree.draw()

#Transforme l'arbre en liste de branches
def extract_chunks(tree):
    chunks = []
    for i in range(len(tree)):
        chunks_inter = []
        for j in range (len(tree[i])):
            chunks_inter.append(tree[i][j])
        chunks.append(chunks_inter)
    return chunks
            
def extract_words_from_chunks(list):
    words = []
    for i in range(len(list)):
        words_inter = []
        for j in range(len(list[i])):
            if type(list[i][j]) == tuple:
                if type(list[i][j][-1]) != nltk.tree.Tree:
                    words_inter.append(list[i][j][0])
                else:
                    for k in range(len(list[i][j][-1][1:])):
                        for l in range(len(list[i][j][-1][1:][k])):
                            if type(list[i][j][-1][k][l]) == tuple:
                                words_inter.append(list[i][j][-1][k][l][0])
                            else:
                                words_inter.append(list[i][j][-1][k][0])  

            elif type(list[i][j][0]) == tuple:
                words_inter.append(list[i][j][0])
            else:
                words_inter.append(list[i][0])
                break
        words.append(words_inter)
    return words





print (chunking(test.filtered_tweets[80]))