import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk 

# Importing the dataset
def afficher_dataset():
    chemin = 'FR-L-MIGR-TWIT-2011-2022.csv'
    dataset = pd.read_csv(chemin, sep=';')
    print(dataset.head())
afficher_dataset()