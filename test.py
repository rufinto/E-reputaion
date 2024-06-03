import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import nltk 

# Importing the dataset
def afficher_dataset():
    chemin = "MIGRTWIT20112022.csv"
    dataset = pd.read_csv(chemin, sep='\t')
    print(dataset.head())
afficher_dataset()