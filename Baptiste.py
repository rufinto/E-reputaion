import numpy as np
import pandas as pd

vecteurs_ref = {('migration', 'dehors'): 24, ('migration', 'aime', 'pas'): 18}
vecteur = ['migration', 'pas','encore','dehors','aime', 'vie', 'dedans']


def similitude(vecteur, vecteurs_ref):
    score = 0
    mots_compte = 0
    for clefs in vecteurs_ref:
        c = sum(1 for mot in clefs if mot in vecteur)
        
        if c > 0:
            mots_compte += c / len(clefs)
            
        if (c / len(clefs)) >= 0.5:
            valeur_totale = vecteurs_ref[clefs]
            score += (c / len(clefs)) * valeur_totale
            
    if mots_compte == 0:
        return 0 
    print(mots_compte)
    score_total = score / mots_compte
    return score_total

# Test the function
similitude(vecteur, vecteurs_ref)
print(similitude(vecteur, vecteurs_ref))
#prendre les mots qui ont une certaine similitude
print(len([ 0, -0.4, -0.3, -0.1, -1, -0.5, -0.6, 0.8, -0.6, -0.5, -0.4, -0.5, -0.6, -0.9, -0.2, -0.6, -0.8, 0, 0, 0, -1, -0.5, -0.8,-0.6, -0.5, -0.8, -0.4, -0.6, -0.7, -0.9, 0, 0, -0.8, -0.4, 0.8, 0, 0.3, 0.2,-0.3, -0.1, -0.5, 0.7, -1, -0.6, 0, 0.4, 0.8, -0.6, -0.9, 0, -0.4, -0.5, -0.5, 0, 0, -0.5, 0, 0, 0.5, -0.4, 0, -0.5, -0.8, 0, -1, -0.8, -0.3, -0.8, 0.5, 0.3, -0.9, 0, -0.5, -0.7, -0.9, 0, -0.5, 0, 0,-0.4, 0.9, -0.9, 0, -0.3, 0.7, 0.5, -0.4, 0, -0.4, -0.3, 0.6, -0.8, -0.5,-0.7, 0, 0.5, -0.6, -0.4, -0.4, -0.8,0]))