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