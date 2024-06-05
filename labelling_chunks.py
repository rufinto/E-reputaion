import labelling
import test




X=[4735, 4601, 4766, 522, 4271, 1901, 5611, 4510, 1760, 5621, 1366, 423, 1426, 3569, 
918, 3010, 778, 5168, 1551, 3591, 4003, 5113, 2309, 153, 5120, 1172, 1181, 1248, 
965, 2974, 5676, 1527, 684, 3019, 3619, 5373, 4058, 5779, 1311, 2275, 2435, 1227, 
3878, 5070, 1211, 672, 1352, 5280, 52, 2985, 4707, 1804, 3640, 4345, 2889, 785, 
2378, 3412, 3284, 951, 3748, 2348, 4674, 772, 1228, 647, 3217, 5459, 1995, 5677, 
4973, 3659, 3005, 5308, 2170, 294, 2608, 957, 807, 2676, 786, 5565, 1634, 1335, 
4658, 5371, 5768, 4780, 4824, 2918, 1950, 2591, 1843, 4806, 567, 5291, 2331, 2531, 
4404, 859, 3866, 3285, 891, 4930, 3345, 5058, 5440, 2766, 1766, 4370, 5482, 4152, 
5402, 1158, 5643, 4187, 1789, 3967, 5246, 3943, 22, 4010, 1204, 270, 2525, 1229, 
2061, 4571, 119, 1694, 2824, 2338, 5097, 297, 1836, 42, 675, 4614, 671, 1841, 
344, 3041, 3511, 5567, 1408, 5065, 939, 228, 5006, 5783, 1055, 779, 1557, 1325, 
2238, 4441, 4900, 3067, 1788, 5272, 3553, 1398, 173, 5784, 4019, 474, 2047, 5087, 
3633, 2204, 589, 2607, 3961, 5327, 2566, 4458, 1356, 4709, 3811, 3188, 2004, 2981, 
2164, 1383, 4837, 4478, 1171, 2541, 1129, 3227, 2250, 3672, 435, 3382, 770, 4538, 
5697, 3331, 3769, 2932, 2998, 2236, 138, 4384, 4419, 1553, 1871, 2251, 2653, 5084, 
3956, 759, 4651, 4590, 1090, 2081, 2099, 1516, 1575, 5207, 841, 300, 2232, 657, 
2880, 1923, 3235, 2285, 2972, 1106, 5004, 3800, 780, 1212, 1791, 3484, 585, 3854, 
1823, 412, 4539, 564, 3720, 2068, 5451, 411, 1588, 5188, 3431, 221, 4678, 1724, 
1443, 5419, 2248, 4468, 5635, 4653, 2449, 416, 5224, 2104, 5539, 2388, 4481, 1350, 
3662, 1235, 171, 2980, 4665, 2866, 219, 4423, 2771, 3232, 662, 2603, 4648, 2774, 
2487, 3480, 2098, 1615, 4445, 265, 1175, 4420, 2654, 2328, 4080, 5078, 3805, 1911, 
2175, 4600, 2928, 1286, 15, 674]

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

def chunks(i):
    return labelling.chunking(test.filtered_tweets[X[i]])

Chunks = {}

i = int(input("Entrez l'indice du tweet à étiqueter: "))
chunks = labelling.chunking(test.filtered_tweets[X[i]])  # Assurez-vous que la fonction chunking retourne une liste de chunks
if chunks is not None:
    for j, chunk in enumerate(chunks):
        while True:
            print(chunk)
            note = input(f"Étiquetez le chunk {j+1} du tweet {i+1} (ou tapez 's' pour passer au suivant): ")
            if note.lower() == 's':
                break  # Sortir de la boucle while et passer au chunk suivant
            else:
                Chunks[(i, j)] = {'chunk': chunk, 'note': note}
                break  # Sortir de la boucle while et passer au chunk suivant
print (Chunks)