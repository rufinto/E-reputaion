import pandas as pd

dataset_name = "FR-L-MIGR-TWIT-2011-2022.csv"
dataset = pd.read_csv(dataset_name)
colonne = ["data__id", "data__text", "data__created_at", "author__username"]

print(dataset[colonne[0]])