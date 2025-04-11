import numpy as np
import pandas as pd
import chardet

with open("ankieta.csv", 'rb') as f:
    result = chardet.detect(f.read())
    print(result)

data = pd.read_csv("ankieta.csv", encoding='ISO-8859-1', sep=";")

czy_sa_braki = data.isna().any().any()
print(czy_sa_braki)

data.columns = ['DZIAŁ', 'STAŻ', 'CZY_KIER', 'PYT_1', 'PYT_2', 'PYT_3', 'PŁEĆ', 'WIEK']

df = pd.DataFrame(data)

# Wyświetl liczbę wystąpień każdej wartości w kolumnach
for kolumna in df.columns:
    print(f"Liczba wystąpień wartości w kolumnie '{kolumna}':")
    print(df[kolumna].value_counts())
    print() 

df['WIEK_KAT'] = pd.cut(df['WIEK'], 
                         bins=[0, 35, 45, 55, float('inf')], 
                         labels=['do 35 lat', '36-45 lat', '46-55 lat', 'powyżej 55 lat'],
                         right=True)  # Domknięcie przedziałów z prawej strony

for kolumna in ['WIEK_KAT']:
    print(f"Liczba wystąpień wartości w kolumnie '{kolumna}':")
    print(df[kolumna].value_counts())
    print()

for kolumna in df.columns:
    print(f"Liczba wystąpień wartości w kolumnie '{kolumna}':")
    print(df[kolumna].value_counts())
    print()  