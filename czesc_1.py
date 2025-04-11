import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import pandas as pd
import plot_likert 
from scipy.stats import beta, binomtest
import math
import scipy
from statsmodels.stats.proportion import proportions_ztest

# 1 i 2

dane = np.genfromtxt('ankieta.csv', delimiter=";", dtype=str, skip_header=1)

dane_wiek = dane[:, 7].astype(float)  # Zakładając, że WIEK jest w kolumnie 8 (indeks 7)
WIEK_KAT = []

for x in dane_wiek:
    if x <= 35: 
        wiek_kat = '<= 35'
    elif 36 <= x <= 45: 
        wiek_kat = 'między 36 a 45'
    elif 46 <= x <= 55: 
        wiek_kat = 'między 46 a 55'
    else: 
        wiek_kat = '>= 55'
    WIEK_KAT.append(wiek_kat)


dane_p_1 = dane[:, 3]  
dane_p_2 = dane[:, 4]  
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(dane_p_1, color='blue', edgecolor='black', bins=len(np.unique(dane_p_1)))
axes[0, 0].set_title('Ilość poszczególnych odpowiedzi na pytanie 1')

axes[0, 1].hist(dane_p_2, color='green', edgecolor='black', bins=len(np.unique(dane_p_2)))
axes[0, 1].set_title('Ilość poszczególnych odpowiedzi na pytanie 2')

labels_1, counts_1 = np.unique(dane_p_1, return_counts=True)
labels_2, counts_2 = np.unique(dane_p_2, return_counts=True)

axes[1, 0].pie(counts_1, labels=labels_1, autopct='%1.1f%%', startangle=90, 
               colors=['red', 'green', 'blue', 'orange'])
axes[1, 0].set_title('Procentowy udział poszczególnych odpowiedzi w pytaniu 1')

axes[1, 1].pie(counts_2, labels=labels_2, autopct='%1.1f%%', startangle=90, 
               colors=['red', 'green', 'blue', 'orange'])
axes[1, 1].set_title('Procentowy udział poszczególnych odpowiedzi w pytaniu 2')

plt.tight_layout()
plt.show()

dane_dzial = dane[:, 0] 
dane_staz = dane[:, 1]  
dane_kier = dane[:, 2]
dane_plec = dane[:, 6]  

pyt_1_unikalne = np.unique(dane_p_1)
dzial_unikalne = np.unique(dane_dzial)

tablica_pyt1_dzial = np.zeros((len(pyt_1_unikalne), len(dzial_unikalne)))

for i in range(len(dane_p_1)):
    pyt_1_idx = np.where(pyt_1_unikalne == dane_p_1[i])[0][0]
    dzial_idx = np.where(dzial_unikalne == dane_dzial[i])[0][0]
    tablica_pyt1_dzial[pyt_1_idx, dzial_idx] += 1

print("Tablica dla PYT_1 i DZIAŁ:")
print(tablica_pyt1_dzial)

staz_unikalne = np.unique(dane_staz)
tablica_pyt1_staz= np.zeros((len(pyt_1_unikalne), len(staz_unikalne)))

for i in range(len(dane_p_1)):
    pyt_1_idx = np.where(pyt_1_unikalne == dane_p_1[i])[0][0]
    staz_idx = np.where(staz_unikalne == dane_staz[i])[0][0]
    tablica_pyt1_dzial[pyt_1_idx, staz_idx] += 1

print("Tablica dla PYT_1 i STAŻ:")
print(tablica_pyt1_staz)

kier_unikalne = np.unique(dane_kier)
tablica_pyt1_kier= np.zeros((len(pyt_1_unikalne), len(kier_unikalne)))

for i in range(len(dane_p_1)):
    pyt_1_idx = np.where(pyt_1_unikalne == dane_p_1[i])[0][0]
    kier_idx = np.where(kier_unikalne == dane_kier[i])[0][0]
    tablica_pyt1_kier[pyt_1_idx, kier_idx] += 1

print("Tablica dla PYT_1 i KIEROWNICTWO:")
print(tablica_pyt1_kier)

plec_unikalne = np.unique(dane_plec)
tablica_pyt1_plec= np.zeros((len(pyt_1_unikalne), len(plec_unikalne)))

for i in range(len(dane_p_1)):
    pyt_1_idx = np.where(pyt_1_unikalne == dane_p_1[i])[0][0]
    plec_idx = np.where(plec_unikalne == dane_plec[i])[0][0]
    tablica_pyt1_plec[pyt_1_idx, plec_idx] += 1

print("Tablica dla PYT_1 i PŁEĆ:")
print(tablica_pyt1_plec)

#tablica jeszcze wiek_kat

#7
CZY_ZADOW = []
czy_zadow = str
dane_p_2 = dane_p_2.astype(float)
for x in dane_p_2:
    if x < 0: 
        czy_zadow = 'nie'
    elif x > 0: 
        czy_zadow = 'tak'
    CZY_ZADOW.append(czy_zadow)

CZY_ZADOW_2 = []
czy_zadow_2 = str
dane_p_1 = dane_p_1.astype(float)
for x in dane_p_1:
    if x < 0: 
        czy_zadow_2 = 'nie'
    elif x > 0: 
        czy_zadow_2 = 'tak'
    CZY_ZADOW_2.append(czy_zadow_2)



#8


#CZĘŚĆ 2

#zad3

probka = np.random.choice(a=dane_wiek, size=int(0.1 * len(dane)), replace=True)
print(probka)

#zad4

def binomial_random_variable(n, p, size=1):
    return np.sum(np.random.rand(size, n) < p, axis=1)

n, p = 10, 0.5
size = 10000
samples = binomial_random_variable(n, p, size)
print(samples)


def multinomial_random_variable(n, p, size=1):
    return np.random.multinomial(n, p, size)

n = 100
p = [0.3, 0.4, 0.3]
size = 10000
samples = multinomial_random_variable(n, p, size)

theoretical_means = np.array([n * p_i for p_i in p])
theoretical_variances = np.array([n * p_i * (1 - p_i) for p_i in p])

empirical_means = np.mean(samples, axis=0)
empirical_variances = np.var(samples, axis=0)

print("Teoretyczne średnie:", theoretical_means)
print("Teoretyczne wariancje:", theoretical_variances)
print("Empiryczne średnie:", empirical_means)
print("Empiryczne wariancje:", empirical_variances)

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(len(p)):
    ax.hist(samples[:, i], bins=np.arange(n+2) - 0.5, density=True, alpha=0.5, label=f'Kategoria {i+1}')

x = np.arange(n+1)
for i, p_i in enumerate(p):
    pmf_theoretical = scipy.special.comb(n, x) * (p_i**x) * ((1 - p_i)**(n - x))
    ax.plot(x, pmf_theoretical, 'o', label=f'Kategoria {i+1} - Teoretyczny')

ax.set_title(f'Histogram próbek i rozkład teoretyczny\n(n={n}, p={p})')
ax.set_xlabel('Liczba prób w kategorii')
ax.set_ylabel('Prawdopodobieństwo')
ax.legend()
plt.show()

#CZĘŚĆ 3 i 4


#CZĘŚĆ 5
#zad 10
successes = 30  
trials = 100    
p0 = 0.5        
result = binomtest(successes, trials, p=p0, alternative='two-sided')

print("Test dokładny dwumianowy:")
print(f"Statystyka testowa (p-value): {result.pvalue:.4f}")
print(f"95% przedział ufności: {result.proportion_ci(confidence_level=0.95)}")

successes = 30  # Liczba sukcesów
trials = 100    # Liczba prób
p0 = 0.5        # Hipotetyczne prawdopodobieństwo sukcesu

stat, p_value = proportions_ztest(count=successes, nobs=trials, value=p0, alternative='two-sided')

print("\nTest asymptotyczny (Z-test proporcji):")
print(f"Statystyka Z: {stat:.4f}")
print(f"p-value: {p_value:.4f}")

#zad11
#1
successes = (dane_plec == 'K').sum()  
trials = len(dane_plec)  
p0 = 0.5  

print(f"Liczba kobiet: {successes}, Liczba pracowników: {trials}")

binom_test_result = binomtest(successes, trials, p=p0, alternative='two-sided')
print("\n🔹 Test dokładny dwumianowy:")
print(f"Statystyka testowa (p-value): {binom_test_result.pvalue:.4f}")
print(f"95% przedział ufności: {binom_test_result.proportion_ci(confidence_level=0.95)}")

if trials > 30: 
    stat, p_value = proportions_ztest(count=successes, nobs=trials, value=p0, alternative='two-sided')
    print("\n🔹 Test asymptotyczny (Z-test proporcji):")
    print(f"Statystyka Z: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")

alpha = 0.05
if binom_test_result.pvalue < alpha:
    print("\n Odrzucamy hipotezę H0: Prawdopodobieństwo, że pracownik to kobieta ≠ 0.5")
else:
    print("\n Brak podstaw do odrzucenia H0: Prawdopodobieństwo, że pracownik to kobieta")

#2
successes = (dane_p_2 >= 0).sum()  
trials = len(CZY_ZADOW)  
p0 = 0.7

print(f"Liczba kobiet: {successes}, Liczba pracowników: {trials}")

binom_test_result = binomtest(successes, trials, p=p0, alternative='two-sided')
print("\n🔹 Test dokładny dwumianowy:")
print(f"Statystyka testowa (p-value): {binom_test_result.pvalue:.4f}")
print(f"95% przedział ufności: {binom_test_result.proportion_ci(confidence_level=0.95)}")

if trials > 30:  
    stat, p_value = proportions_ztest(count=successes, nobs=trials, value=p0, alternative='two-sided')
    print("\n🔹 Test asymptotyczny (Z-test proporcji):")
    print(f"Statystyka Z: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")

alpha = 0.05
if binom_test_result.pvalue < alpha:
    print("\n Odrzucamy hipotezę H0: Prawdopodobieństwo, że pracownik to kobieta ≠ 0.5")
else:
    print("\n Brak podstaw do odrzucenia H0: Prawdopodobieństwo, że pracownik to kobieta")


#12
