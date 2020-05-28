#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sct

from math import ceil
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_20newsgroups

from sklearn.preprocessing import(OneHotEncoder, Binarizer, KBinsDiscretizer, StandardScaler)
from sklearn.feature_extraction.text import(CountVectorizer, TfidfTransformer, TfidfVectorizer)


# In[2]:


# Algumas configurações para o matplotlib.

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[5]:


countries = pd.read_csv("countries.csv")


# In[6]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# In[7]:


# Removendo espacos da direita e esquerda de cada string nas respectivas colunas
countries['Region'] = countries['Region'].str.strip()
countries['Country'] = countries['Country'].str.strip()


# In[8]:


# retirandos features não numericas
features_numeric = list(countries.columns.drop(['Country','Region','Population','Area']))

# substituindo ',' por '.' e assim tranformando as features de string para numerica
countries[features_numeric] = countries[features_numeric].applymap(lambda x : float(str(x).replace(',','.')))


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[10]:


def q1():
    return list(np.sort(countries['Region'].unique()))

q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[12]:


def q2():
    discretizar = KBinsDiscretizer(n_bins=10, encode="ordinal",strategy="quantile")
    intervalo = discretizar.fit_transform(countries[["Pop_density"]])
    
    resposta = len(intervalo) - (0.9 * len(intervalo))
    return ceil(resposta)

q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[13]:


def q3():
    one_hot_encoder = OneHotEncoder(sparse=False)
    
    countries_drop = countries[['Region', 'Climate']].dropna(subset = ['Region', 'Climate'])
    
    region_climate_encoder = one_hot_encoder.fit_transform(countries_drop)
    
    return int(region_climate_encoder.shape[1] + 1)

q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[10]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[15]:


def q4():
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("padronizacao", StandardScaler())
    ])
    
    numeric_features = countries.select_dtypes(include=['float64', 'int64'])
    pipeline.fit(numeric_features)
    test = pipeline.transform([test_country[2:]])
    arable = test[:, numeric_features.columns.get_loc("Arable")]

    return round(arable.item(), 3)


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[17]:


def q5():
    net_migration = countries.Net_migration.copy()
    q1 = net_migration.quantile(0.25)
    q3 = net_migration.quantile(0.75)
    iqr = q3 - q1

    n_outliers = [q1 - 1.5 * iqr, q3 + 1.5 * iqr]
   
    outliers_abaixo = net_migration[(net_migration < n_outliers[0])]
    outliers_acima  = net_migration[(net_migration > n_outliers[1])]
    
    resultado = (len(outliers_abaixo),len(outliers_acima),False)
    return tuple(resultado)

q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[19]:


# import dataset
categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

def q6():
    count_vectorizer = CountVectorizer()
    
    # Realizando o fit e transform com os dados do corpus
    newsgroups_counts = count_vectorizer.fit_transform(newsgroups_train.data)
    
    # índice da palavra'phone'
    phone = count_vectorizer.vocabulary_[u"phone"]     
    
    return int(newsgroups_counts[:,phone].sum())

q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[21]:


def q7():
    tfidf_vectorizer = TfidfVectorizer()

    tfidf_vectorizer.fit(newsgroups_train.data)

    newsgroups_tfidf_vectorized = tfidf_vectorizer.transform(newsgroups_train.data)

    phone = tfidf_vectorizer.vocabulary_.get(u"phone")

    res = round(newsgroups_tfidf_vectorized[:,phone].sum(),3)

    return float(res)

q7()

