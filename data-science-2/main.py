#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import scipy.stats as sct
import seaborn as sns


# In[3]:


sns.set()
plt.style.use('ggplot')
from IPython.core.pylabtools import figsize
figsize(12, 8)
palette = sns.color_palette(["#4286f4","#f44141"])


# In[6]:


athletes = pd.read_csv("athletes.csv")


# In[7]:


def get_sample(df, col_name, n=100, seed=42):
    np.random.seed(seed)    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False)
    return df.loc[random_idx, col_name]


# ## Inicia sua análise a partir daqui

# In[8]:


# colunas
athletes.columns


# In[9]:


# dataframe
athletes.head()


# In[10]:


# dimensões
athletes.shape


# In[11]:


# informações
athletes.info()


# In[12]:


# tipos de dados
athletes.dtypes


# In[13]:


# tabela percentual de dados faltantes
total = athletes.isnull().sum().sort_values(ascending=False)
percentual = (athletes.isnull().sum()/athletes.isnull().count()).sort_values(ascending=False)


missing_data = pd.concat([total, percentual], axis=1,join='outer', keys=['Dados Ausentes', '% Percentual'])
missing_data.index.name =' Variaveis numericas'
missing_data.head(20)


# In[14]:


# algumas caracteristicas
athletes.describe()


# In[48]:


#distribuição da altura
sns.distplot(athletes['height'])


# In[47]:


sns.boxplot(athletes['height'])


# In[49]:


# distribuição da peso
sns.distplot(athletes['weight'])


# In[50]:


sns.boxplot(athletes['weight'])


# In[15]:


print(f'Media peso: {athletes.weight.mean()} e altura: {athletes.height.mean()}')


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[17]:


def q1():
    sh = sct.shapiro(get_sample(athletes,'height',3000))
    return bool(sh[1] > 0.05)  # p_valor > alpha 

q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[86]:


amostra = get_sample(athletes,'height',3000)
amostra.plot(kind = 'hist', bins = 25)


# In[79]:


sm.qqplot(amostra, fit=True, line="45");


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[20]:


def q2():
    jb = sct.jarque_bera(get_sample(athletes,'height',3000))
    return bool(jb[1] > 0.05)  # p_valor > alpha 

q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[1]:


def q3():
    per = sct.normaltest(get_sample(athletes,'weight',3000))
    return bool(per[1] > 0.05)  # p_valor > alpha 

q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[28]:


amostra = get_sample(athletes,'weight',3000)
amostra.plot(kind = 'hist', bins = 25)


# In[29]:


sm.qqplot(amostra, fit=True, line="45");


# In[30]:


sns.boxplot(amostra)


# In[31]:


sns.distplot(amostra)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[35]:


def q4():
    res = sct.normaltest(np.log(get_sample(athletes,'weight',3000)))
    return bool(res[1] > 0.05)  # p_valor > alpha 

q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[41]:


amostra = get_sample(athletes,'weight',3000) # amostra aleatoria
weight_log = np.log(amostra) # transformação logaritmica 
sct.normaltest(weight_log) # teste de D'Agostino-Pearson


# In[42]:


weight_log.plot(kind = 'hist', bins = 25)


# In[43]:


sns.boxplot(weight_log)


# In[44]:


sns.distplot(weight_log)


# In[45]:


sm.qqplot(weight_log, fit=True, line="45");


# > __Para as questão 5, 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[69]:


def q5():
    bra = athletes.loc[athletes.nationality == 'BRA','height'].dropna()
    usa = athletes.loc[athletes.nationality == 'USA','height'].dropna()
    res = sct.ttest_ind(bra, usa, equal_var = False)
    return bool(res[1] > 0.05)

q5()


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[71]:


def q6():
    bra = athletes.loc[athletes.nationality == 'BRA','height'].dropna()
    can = athletes.loc[athletes.nationality == 'CAN','height'].dropna()
    res = sct.ttest_ind(bra, can, equal_var = False)
    return bool(res[1] > 0.05)

q6()


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[82]:


def q7():
    usa = athletes.loc[athletes.nationality == 'USA','height'].dropna()
    can = athletes.loc[athletes.nationality == 'CAN','height'].dropna()
    res = sct.ttest_ind(usa, can, equal_var = False)
    return float(round(res[1],8))

q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[84]:


bra = athletes.loc[athletes.nationality == 'BRA','height'].dropna()
usa = athletes.loc[athletes.nationality == 'USA','height'].dropna()
can = athletes.loc[athletes.nationality == 'CAN','height'].dropna()


# In[87]:


sct.ttest_ind(usa, can, equal_var = False)


# In[88]:


sct.ttest_ind(usa, bra, equal_var = False)


# In[89]:


sct.ttest_ind(bra, can, equal_var = False)

