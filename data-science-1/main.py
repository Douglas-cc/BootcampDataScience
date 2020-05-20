#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


df


# In[5]:


# dimensões do dataframe
df.shape


# In[6]:


# tipos de dados
df.dtypes.unique()


# In[7]:


# colunas
df.columns


# ## Distribuição Normal

# In[8]:


# verificar o grafico da distribuição normal
sns.distplot(df.normal);


# In[9]:


# distribuição normal
media_norm = df.normal.mean()
mediana_norm = df.normal.median()
std_norm = df.normal.std()
var_norm = df.normal.std()

print(f'Mediana: {mediana_norm}')
print(f'Media: {media_norm}')
print(f'Desvio padrão: {std_norm}')
print(f'Variancia: {var_norm}')


# Podemos encontrar $P(X \leq 20)$ com a função `cdf()`:

# In[10]:


sct.norm.cdf(20, loc=media_norm , scale=std_norm)


# Obviamente, como a distribuição é normal é simétrica em torno da média $\mu$, a probabilidade de $X$ assumir um valor menor ou igual à média deve ser 0.5 (50%), ou seja, $P(X \leq \mu) = 0.5$:

# In[11]:


sct.norm.cdf(media_norm, loc=media_norm , scale=std_norm)


# Podemos utilizar a função `cdf()` em conjunto com a função `linspace()` do NumPy para gerar um gráfico da CDF:

# In[12]:


f = lambda x: sct.norm.cdf(df.normal, loc=media_norm , scale=std_norm)

cdf = f(df.normal)

sns.lineplot(df.normal, cdf);


# Às vezes, estamos interessados no complemento da CDF, ou seja, na probabilidade a partir da outra ponta da distribuição. Para isso, usamos a função `sf()`. Por exemplo, utilizamos essa função para achar $P(X \geq 18)$:

# In[13]:


sct.norm.sf(18, loc=media_norm , scale=std_norm)


# Note como esse valor é o complemento da probabilidade encontrado acima com a CDF.

# Por fim, podemos querer saber o valor da função densidade de probabilidade, $f(x)$ , em determinado ponto. Apesar de não ser tão útil normalmente (por __não__ ser representar uma probabilidade), ela pode ter seus usos. Para isso, utilizamos a função `pdf()`. Para acharmos $f(18)$:

# In[14]:


sct.norm.pdf(18, loc=media_norm , scale=std_norm)


# Na distribuição normal, a função $f(x)$ assume seu valor máximo na média:

# In[15]:


sct.norm.pdf(media_norm, loc=media_norm , scale=std_norm)


# Como fizemos com a CDF, podemos utilizar a função `linspace()` para gerar o gráfico da pdf:

# In[16]:


f = lambda x: sct.norm.pdf(df.normal, loc=media_norm , scale=std_norm)

pdf = f(df.normal)

sns.lineplot(df.normal, pdf);


# Também pode ser útil encontrar o quantil para determinada probabilidade (acumulada a partir da cauda à esquerda). Por exemplo, podemos nos perguntar qual o valor de $X$ que acumula 25% da probabilidade, ou seja, qual valor de $x$ tal que $P(X \leq x) = 0.25$? Respondemos esse tipo de pergunta com a função `ppf()`:

# In[17]:


q1_norm = sct.norm.ppf(0.25, loc=media_norm , scale=std_norm) # primeiro quartil
print(q1_norm)


# Se quiséssemos o saber o quantil para a probabilidade acumulada a partir da cauda à direita, usaríamos a função `isf()`. Por exemplo, se quisermos encontrar $x$ tal que $P(X \geq x) = 0.25$:

# In[18]:


q2_norm = sct.norm.isf(0.50, loc=media_norm , scale=std_norm) # segundo quartil
print(q2_norm)


# In[19]:


q3_norm = sct.norm.isf(0.25, loc=media_norm , scale=std_norm) # terceiro qaurtil
print(q3_norm)


# In[20]:


#intervalo interquartil da normal
irq_norm = q3_norm - q1_norm
print(irq_norm)


# In[21]:


# Resumindo
df.normal.describe()


# In[22]:


plt.boxplot(df.normal, vert = False)


# ## Distribuição Binomial

# In[23]:


sns.distplot(df.binomial, kde = False);


# In[24]:


df.columns


# In[25]:


binomial = df.binomial
media_binom = binomial.mean()
std_binom = binomial.std()


# In[26]:


binomial.describe()


# In[27]:


plt.boxplot(df.binomial, vert = False)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` do `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[28]:


def q1():
    norm = np.quantile(df['normal'], [0.25, 0.50, 0.75]) 
    binom = np.quantile(df['binomial'], [0.25, 0.50, 0.75]) 
    return tuple(np.round(norm - binom, 3))


# In[29]:


q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda com um único escalar arredondado para três casas decimais.

# In[30]:


def q2():
    cauda_inf = df.normal.mean() - df.normal.std()
    cauda_sup = df.normal.mean() + df.normal.std()
    
    prob_inf = sct.norm.cdf(cauda_inf,loc = 20, scale = 4)
    prob_sup = sct.norm.cdf(cauda_sup,loc = 20, scale = 4)
    
    return round((prob_sup - prob_inf),3)


# In[31]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[32]:


def q3():
    dif_m = round((df.binomial.mean() - df.normal.mean()),3) # diferencia media
    dif_v = round((df.binomial.var() - df.normal.var()),3) # diferencia variancia
    return (dif_m, dif_v)


# In[33]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[34]:


stars = pd.read_csv("pulsar_stars.csv")

features = ["mean_profile", "sd_profile", "kurt_profile", "skew_profile",
            "mean_curve", "sd_curve", "kurt_curve","skew_curve","target"]

stars.rename({old_name: new_name for (old_name, new_name) in zip(stars.columns,features)}, axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[35]:


# colunas
stars.columns


# In[36]:


stars.info()


# In[37]:


stars.shape


# Bom como podemos ver nos dados padronizado, a media é igual a -2,5. Ai voce deve estar se perguntando, não deveria ser zero, assim como a variação é igual 1? BOm este valor de -2,5 padronizado equivale a zero. Melhor forma de entender é através do grafico que plotamos anteriormente 

# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[38]:


def q4():
    filtrado = stars[stars.target == False].mean_profile
    padronizado = (filtrado - filtrado.mean())/filtrado.std()
    ecdf = ECDF(padronizado)
    quantis = sct.norm.ppf([0.80,0.90,0.95], loc = 0, scale = 1)
    return tuple(np.around(ecdf(quantis),3))


# In[39]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[40]:


def q5():
    filtrado = stars[stars.target == False].mean_profile
    padronizado = (filtrado - filtrado.mean())/filtrado.std()
    quartis_do_df = np.quantile(padronizado, [0.25, 0.50, 0.75])
    quartis_teoricos = sct.norm.ppf([0.25, 0.50, 0.75], loc=0, scale= 1)
    return tuple(np.around(quartis_do_df - quartis_teoricos,3))


# In[41]:


q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
