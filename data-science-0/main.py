#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[18]:


import pandas as pd
import numpy as np


# In[19]:


df = pd.read_csv("black_friday.csv")


# In[20]:


mododuruca = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[21]:


df.head(5)


# In[22]:


# colunas do dataset
df.columns


# In[23]:


# informações sobre os dados
df.info()


# In[24]:


# verificados as colunas de categoria de produto
df[['Product_Category_1','Product_Category_2','Product_Category_3']].head(5)


# In[25]:


# dados faltantes em todo dataset
df.isna().sum()


# In[26]:


# verificando as dimensões do dataframe
df.shape


# In[27]:


# contando o genero das pessoas
df.Gender.value_counts()


# In[28]:


# contando faixe etaria 
df.Age.value_counts()


# In[29]:


df.head()


# In[30]:


# tipos de dados no dataframe
df.dtypes.unique()


# In[31]:


# criando um novo dataframe
aux = pd.DataFrame({'Colunas': df.columns, 'Tipos':df.dtypes, 'ausentes': df.isna().sum()})


# In[32]:


#percentual dados faltantes
aux['percentual'] = aux['ausentes'] / df.shape[0]


# In[33]:


# valor que mais se repete no na coluna Product_Category_3
df.Product_Category_3.value_counts().head(1)


# In[34]:


# somando todos os valores da coluna ausentes
aux['ausentes'].sum()


# In[35]:


# percentual da coluna ausentes
aux['ausentes'] / df.shape[0]


# In[36]:


# soma dos valores ausentes
df.Product_Category_3.isna().sum()  


# In[37]:


df.Purchase


# In[38]:


# novo dataframe
aux = pd.DataFrame({'Colunas': df.columns, 'Tipos':df.dtypes, 'ausentes': df.isna().sum()})


# In[39]:


print(aux)


# In[40]:


aux['ausentes'] / df.shape[0]


# In[41]:


result = df.shape


# In[42]:


print(result)


# In[43]:


df.head()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[44]:


def q1():
    return mododuruca.shape
    pass


# In[45]:


q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[46]:


def q2():
    faixa_etaria = df[df.Age == '26-35']
    mulheres = faixa_etaria[faixa_etaria.Gender == 'F']
    result = len(mulheres.Gender)
    return result
    pass


# In[47]:


q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[48]:


def q3():
    return df.User_ID.nunique() 
    pass


# In[49]:


q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[50]:


def q4():
    return df.dtypes.nunique()
    pass


# In[51]:


q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[52]:


def q5():
    return (df.shape[0] - df.dropna().shape[0])/df.shape[0]  
    pass


# In[53]:


q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[54]:


def q6():
    return df.Product_Category_3.isna().sum()  
    pass


# In[55]:


q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[56]:


def q7():
    return df['Product_Category_3'].value_counts().idxmax()
    pass


# In[57]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[58]:


def q8():
    maximo = df['Purchase'].max()
    minimo = df['Purchase'].min()
    normalizado = (df['Purchase'] - minimo)/(maximo - minimo)
    return float(normalizado.mean())
    pass


# In[59]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[60]:


def q9():
    df['Purchase_nor'] = (df.Purchase - df.Purchase.mean())/df.Purchase.std()
    return  len([i for i in df['Purchase_nor'] if i > -1 and i <1])
    pass


# In[61]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[62]:


def q10():
    soma = df['Product_Category_2'].isnull() != df['Product_Category_3'].isnull()
    result = soma.sum()> 0
    return result


# In[63]:


q10()


# In[ ]:




