import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Passo 2 - Carregamento dos dados

#carregando a planilha em uma variavel (local ou )

link_da_planilha = "https://docs.google.com/spreadsheets/d/1vS5XdXcRXezck5MPkNBkxiQmXDI6b7b_JfVWJ0V63to/edit?usp=sharing"

#Essas duas linhas transformam o link do Google
# em um link de download da tabela em foprmato de tabela csv

id_da_planilha = link_da_planilha.split("/d/")[1].split("/")[0]
url_para_baixar = "https://docs.google.com/spreadsheets/d/" + id_da_planilha + "/export?format=csv"

df = pd.read_csv(url_para_baixar)
print("Total de viaturas na tabela:", len(df))
df.head(10)



## Passo 3 - Escolher a coluna que o modelo vai olhar

colunas_sem_resposta = ["km_rodado", "idade_anos", "qtd_manutencoes_anteriores"]

x = df[colunas_sem_resposta]    

print("O computador só vai ver:", list(x.columns))
print("Ele não sabe se a viatura precisou de manutenção")


# Passo 4 Padronizar os numeros

scaler = StandardScaler()
x_padronizado = scaler.fit_transform(x)

print("Numeros padronizados! Agora todas as colunas tem a mesma importancia")



# Passo 5 Pedir para o computador dividir em dois grupos 


kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
grupos = kmeans.fit_predict(x_padronizado)
                            
df["grupos_encontrados"] = grupos

print("Pronto, o computador separou as 60 vtrs em 2 grupos")
print(df["grupos_encontrados"].value_counts().to_string())