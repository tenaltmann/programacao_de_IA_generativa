## Passo 1 - Preparar Ferramentas\


# pandas - organizar os dados em forma de tabela com uma planilha
import pandas as pd

# essas ferramentas vem do Scikit-learn, a caixa de ferramenta do Machine Learning
from sklearn.model_selection import train_test_split    #treina e divide os dados em conjunto de teste
from sklearn.tree import DecisionTreeClassifier         #cria modelo de arvore de decisao
from sklearn.metrics import accuracy_score              #calcula a precisão


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

colunas_entradas = ["km_rodado", "idade_anos", "qtd_manutencoes_anteriores"]

coluna_alvo = "precisa_manutencao"

x = df[colunas_entradas]    #as pistas para o modelo
y = df[coluna_alvo]         #a resposta certa de cada viatura

print("O modelo vai olhar para :", colunas_entradas)

# Passo 4 - Separar viaturas para treino e para teste::

# PARAMETRO 1 - qual fatia das viaturas ficam separadas para o teste
# 0.2 significa 20% das viaturas ficam fora do treino, so para teste depois

fatia_para_teste = 0.2

x_treino, x_teste, y_treino, y_teste = train_test_split(
    x, y, test_size=fatia_para_teste, random_state=42
)

print("Viatura para Treino:", len(x_treino))
print("Viatura guardadas para teste :", len(x_teste))