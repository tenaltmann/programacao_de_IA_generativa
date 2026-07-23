### ASSISTENTE DE ESCALA DE SERVIÇO
#
# Receber mensagens em linguagem natural sobre a escala de serviço
# Identificar o sentimento (positivo, negativo, neutro)
# Reconhecer a intenção (consulta, troca, reclamação, confirmação)
# Extrair entidades (nomes, datas, postos)
# Gerar uma resposta adequada


## Passo 1 - Importando as bibliotecas

import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re



## Passo 2 - Importando as bibliotecas

# Baixar o lexico de sentimento de NLTK
nltk.download('vader_lexicon', quiet=True)
# Baixar o modelo em portugues do Spacy
nlp = spacy.load('pt_core_news_sm')
# Criar o analisador de sentimento
sia = SentimentIntensityAnalyzer()

print("Tudo foi carregado com sucesso")

