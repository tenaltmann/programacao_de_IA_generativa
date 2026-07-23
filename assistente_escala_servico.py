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
