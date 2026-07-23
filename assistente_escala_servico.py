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



## Passo 3 - quebrando o texto em pedaços

# Exemplo de mensagem
texto = "Preciso trocar minha escala de quarta para sexta com o Cabo Abudes"

# o spacy processa o texto inteiro de uma vez
doc = nlp(texto)

# Vamos ver cada token e sua classe gramatical
print("TOKEN" .ljust(20), "CLASSE GRAMATICAL")
print("-" * 40)
for token in doc:
  print(f'{token.text.ljust(20)} {token.pos_}')


## Passo 4 - Filtra o STopwords

palavras_uteis = [token.text for token in doc if not token.is_stop and not token.is_punct]

print("Frase Original: ")
print(f" '{texto}'")
print()
print("Palavras úteis (sem stopwords): ")
print(f"{palavras_uteis}")
