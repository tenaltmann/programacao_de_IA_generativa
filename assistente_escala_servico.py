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


## Passo 5 -  Analise de sentimento


frases = [
    'A escala deste mês ficou muito bem organizada, parabéns',
    'Preciso consultar minha escala de serviço da próxima semana',
    'É um absurdo eu estar de serviço no terceiro domingo seguido',
    'Confirmo que estarei presente no serviço de quarta-feira',
    'Estou insatisfeito com a distribuição injusta das escalas',
    'Solicito permuta de serviço com o Soldado Santos para o dia 15',
]

print("Analise de sentimento")
print("=" * 70)
for frase in frases:
  scores = sia.polarity_scores(frase)
  #determinar o sentimento predominante
  if scores['compound'] >= 0.05:
    sentimento = "POSITIVO"
  elif scores['compound'] <= -0.05:
    sentimento = "NEGATIVO"
  else:
    sentimento = "NEUTRO"


  print(f"\n'{frase}'")
  print(f"{sentimento} (score: {scores['compound']:.2f})")




## Passo 6 - Estração das entidades

#Mensagem com várias entidades
mensagem = "O Sargento oliveira precisa trocar o serviço do dia 20 de agosto com o Cabo Ferreira do 4° BIL"


doc = nlp(mensagem)

print(f"mensagem: {mensagem}")
print()
print("ENTIDADES ENCONTRADAS: ")
print("-" * 40)

if doc.ents:
  for ent in doc.ents:
    print(f" {ent.text.ljust(25)} - {ent.label} ({spacy.explain(ent.label_)})")
else:
  print("Nennhuma entidade detectada pelo modelo.")



# Passo 7 - Extrair Entidades

def extrair_entidades_militar(texto):
    """Extrai entidades relevantes para o contexto militar usando regras."""
    entidades = {
        'postos': [],
        'nomes': [],
        'datas': [],
        'unidades': []
    }

    # Postos e graduações
    postos = ['Soldado', 'Cabo', 'Sargento', 'Terceiro-Sargento', 'Segundo-Sargento',
              'Primeiro-Sargento', 'Subtenente', 'Tenente', 'Capitão', 'Major',
              'Coronel', 'General']
    for posto in postos:
        # Busca o posto seguido de um nome próprio (palavra começando com maiúscula)
        padrao = rf'{posto}\s+([A-Z][a-záéíóú]+)'
        matches = re.findall(padrao, texto)
        for nome in matches:
            entidades['postos'].append(posto)
            entidades['nomes'].append(f'{posto} {nome}')

    # Datas
    dias_semana = re.findall(r'(segunda|terça|quarta|quinta|sexta|sábado|domingo)(?:-feira)?', texto, re.IGNORECASE)
    entidades['datas'].extend(dias_semana)

    datas_num = re.findall(r'dia\s+(\d{1,2}(?:\s+de\s+\w+)?)', texto, re.IGNORECASE)
    entidades['datas'].extend(datas_num)

    # Unidades militares
    unidades = re.findall(r'\d+[ºª]\s*\w+', texto)
    entidades['unidades'].extend(unidades)

    return entidades

# Testar
msg = 'O Sargento Oliveira precisa trocar o serviço de quarta com o Cabo Ferreira do 3º BIL'
resultado = extrair_entidades_militar(msg)

print(f'Mensagem: "{msg}"')
print()
for tipo, valores in resultado.items():
    if valores:
        print(f'  {tipo.upper()}: {valores}')



## Passo 8 - Reconhecimento de Intenção


def classificar_intencao(texto):
    """Classifica a intenção da mensagem usando palavras-chave."""
    texto_lower = texto.lower()

    # Dicionário de intenções e suas palavras-chave
    intencoes = {
        'TROCA_SERVICO': ['trocar', 'troca', 'permuta', 'permutar', 'substituir', 'substituição'],
        'CONSULTA_ESCALA': ['consultar', 'consulta', 'ver', 'qual', 'quando', 'horário', 'escala', 'minha escala'],
        'RECLAMACAO': ['absurdo', 'injusto', 'insatisfeito', 'reclamar', 'reclamação', 'problema', 'errado'],
        'CONFIRMACAO': ['confirmo', 'confirmar', 'presente', 'estarei', 'ok', 'ciente', 'afirmativo'],
        'SAUDACAO': ['bom dia', 'boa tarde', 'boa noite', 'olá', 'oi'],
    }

    # Verificar cada intenção
    scores = {}
    for intencao, palavras in intencoes.items():
        score = sum(1 for p in palavras if p in texto_lower)
        if score > 0:
            scores[intencao] = score

    if scores:
        return max(scores, key=scores.get)
    return 'NAO_IDENTIFICADO'

# Testar com várias mensagens
mensagens_teste = [
    'Quero trocar minha escala de quarta com o Soldado Lima',
    'Quando é meu próximo serviço?',
    'É um absurdo essa escala injusta',
    'Confirmo presença no serviço de sexta-feira',
    'Bom dia, tudo bem?',
    'Preciso ver a escala do mês que vem',
]

print('RECONHECIMENTO DE INTENÇÃO')
print('=' * 60)
for msg in mensagens_teste:
    intencao = classificar_intencao(msg)
    print(f'\n"{msg}"')
    print(f'  → Intenção: {intencao}')



# Passo 9 - Montando o assitente completo

def gerar_resposta(intencao, entidades, sentimento):
    respostas = {
        'TROCA_SERVICO': 'Pedido de troca registrado. Vou verificar a disponibilidade e retorno em breve.',
        'CONSULTA_ESCALA': 'Consultando a escala de serviço... Sua próxima escala está prevista para os dias indicados no quadro de serviço.',
        'RECLAMACAO': 'Sua reclamação foi registrada e será encaminhada ao responsável pela escala para análise.',
        'CONFIRMACAO': 'Presença confirmada com sucesso. Obrigado pela confirmação.',
        'SAUDACAO': 'Bom dia! Sou o assistente de escala de serviço. Como posso ajudar?',
        'NAO_IDENTIFICADO': 'Não entendi sua solicitação. Pode reformular? Posso ajudar com consultas, trocas e confirmações de escala.'
    }
    resposta = respostas.get(intencao, respostas['NAO_IDENTIFICADO'])

    # Personalizar se temos entidades
    if entidades['nomes']:
        resposta += f'\n   Militar(es) envolvido(s): {", ".join(entidades["nomes"])}'
    if entidades['datas']:
        resposta += f'\n   Data(s) mencionada(s): {", ".join(entidades["datas"])}'
    if entidades['unidades']:
        resposta += f'\n   Unidade: {", ".join(entidades["unidades"])}'

    #Alerta se sentimento for negativo
    if sentimento == "NEGATIVO":
      resposta += "\n Detectei insatisfação - priorizando atendimento"
    
    return resposta


def assistente_escala(mensagem):
    """Pipeline completo do assistente de escala de serviço."""
    print(f' Mensagem: "{mensagem}"')
    print('-' * 60)

    # 1. Análise de sentimento
    scores = sia.polarity_scores(mensagem)
    if scores['compound'] >= 0.05:
        sentimento = 'POSITIVO'
    elif scores['compound'] <= -0.05:
        sentimento = 'NEGATIVO'
    else:
        sentimento = 'NEUTRO'
    print(f'   Sentimento: {sentimento} ({scores["compound"]:.2f})')

    # 2. Classificar intenção
    intencao = classificar_intencao(mensagem)
    print(f'   Intenção: {intencao}')

    # 3. Extrair entidades
    entidades = extrair_entidades_militar(mensagem)
    for tipo, valores in entidades.items():
        if valores:
            print(f'   {tipo}: {valores}')

    # 4. Gerar resposta
    resposta = gerar_resposta(intencao, entidades, sentimento)
    print(f'\n Resposta:')
    print(f'  {resposta}')
    print()
    return resposta

    ## Passo 10 - Tetando o assitente

mensagens = [
    'Bom dia, preciso consultar a escala de serviço desta semana',
    'Solicito troca de serviço de quinta com o Cabo Ferreira do 2º BIL',
    'Estou insatisfeito com a escala, é um absurdo ter 3 serviços seguidos',
    'Confirmo presença no serviço de sábado',
    'O Sargento Ribeiro pode assumir meu serviço de terça?',
]

print(' ASSISTENTE DE ESCALA DE SERVIÇO')
print('=' * 60)
for msg in mensagens:
    assistente_escala(msg)
    print('=' * 60)