## Instalação

# transformer - padrao da industria para modelos de ia 
# sentencepiece e protbuf - processamento de vocabulario dos modelos
# spacy - processamento de linguagem natural
#
#
print("instalação concluida")




## Passo 2 - Carregando os modelos e importando as libs


from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import spacy
import re


print('Carregando o modelo de sentimento...')
analisador_sentimento = pipeline(
    'sentiment-analysis',
    model='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
    top_k= None # retorna score de todas as classes
)
print("Sentimento OK")


# Modelo de INTENÇÃO
# Não precisa de treino específico 

classificador_intencao = pipeline(
    'zero-shot-classification',
    model='joeddav/xlm-roberta-large-xnli'
)

print('Intenção OK')

# Modelo Generativo para respostas convencionais em português utilizando T5
print('Carregando modelo de geração de texto...')
t5_tokenizer = T5Tokenizer.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
t5_model = T5ForConditionalGeneration.from_pretrained('unicamp-dl/ptt5-base-portuguese-vocab')
print('Gerador de texto OK')

# spacy para entidades
nlp = spacy.load('pt_core_news_sm')
print("Spacy OK")
print("Todos os modelos foram carregados")





# Passo 3 - Teste rapido do modelo generativo



def gerar_com_t5(prompt, max_length=80):
  "Gerar Textos utilizando o PTT5"
  inputs = t5_tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
  outputs = t5_model.generate(
      **inputs,
      max_length=max_length,
      num_beams=4,
      early_stopping=True,
      no_repeat_ngram_size=2
  )
  return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

prompts_teste=[
    'O serviço militar é importante porque',
    'A escala de serviço precisa ser',
    'Para solicitar troca de serviço',
]

print("TESTE DO MODELO GENERATIVO (PTT5)")
print("=" * 50)
for prompt in prompts_teste:
  resultado = gerar_com_t5(prompt)
  print(f'\nPrompt: {prompt}')
  print(f'{resultado}')


def analisar_sentimento(texto):
    """Analisa sentimento usando modelo multilíngue treinado."""
    resultado = analisador_sentimento(texto)[0]

    # Mapear labels para português
    mapa = {'positive': 'POSITIVO', 'negative': 'NEGATIVO', 'neutral': 'NEUTRO'}

    # Pegar o sentimento com maior score
    melhor = max(resultado, key=lambda x: x['score'])
    label_pt = mapa.get(melhor['label'], melhor['label'])

    return label_pt, melhor['score'], resultado

# Testar com frases que o VADER errava
frases = [
    'A escala deste mês ficou muito bem organizada, parabéns',
    'Preciso consultar minha escala da próxima semana',
    'É um absurdo eu estar de serviço no terceiro domingo seguido',
    'Confirmo que estarei presente no serviço de quarta',
    'Até que enfim uma escala justa, obrigado',
    'Estou insatisfeito com a distribuição das escalas',
    'Que bom que consegui a troca, valeu Sargento!',
    'Ninguém merece ficar de serviço no Natal de novo',
]

print('ANÁLISE DE SENTIMENTO — MODELO TREINADO')
print('=' * 65)
for frase in frases:
    sentimento, confianca, _ = analisar_sentimento(frase)
    emoji = {'POSITIVO': '😊', 'NEGATIVO': '😠', 'NEUTRO': '😐'}.get(sentimento, '❓')
    print(f'\n"{frase}"')
    print(f'  → {emoji} {sentimento} (confiança: {confianca:.1%})')




    ### PASSO 5  intenção do modelo


INTENCOES = [
    'solicitar troca de escala de serviço',
    'consultar escala ou horário de serviço',
    'fazer reclamação sobre a escala',
    'confirmar presença no serviço',
    'pedir informação geral',
    'cumprimentar ou saudação',
]

def classificar_intencao(texto):
  resultado = classificador_intencao(
      texto,
      candidate_labels=INTENCOES,
      hypothesis_template = "Este texto é sobre {}."
  )
  return resultado['labels'][0], resultado['scores'][0], resultado


# Testar
mensagens = [
    'Quero trocar minha escala de quarta com o Soldado Lima',
    'Quando é meu próximo serviço?',
    'É um absurdo essa escala injusta',
    'Confirmo presença no serviço de sexta-feira',
    'Bom dia, tudo bem?',
    'Quantas pessoas ficam no serviço noturno?',
    'Não concordo com a escala, quero falar com o responsável',
    'Posso pegar o serviço do Cabo Mendes na terça?',
]

print('CLASSIFICAÇÃO DE INTENÇÃO — ZERO-SHOT')
print('=' * 65)
for msg in mensagens:
    intencao, confianca, _ = classificar_intencao(msg)
    print(f'\n"""{msg}"""')
    print(f'   {intencao} ({confianca:.1%})')


    ## Passo 6 - Extração de entidades

def extrair_entidades(texto):
    """Extrai entidades relevantes usando spaCy + regras customizadas."""
    doc = nlp(texto)
    entidades = {'nomes': [], 'datas': [], 'unidades': [], 'spacy_ents': []}

    # Entidades do spaCy
    for ent in doc.ents:
        entidades['spacy_ents'].append(f'{ent.text} ({ent.label_})')

    # Postos + nomes (regex)
    postos = ['Soldado', 'Cabo', 'Sargento', 'Terceiro-Sargento', 'Segundo-Sargento',
              'Primeiro-Sargento', 'Subtenente', 'Tenente', 'Capitão', 'Major',
              'Coronel', 'General']
    for posto in postos:
        matches = re.findall(rf'{posto}\s+([A-Z][a-záéíóú]+)', texto)
        for nome in matches:
            entidades['nomes'].append(f'{posto} {nome}')

    # Datas
    dias = re.findall(r'(segunda|terça|quarta|quinta|sexta|sábado|domingo)(?:-feira)?', texto, re.IGNORECASE)
    entidades['datas'].extend(dias)
    datas_num = re.findall(r'dia\s+(\d{1,2}(?:\s+de\s+\w+)?)', texto, re.IGNORECASE)
    entidades['datas'].extend(datas_num)

    # Unidades
    unidades = re.findall(r'\d+[ºª]\s*\w+', texto)
    entidades['unidades'].extend(unidades)

    return entidades

# Teste rápido
msg = 'O Sargento Oliveira quer trocar com o Cabo Ferreira do 3º BIL na sexta'
ents = extrair_entidades(msg)
print(f'Mensagem: "{msg}"\n')
for tipo, valores in ents.items():
    if valores:
        print(f'  {tipo}: {valores}')



### Passo 7 - Respostas Conversacionais


import random

def gerar_resposta_conversacional(intencao, entidades, sentimento):
    """Gera respostas naturais e contextualizadas."""

    # Templates variados para cada intenção (simula conversação natural)
    templates = {
        'solicitar troca de escala de serviço': [
            'Entendido! Vou registrar o pedido de troca{nomes}. Preciso confirmar a disponibilidade{datas} e retorno assim que tiver uma resposta do responsável pela escala.',
            'Pedido de permuta recebido{nomes}. Vou verificar se há compatibilidade de horário{datas} e te dou um retorno em breve.',
            'Certo, vou encaminhar a solicitação de troca{nomes}{datas}. Fique atento que o responsável pela escala vai confirmar.',
        ],
        'consultar escala ou horário de serviço': [
            'Consultando a escala{datas}... De acordo com o quadro de serviço atual, sua próxima escala está prevista conforme publicado. Quer que eu verifique algum dia específico?',
            'Deixa eu verificar{datas}... A escala vigente está publicada no quadro. Posso te ajudar com algum detalhe específico?',
            'Vou buscar a informação da escala{datas} pra você. Enquanto isso, lembre que o quadro atualizado fica disponível no mural da companhia.',
        ],
        'fazer reclamação sobre a escala': [
            'Entendo sua insatisfação e vou registrar essa reclamação formalmente. Ela será encaminhada ao responsável pela confecção da escala para análise. Quer adicionar mais algum detalhe?',
            'Sua reclamação foi anotada. Vou garantir que o responsável pela escala tome ciência. Se preferir, posso agendar uma conversa direta com ele.',
            'Registrei a reclamação. É importante que esse tipo de feedback chegue ao responsável, e vou encaminhar agora. Quer protocolar formalmente?',
        ],
        'confirmar presença no serviço': [
            'Presença confirmada{datas}! Obrigado pela confirmação. Lembre-se de verificar o horário de apresentação no quadro.',
            'Confirmado{datas}! Anotei sua presença. Qualquer imprevisto, me avise com antecedência.',
            'Certo, confirmação registrada{datas}. Bom serviço!',
        ],
        'pedir informação geral': [
            'Posso ajudar com consultas de escala, pedidos de troca, confirmações de presença e registrar reclamações. O que você precisa?',
            'Estou aqui para ajudar! Posso consultar escalas, registrar trocas, confirmar presença ou encaminhar reclamações. Como posso te ajudar?',
        ],
        'cumprimentar ou saudação': [
            'Bom dia! Sou o assistente de escala de serviço. Posso te ajudar com consultas, trocas, confirmações ou reclamações. O que precisa?',
            'Olá! Tudo bem? Estou à disposição para ajudar com a escala de serviço. É só falar!',
            'E aí! Beleza? Sou o assistente de escala. Pode mandar sua dúvida!',
        ],
    }

    # Escolher template
    opcoes = templates.get(intencao, ['Não entendi bem. Pode reformular? Posso ajudar com escalas, trocas e confirmações.'])
    resposta = random.choice(opcoes)

    # Inserir entidades
    nomes_str = f' envolvendo {" e ".join(entidades["nomes"])}' if entidades['nomes'] else ''
    datas_str = f' para {", ".join(entidades["datas"])}' if entidades['datas'] else ''
    resposta = resposta.replace('{nomes}', nomes_str).replace('{datas}', datas_str)

    # Adicionar alerta se sentimento negativo
    if sentimento == 'NEGATIVO':
        resposta += '\nDetectei que você não está satisfeito — vou priorizar seu atendimento.'

    return resposta

