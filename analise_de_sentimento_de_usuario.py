
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator
import re

print("bibliotecas carrecagas")


## Passo 3 - Configurações

NUM_PALAVRAS = 10000
TAMANHO_MAXIMO = 200


## Passo 4 - treinamneot do metodo

    # Carregar dados

(entrada_treino, saida_treino), (entrada_teste, saida_teste) = imdb.load_data(num_words=NUM_PALAVRAS)
entrada_treino = pad_sequences(entrada_treino, maxlen=TAMANHO_MAXIMO)
entrada_teste = pad_sequences(entrada_teste, maxlen=TAMANHO_MAXIMO)


## Passo 5 - Criar a rede neural 

modelo = keras.Sequential([
    layers.Embedding(NUM_PALAVRAS, 64, input_length=TAMANHO_MAXIMO),
    layers.Conv1D(64, 5, activation='relu'), # Fazer um Filtro de palavras Especiais
    layers.GlobalMaxPool1D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
modelo.summary()


## Passo 6 - Compilar e treinar


modelo.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
modelo.fit(entrada_treino, saida_treino, epochs=15, batch_size=512, validation_split=0.2, verbose=1)


erro,acuracia = modelo.evaluate(entrada_teste, saida_teste)
print(f"\n Acurácia:  {acuracia:.2%}")



## Passo 7 - desenvolvimento do ChatBot

dicionario= imdb.get_word_index()
tradutor = GoogleTranslator(source='pt', target='en')

def preparar(texto_pt):
  texto_en = tradutor.translate(texto_pt)
  limpo = re.sub(r'[^a-z\s]', ' ', texto_en.lower())
  sequencia = [dicionario.get(p, 0) + 3 for p in limpo.split() if dicionario.get(p, 0) + 3 < NUM_PALAVRAS]
  return pad_sequences([sequencia], maxlen=TAMANHO_MAXIMO), texto_en



  ## Passo 8 - O meu ChatBot


## Passo 8 - O meu ChatBot


print("\n" + "=" * 50)
print("CHATBOT ANALISADOR DE SENTIMENTOS")
print("=" * 50)

while True:
  texto = input('Você: ')
  if texto.lower() in ["sair", "exit", "quit" ]:
    print("\n Até a proxima!")
    break
  if not texto.strip():
    print("Digite algo! \n")
    continue

  entrada, en = preparar(texto)
  probabilidade = modelo.predict(entrada, verbose=0)[0][0]
  print(f'\nTradução: \"{en}\"')
  if probabilidade>= 0.7:
    print(f'POSITIVA({probabilidade:.1%}) - Gostou Bastante')
  elif probabilidade>= 0.5:
    print(f'LEVEMENTE POSITIVA({probabilidade:.1%}) - Gostou, com ressalvas.')
  elif probabilidade>= 0.3:
    print(f'LEVEMENTE NEGATIVA({probabilidade:.1%}) - Não curtiu muito.')
  else:
    print(f'NEGATIVA({probabilidade:.1%}) - Não Gostou')



