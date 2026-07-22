
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator

print("bibliotecas carrecagas")


## Passo 3 - Configurações

NUM_PALAVRAS = 10000
TAMANHO_MAXIMO = 200


## Passo 4 - treinamneot do metodo

    # Carregar dados

(entrada_treino, saida_treino), (entrada_teste, saida_teste) = imdb.load_data(num_words=NUM_PALAVRAS)
entrada_treino = pad_sequences(entrada_treino, maxlen=TAMANHO_MAXIMO)
entrada_teste = pad_sequences(entrada_teste, maxlen=TAMANHO_MAXIMO)