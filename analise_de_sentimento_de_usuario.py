
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


