## Passo 1 Carregar as ferramentas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

print("TensorFlow versão:", tf.__version__)
