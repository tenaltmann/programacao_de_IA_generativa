## Passo 1 Carregar as ferramentas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

print("TensorFlow versão:", tf.__version__)


frases_manutencao = [
    "o ar condicionado esta quebrado",
    "preciso de conserto na porta do alojamento",
    "a viatura esta com problema no motor",
    "o computador do setor não liga",
    "goteira no teto da sala de aula",
    "a impressora parou de funcionar",
    "precisamos consertar a fechadura",
    "o elevador esta com defeito",
    "a rede elétrica precisa de reparo",
    "equipamento com defeito precisa de manutenção", 
]
print(f"Manutenção: {len(frases_manutencao)} frases")