## Passo 1 Carregar as ferramentas

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

print("TensorFlow versão:", tf.__version__)


## Passo 2 - Dados para treinamento
        # Tipo 0 = MATERIAL  
        # Tipo 1 = MANUTENÇÃO  
        # Tipo 2 = LICENÇA


# Frases de exemplo - Tipo 0: Licença

frases_material = ["preciso de combustível para a viatura",
                   "solicito material de escritório",
                   "falta papel na impressora do setor",
                   "precisamos de diesel urgente",
                   "solicitar canetas e cadernos",
                   "o estoque de material de limpeza acabou",
                   "preciso de toner para impressora",
                   "requisição de combustível para viatura",
                   "solicito compra de material de expediente",
                   "precisamos repor o estoque de papel", ]
                  
print(f"manutenção: {len(frases_material)} frases")


# Frases de exemplo - Tipo 1: Manutenção

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


# Frases de exemplo - Tipo 2: Licença
frases_licenca = [
    "solicito licença para o dia quinze",
    "preciso de dispensa médica",
    "solicitar férias para o próximo mês",
    "pedido de licença por motivo familiar",
    "solicito abono de falta",
    "preciso sair mais cedo hoje",
    "solicitação de transferência de setor",
    "pedido de licença para consulta médica",
    "solicito dispensa para resolver assunto pessoal",
    "preciso de autorização para ausência",
]

print(f"Licença: {len(frases_licenca)} frases")


# Frases de treino que tem todos os tipos de frases misturados 

frases_treino = frases_material + frases_licenca + frases_manutencao

# Rotulos: 0: Material, 1: Licença, 2: Manutenção
rotulos_treino = [0]*10 + [1]*10 + [2]*10

# Nomes para mostrar o resultado bonito
nome_tipos = ["MATERIAL","MANUTENÇÃO","LICENÇA"]

print(f"Total: {len(frases_treino)} frases, {len(rotulos_treino)} rótulos")

## PAsso 3 - Tranformar texto em numeros

          # 1 - Criar um dicionario (cada palavra = um numero) 
          # 2 - Converter as frases em sequência de números
          # 3 - Padronizar o tamnho com zeros (padding)

# Criando o dicionário

tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(frases_treino)

# Verificando o dicionario
#print("Dicionario (primeiras 10 palavras): ")

for palavra, numero in list(tokenizer.word_index.items())[:10]:
  print(f' "{palavra}" {numero}')


# Converter as frases em sequencia de números 

sequencias = tokenizer.texts_to_sequences(frases_treino)

# Padronizar o tamanho (Todas ficam com 10 números)

x_treino = pad_sequences(sequencias, maxlen=10, padding='pre')
y_treino = np.array(rotulos_treino)

# Validação de exemplo

print(f'Frases: "{frases_treino[0]}"')
print(f'Números: "{list(sequencias[0])}"')

print(f'Com Padding: "{list(x_treino[0])}"')
print(f'\nFormato: {x_treino.shape} → {x_treino.shape[0]} frases de {x_treino.shape[1]} números')



## Passo 4 - Montar a rede Neural

    # Vamos empilhar 4 camadas com sequencia

tamanho_vocabulario = len(tokenizer.word_index) + 1


modelo = keras.Sequential([
    keras.layers.Embedding(tamanho_vocabulario, 16, input_length=10),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax')    
])

modelo.summary()



  # Compilar = preparar para testar

modelo.compile(
    loss='sparse_categorical_crossentropy',   # é o medidor de erros. O sparse é o rotulo
    optimizer='adam',                         # é o algoritmo inteligente que vai no modelo e ajusta
    metrics=['accurancy']                     # o placar mostra a  porcentagem de acertos e erros
)

print("Modelo Compilado")


