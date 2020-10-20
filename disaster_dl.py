import pandas as pd
import numpy as np

GLOVE = './glove/glove.6B.100d.txt'
embedding_dict= {}

with open(GLOVE,'r', encoding='utf-8') as f:
    for line in f:
        values=line.split()
        word = values[0]
        vectors=np.asarray(values[1:],'float32')
        embedding_dict[word]=vectors
print("words loaded!")
f.close()
# f = open(GLOVE, 'r', encoding='utf-8')
print(embedding_dict['the'])