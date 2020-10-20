import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords, wordnet
import re
import string
from nltk.tokenize import word_tokenize
from keras.layers import Embedding, LSTM, SpatialDropout1D, Dense
from keras.initializers import Constant
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


GLOVE = './glove/glove.6B.100d.txt'
DIMS = 100
TRAINING_FILE_NAME = 'dataset/train.csv'

def read_csv(file_name):
    data = pd.read_csv(file_name)

    # create dataframe
    train_df = pd.DataFrame(data)

    return train_df

def pre_process_text(df):
    words = set(nltk.corpus.words.words())
    stop_words = set(stopwords.words('english'))
    texts = []
    for _, row in df.iterrows():
        text = row['text']
        # remove word that is not in English corpus and transform them to lower case
        text = " ".join(w.lower() for w in nltk.wordpunct_tokenize(text) if w.lower() in words)

        # # remove http tag
        text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', text)
        
        #remove number
        text = re.sub(r'\d+','',text)
        
        #remove punctuation mark
        text = text.translate(str.maketrans('','', string.punctuation))
        
        #remove extra white space
        text = text.strip()
        texts.append(text)

    df['text'] = texts
    # print(df['text'])
    return df

def metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print("-"*50)
    print(classification_report(pred_tag, y_test))


df = read_csv(TRAINING_FILE_NAME)
df = pre_process_text(df)
X = df['text'].values
y = df['target'].values

tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(X)
vocab_length = len(tokenizer.word_index) + 1

# read GloVe and save into embedding_dict
embedding_dict= {}
with open(GLOVE,'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict[word] = vectors
print("words loaded!")
f.close()

# store word that is in GloVe in embedding_matrix
embedding_matrix = np.zeros((vocab_length , DIMS))
print(tokenizer.word_index.items())
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_dict.get(word)

    # words not found in embedding index will be all-zeros.
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

# find longest tokenenized sentence and convert other sentence to the same length
longest_train = max(X, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))
padded_sentence = pad_sequences(tokenizer.texts_to_sequences(), length_long_sentence, padding='post')

embedding = Embedding(input_dim=embedding_matrix.shape[0]
                      ,output_dim=embedding_matrix.shape[1]
                      ,embeddings_initializer=Constant(embedding_matrix)
                      ,input_length=length_long_sentence)

model = Sequential()
model.add(embedding)
# regularization technique, which aims to reduce the complexity of the model with the goal to prevent overfitting.
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
# simply a layer where each unit or neuron is connected to each neuron in the next layer.
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=3e-4)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

print(model.summary())
X_train, X_test, y_train, y_test = train_test_split(padded_sentence, y,test_size=0.25)

checkpoint = ModelCheckpoint(
    'model.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)

history = model.fit(X_train
                    ,y_train
                    ,epochs=7
                    ,batch_size=32
                    ,validation_data=[X_test, y_test]
                    ,verbose = 1
                    ,callbacks= [reduce_lr, checkpoint])

loss, accuracy = model.evaluate(X_test, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)

preds = model.predict_classes(X_test)
metrics(preds, y_test)

model.load_weights('model.h5')
preds = model.predict_classes(X_test)
metrics(preds, y_test)

# sequence = tokenizer.texts_to_sequences(texts)
# print(tokenizer.word_index)
# print(sequence)
