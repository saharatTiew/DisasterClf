import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from joblib import dump, load
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.corpus.reader import wordnet
from nltk.stem import LancasterStemmer, PorterStemmer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

# in train set, keyword has NaN (missing value) = 61 rows
# in train set, location has NaN (missing value) = 2533 rows

TRAINING_FILE_NAME = 'train.csv'
KEYWORD_OHE_PATH = './lib/keyword_ohe.lib'
TEXT_VECTORIZER_PATH = './lib/text_vectorizer.lib'
TRAINING_DATAFRAME_PATH = './lib/training_df.lib'
TARGET_DATAFRAME_PATH = './lib/target_df.lib'
KEYWORD_LBE_PATH = './lib/keyword_lbe.lib'
USE_LABEL_ENCODER = False
SAVE_MODEL = False
USE_LEMMATIZER = False
USE_LANCASTER_STEM = False

def download_nltk_package():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('words')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def data_info(df):
    print('keyword features')
    print('-------------------------------------')
    feature_keywords = df['keyword'].value_counts()
    print(feature_keywords)
    print('######################################')
    print(f'there are {feature_keywords.count()} unique features')
    print('-------------------------------------')
    
    print()

    print('location features')
    print('-------------------------------------')
    feature_locations = df['location'].value_counts()
    print(feature_locations)
    print('######################################')
    print(f'there are {feature_locations.count()} unique features')
    print('-------------------------------------')
    
    print()

    print('label')
    print('-------------------------------------')
    feature_locations = df['target'].value_counts()
    print(feature_locations)
    
    print()

def read_csv(file_name):
    data = pd.read_csv(file_name)

    # create dataframe
    train_df = pd.DataFrame(data)

    return train_df

# find part of speech of word
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.VERB

def pre_process_text(df, use_lemmatizer, use_lancaster_stem):
    words = set(nltk.corpus.words.words())
    lemmatizer = WordNetLemmatizer()
    lancaster_stemmer = LancasterStemmer()
    porter_stemmer = PorterStemmer()
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
        
        # tokenize word (change to list of terms)
        text_tokenize = word_tokenize(text)
        
        # lemmatize (or stem, depends on the option) every word
        root_texts = []
        for word in text_tokenize:
            if use_lemmatizer:
                root_texts.append(lemmatizer.lemmatize(word, get_wordnet_pos(word)))
            elif use_lancaster_stem:
                root_texts.append(lancaster_stemmer.stem(word))
            else:
                root_texts.append(porter_stemmer.stem(word))

        # transform list to string 
        text = " ".join(root_texts)

        texts.append(text)

    df['text'] = texts
    # print(df['text'])
    return df
        
def pre_processing(df, keyword_ohe_path, keyword_lbe_path, text_vectorizer_path, 
                   df_path, save_model, use_label_encoder, use_lemmmatizer, 
                   use_lancaster_stem, target_df_path, vectorizer_input):

    # drop column location which have so many missing values
    # preprocess_df = df.drop(columns='location')
    # drop all row that column keyword is NaN
    # preprocess_df = preprocess_df.dropna()
    # separate label from other attributes and reset index of the label
    # label = preprocess_df.iloc[:,-1]
    # label = label.reset_index(drop=True)
    preprocess_df = df
    shape = preprocess_df.shape[1]

    if use_label_encoder:
        # encode the keyword column using label encoder
        encoder =  LabelEncoder()
        preprocess_df['keyword'] = encoder.fit_transform(preprocess_df['keyword'])

        if save_model:
            dump(encoder, keyword_lbe_path)
    else: 
        # encode the keyword column using one hot encoder
        encoder = OneHotEncoder()
        keyword_temp = np.array(preprocess_df['keyword']).reshape(-1,1)
        keyword_encoder = encoder.fit_transform(keyword_temp).toarray()

        new_keyword = pd.DataFrame(keyword_encoder)

        # dump keyword encoder
        if save_model:
            dump(encoder, keyword_ohe_path)

        # concat encoded keyword back to the dataset
        preprocess_df = pd.concat([preprocess_df.reset_index(drop=True), new_keyword.reset_index(drop=True)],axis=1)
        preprocess_df = pd.DataFrame(preprocess_df)
        preprocess_df.rename(columns=dict(zip(preprocess_df.columns[shape:], 
                                np.array(encoder.categories_).ravel())), inplace=True)

    # perform text cleaning
    preprocess_df = pre_process_text(preprocess_df, use_lemmmatizer, use_lancaster_stem)
    
    vectorizer = None

    if vectorizer_input is None:
        # vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True)
        vectorizer = CountVectorizer(stop_words='english')
        text_vector = vectorizer.fit_transform(preprocess_df['text']).toarray()
    else:
        vectorizer = vectorizer_input
        text_vector = vectorizer_input.transform(preprocess_df['text']).toarray()

    # Truncated svd to remove dimensionality for sparse data
    svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
    # svd = TruncatedSVD(n_components=2, n_iter=10, random_state=42)
    text_vector_tran = svd.fit_transform(text_vector)
    
    new_text = pd.DataFrame(text_vector_tran)
    # new_text = pd.DataFrame(text_vector)

    # dump text vectorizer
    if save_model:
        dump(vectorizer, text_vectorizer_path)

    # drop column keyword and text
    if not use_label_encoder:
        preprocess_df = preprocess_df.drop(columns='keyword')

    preprocess_df = preprocess_df.drop(columns='text')
    shape_2 = preprocess_df.shape[1]

    # concat vector of text to the dataset
    preprocess_df = pd.concat([preprocess_df.reset_index(drop=True), new_text.reset_index(drop=True)],axis=1)
    preprocess_df.rename(columns=dict(zip(preprocess_df.columns[shape_2:], 
                            vectorizer.get_feature_names())), inplace=True)
    
    # dump dataframe
    if save_model:
        dump(preprocess_df, df_path)

    return preprocess_df, vectorizer


# download_nltk_package()
df = read_csv(TRAINING_FILE_NAME)


X = df.drop(columns='location')
# drop all row that column keyword is NaN
X = X.dropna()

y = X['target']
X = X.drop(columns=['id', 'target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

X_train, vectorizer = pre_processing(X_train, KEYWORD_OHE_PATH, KEYWORD_LBE_PATH, TEXT_VECTORIZER_PATH, 
                    TRAINING_DATAFRAME_PATH, SAVE_MODEL, USE_LABEL_ENCODER, 
                    USE_LEMMATIZER, USE_LANCASTER_STEM, TARGET_DATAFRAME_PATH, None)

X_test, _ = pre_processing(X_test, KEYWORD_OHE_PATH, KEYWORD_LBE_PATH, TEXT_VECTORIZER_PATH, 
                    TRAINING_DATAFRAME_PATH, SAVE_MODEL, USE_LABEL_ENCODER, 
                    USE_LEMMATIZER, USE_LANCASTER_STEM, TARGET_DATAFRAME_PATH, vectorizer)

# clf = LinearSVC()
# clf = RandomForestClassifier()
clf = BernoulliNB()
# clf = LogisticRegression(c=1, penalty='l2')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))