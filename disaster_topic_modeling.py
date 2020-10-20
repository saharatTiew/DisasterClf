import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

TOPIC_5_BIGRAM = './topic_modeling_data/topic-5-bigram-dis-vector.csv'
TOPIC_5 = './topic_modeling_data/topic-5-dis-vector.csv'
TOPIC_222_BIGRAM = './topic_modeling_data/topic-222-bigram-dis-vector.csv'
TOPIC_222 = './topic_modeling_data/topic-222-dis-vector.csv'

def read_csv(file_name):
    data = pd.read_csv(file_name)

    # create dataframe
    train_df = pd.DataFrame(data)

    return train_df

# df = read_csv(TOPIC_5_BIGRAM).drop_duplicates()
df = read_csv(TOPIC_222_BIGRAM)
# print(df)
# d_1 = df.loc[df['target'] == 1]
# d_0 = df.loc[df['target'] == 0]

# d_1.to_csv('1.csv', index=False)
# d_0.to_csv('0.csv', index=False)

y = df['target']
X = df.drop(columns=['Unnamed: 0', 'target'])
# pca = PCA(n_components=20)
# X = pca.fit_transform(X)
# print(X)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# print(X)

# feature_locations = df['target'].value_counts()
# print(feature_locations)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# clf = BernoulliNB()
# clf = MultinomialNB()
# clf = GaussianNB()
# clf = RandomForestClassifier(n_jobs=3, n_estimators=500, verbose=True)
# clf = SVC(kernel='linear')
clf = LogisticRegression(class_weight='balanced', solver='newton-cg')
# clf = AdaBoostClassifier(n_estimators=500)
clf.fit(X_train, y_train)
print(clf)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))