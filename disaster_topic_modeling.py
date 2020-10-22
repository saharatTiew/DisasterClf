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

TOPIC_5_BIGRAM_NEW = './topic_modeling_data/topic_modeling_new/topic-5-bigram-new-dis-vector.csv'
TOPIC_5_NEW = './topic_modeling_data/topic_modeling_new/topic-5-new-dis-vector.csv'
TOPIC_222_BIGRAM_NEW = './topic_modeling_data/topic_modeling_new/topic-222-bigram-new-dis-vector.csv'
TOPIC_222_NEW = './topic_modeling_data/topic_modeling_new/topic-222-new-dis-vector.csv'

TEST_PREDICT_FILE = './dataset/topic-test-222-dis-vector.csv'
SUBMISSION_FILE = './submission/disaster_topic_modelling.csv'
SAMPLE_SUBMISSION_FILE = './dataset/sample_submission.csv'

def read_csv(file_name):
    print(str(file_name))
    print('------------------------------')
    data = pd.read_csv(file_name)

    # create dataframe
    train_df = pd.DataFrame(data)

    return train_df

df = read_csv(TOPIC_222_NEW)
# print(df)

y = df['target']
X = df.drop(columns=['Unnamed: 0', 'target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)

# classification model
# these 3 models, especially BernoulliNB does not work well with these dataset
# clf = BernoulliNB()
# clf = MultinomialNB()
# clf = SVC(kernel='linear')

# clf = GaussianNB()
clf = RandomForestClassifier(n_jobs=3, n_estimators=500, verbose=True)
# clf = LogisticRegression(class_weight='balanced', solver='newton-cg')
# clf = AdaBoostClassifier(n_estimators=500)

clf.fit(X_train, y_train)

print(clf)
y_pred = clf.predict(X_test)
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))


# prepare file to submission to kaggle
test = read_csv(TEST_PREDICT_FILE)
sample_sub= read_csv(SAMPLE_SUBMISSION_FILE)
test = test.drop(columns=['Unnamed: 0'])

sample_sub['target'] = clf.predict(test)
sample_sub.to_csv(SUBMISSION_FILE,index=False)