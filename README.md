# DisasterClf

## Description
These code are used to train the model used for the competition "Real or Not? NLP with Disaster Tweets" 
> https://www.kaggle.com/c/nlp-getting-started/overview/description

## Approach
1. Normal Machine Learning
2. Topic Modeling output as features to train Supervised Machine Learning
3. Neural Network (GloVe, LSTM)

## Accuracy
1. Bernoulli Naive Bayes with Lemmatization, OneHotEncoder, CountVectorizer -> 78.92 %
2. 222 Topic Unigram with Random Forest Classifier -> 61.87 %
3. GloVe With LSTM -> 79.20 %

## Conclusion
By far the approach that generate the best accuracy is Neural Network using GloVe with LSTM
