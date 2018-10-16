
# coding: utf-8

# # Differentiate Text Message
# 
# In this project you will explore text message data and create models to predict if a message is spam or not. 

# In[ ]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)


# In[ ]:


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[ ]:


def answer_one():
    
    
    return len(spam_data[spam_data['target']==1])/len(spam_data)*100


# In[ ]:


answer_one()


# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    
    vect = CountVectorizer().fit(X_train)
    
    return max(vect.get_feature_names(), key=lambda token: len(token))


# In[ ]:


answer_two()


# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_vectorized, y_train)
    predictions = clf.predict(vect.transform(X_test))
    
    return roc_auc_score(y_test, predictions)


# In[ ]:


answer_three()


# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    feature_names = np.array(vect.get_feature_names())
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    small_index = feature_names[sorted_tfidf_index[:20]]
    small_value = X_train_vectorized.max(0).toarray()[0][sorted_tfidf_index[:20]]
    small_tuple = [(value, word) for word, value in zip(small_index, small_value)]
    small_tuple.sort()
    small_index = [element[1] for element in small_tuple]
    small_value = [element[0] for element in small_tuple]
    small_series = pd.Series(small_value, index=small_index)
    
    large_index = feature_names[sorted_tfidf_index[-20:]]
    large_value = X_train_vectorized.max(0).toarray()[0][sorted_tfidf_index[-20:]]
    large_tuple = [(-value, word) for word, value in zip(large_index, large_value)]
    large_tuple.sort()
    large_index = [element[1] for element in large_tuple]
    large_value = [-element[0] for element in large_tuple]
    large_series = pd.Series(large_value, index=large_index)
    
    return (small_series, large_series)


# In[ ]:


answer_four()


# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[ ]:


def answer_five():
    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(vect.transform(X_test))
    
    return roc_auc_score(y_test, predictions)


# In[ ]:


answer_five()


# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[ ]:


def answer_six():
    
    spam_data['length'] = spam_data['text'].str.len()
    nonSpam = spam_data[spam_data['target']==0]
    Spam = spam_data[spam_data['target']==1]
    
    return (nonSpam['length'].sum()/len(nonSpam), Spam['length'].sum()/len(Spam))


# In[ ]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[ ]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[ ]:


from sklearn.svm import SVC

def answer_seven():
    
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_test_vectorized = vect.transform(X_test)
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    model = SVC(C=10000)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    
    return roc_auc_score(y_test, predictions)


# In[ ]:


answer_seven()


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[ ]:


def answer_eight():
    
    import re
    spam =[re.findall("[0-9]", i) for i in spam_data['text'][spam_data['target']==1]]
    non_spam =[re.findall("[0-9]", i) for i in spam_data['text'][spam_data['target']==0]]
    
    return (np.mean(list(map(len, non_spam))), np.mean(list(map(len, spam))))


# In[ ]:


answer_eight()


# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[ ]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_train_digits = X_train.str.findall(r'(\d)')
    X_train_vectorized = add_feature(X_train_vectorized, list(map(len, X_train_digits)))
    
    X_test_vectorized = vect.transform(X_test)
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    X_test_digits = X_test.str.findall(r'(\d)')
    X_test_vectorized = add_feature(X_test_vectorized, list(map(len, X_test_digits)))
    
    model = LogisticRegression(C=100)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    
    return roc_auc_score(y_test, predictions)


# In[ ]:


answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[ ]:


def answer_ten():
    
    import re
    spam =[re.findall("\W", i) for i in spam_data['text'][spam_data['target']==1]]
    non_spam =[re.findall("\W", i) for i in spam_data['text'][spam_data['target']==0]]
    
    return (np.mean(list(map(len, non_spam))), np.mean(list(map(len, spam))))


# In[ ]:


answer_ten()


# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[ ]:


def answer_eleven():
    
    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_train_digits = X_train.str.findall(r'(\d)')
    X_train_vectorized = add_feature(X_train_vectorized, list(map(len, X_train_digits)))
    X_train_nonchar = X_train.str.findall(r'(\W)')
    X_train_vectorized = add_feature(X_train_vectorized, list(map(len, X_train_nonchar)))
    
    X_test_vectorized = vect.transform(X_test)
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    X_test_digits = X_test.str.findall(r'(\d)')
    X_test_vectorized = add_feature(X_test_vectorized, list(map(len, X_test_digits)))
    X_test_nonchar = X_test.str.findall(r'(\W)')
    X_test_vectorized = add_feature(X_test_vectorized, list(map(len, X_test_nonchar)))
    
    model = LogisticRegression(C=100)
    model.fit(X_train_vectorized, y_train)
    predictions = model.predict(X_test_vectorized)
    
    feature_names = np.array(vect.get_feature_names())
    feature_names = np.concatenate((feature_names, np.array(['length_of_doc', 'digit_count', 'non_word_char_count'])))
    
    sorted_coef_index = model.coef_[0].argsort()
    small = list(feature_names[sorted_coef_index[:10]])
    large = list(feature_names[sorted_coef_index[:-11:-1]])
    
    return roc_auc_score(y_test, predictions), small, large


# In[ ]:


answer_eleven()

