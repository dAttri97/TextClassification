#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
get_ipython().magic(u'matplotlib inline')


# ### Improting the dataset

# In[2]:


df = pd.read_table('SMSSpamCollection',header=None,encoding='utf-8')


# In[3]:


print(df.info())
print(df.head(5))


# ### Data Preprocessing

# In[4]:


df.rename(columns={0:'Type',1:'Message'},inplace=True)


# In[5]:


df.head(5)


# The column have be renamed for better understanding of the user

# In[6]:


classes = df['Type']
print(classes.value_counts())
print(classes.value_counts()/len(classes))
(classes.value_counts()/len(classes)).plot.bar()


# More than 86% of the messages are useful and only 14 messages can be considered spam

# #### LabelEncoding the Types

# In[7]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(classes)
print(classes[:5])
print(Y[:5])


# The column type has been LabelEncoded for training the model

# **Separating the Messages for further processing**

# In[8]:


text = df['Message']
print(text[:10])
type(text)


# ### Regular Expression

# The messages have been randomly written and there is no guarantee of correct spelling and grammer.
# This can be challenging to process. To overcome such obstacles we must separate the messages with e-mails, phone numbers and urls and other numerical symbols.

# Some common regular expression metacharacters - copied from wikipedia
# 
# ^ Matches the starting position within the string. In line-based tools, it matches the starting position of any line.
# 
# . Matches any single character (many applications exclude newlines, and exactly which characters are considered newlines is flavor-, character-encoding-, and platform-specific, but it is safe to assume that the line feed character is included). Within POSIX bracket expressions, the dot character matches a literal dot. For example, a.c matches "abc", etc., but [a.c] matches only "a", ".", or "c".
# 
# [ ] A bracket expression. Matches a single character that is contained within the brackets. For example, [abc] matches "a", "b", or "c". [a-z] specifies a range which matches any lowercase letter from "a" to "z". These forms can be mixed: [abcx-z] matches "a", "b", "c", "x", "y", or "z", as does [a-cx-z]. The - character is treated as a literal character if it is the last or the first (after the ^, if present) character within the brackets: [abc-], [-abc]. Note that backslash escapes are not allowed. The ] character can be included in a bracket expression if it is the first (after the ^) character: []abc].
# 
# [^ ] Matches a single character that is not contained within the brackets. For example, [^abc] matches any character other than "a", "b", or "c". [^a-z] matches any single character that is not a lowercase letter from "a" to "z". Likewise, literal characters and ranges can be mixed.
# 
# $ Matches the ending position of the string or the position just before a string-ending newline. In line-based tools, it matches the ending position of any line.
# 
# ( ) Defines a marked subexpression. The string matched within the parentheses can be recalled later (see the next entry, \n). A marked subexpression is also called a block or capturing group. BRE mode requires ( ).
# 
# \n Matches what the nth marked subexpression matched, where n is a digit from 1 to 9. This construct is vaguely defined in the POSIX.2 standard. Some tools allow referencing more than nine capturing groups.
# 
# * Matches the preceding element zero or more times. For example, abc matches "ac", "abc", "abbbc", etc. [xyz] matches "", "x", "y", "z", "zx", "zyx", "xyzzy", and so on. (ab)* matches "", "ab", "abab", "ababab", and so on.
# 
# {m,n} Matches the preceding element at least m and not more than n times. For example, a{3,5} matches only "aaa", "aaaa", and "aaaaa". This is not found in a few older instances of regexes. BRE mode requires {m,n}.

# In[9]:


# Replace email addresses with 'email'
processed = text.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')

# Replace URLs with 'webaddress'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')

# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
    
# Replace numbers with 'numbr'
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')

# making all the msgs lower_case because Hello,HELLO and hello are all the same
processed = processed.str.lower()

print(processed.head(5))
print(processed.tail(5))


# In[10]:


# removing the stop words from the msgs as they don't add any useful info

from nltk.corpus import stopwords

words = set(stopwords.words('english'))

processed = processed.apply(lambda x:' '.join(term for term in x.split(' ') if term not in words))


# In[11]:


# removing the stem in words
stem = nltk.PorterStemmer()

processed = processed.apply(lambda x:' '.join(stem.stem(term) for term in x.split()))

print(processed[:5])


# In[12]:


# tokenizing all the msgs for word frequency
from nltk.tokenize import word_tokenize
 
all_words = []

for msg in processed:
    words = word_tokenize(msg)
    for word in words:
        all_words.append(word)
        
all_words = nltk.FreqDist(all_words)

#printing the neseccary Info
print("The total Number of unique words : {}".format(len(all_words)))

print("The Most common words in all the unique words: {}".format(all_words.most_common(15)))


# **Since we have a very large dataset we are only going to use the 1500 most common words to train our algo**

# In[13]:


word_feature = list(all_words.keys())[:1500]


# In[14]:


# defining a fucntion to find the common words in a particular msg

def find_feat(msg):
    words = word_tokenize(msg)
    dic = {}
    for word in word_feature:
        dic[word] = (word in words)
    return dic


# In[15]:


dic = find_feat(processed[0])
for key, value in dic.items():
    if value == True:
        print(key)


# In[16]:


# generating hte dictionary for every msg in the data set
messages = zip(processed,Y)

# defining the reproducable sedd
seed = 1
np.random.seed = seed
np.random.shuffle(messages) 

feature_set = [(find_feat(text),label) for (text,label) in messages]


# In[17]:


# spiliting the data in train and test
from sklearn.model_selection import train_test_split
training, testing = train_test_split(feature_set,test_size = 0.25, random_state=seed)

len(training),len(testing)


# ### Importing the SklearnClassifier in NLTK

# In[18]:


from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))


# ### Importing the Necessary Classifiers

# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))


# ### Building the VotingClassifier for Ensembel Modelling

# In[20]:


from sklearn.ensemble import VotingClassifier

names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))


# ### Making class label prediction for testing set

# In[21]:


txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)


# ### Printing a confusion matrix and a classification report

# In[22]:



print(classification_report(labels, prediction))

pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])

