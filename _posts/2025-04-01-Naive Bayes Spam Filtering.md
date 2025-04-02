---
title: Naive Bayes Spam Filtering
categories: [ Bayes Theorem]
tags: [ Naive Bayes
] 
image: /images/NaiveBayes.png
---

# Naive Bayes Spam Filtering
The following is a step by step quide for spam fitering using Naive Bayes. Refer to the previous posts for a deeper explanation on Bayes Theorem and Naive Bayes. In spam filtering we want to find the likelihood of specific message being spam. But a message consists of multiple words. In order to find the combined probability of the words we first have to find the probability of each separate word being a spam word. This is sometimes called spaminess of a word. we can calculate it by using a special case of Bayes Theorem where the event is a binary variable.

![](/images/BTBINARY.png)

where,

P(S|W) is the probability that a message is a spam, knowing that a specific word is in it.

P(W|S) is the probability that the specific word appears in spam messages.

P(S) is the overall probability that any given message is spam.

P(W|H) is the probability that the specific word appears in ham messages.

P(H)  is the overall probability that any given message is ham.

We want our filter to be unbiased toward incoming email, so we are going to assume that the probabilities of receiving both spam and ham are equal to 50%. This allows us to simplify the formula to the following:

![](/images/SIMPLIFIEDFORMULA.png)


Now to the implementation.

**Key Steps in the Model:**

**Training Phase:**

Collect labeled emails (spam or not spam).

Compute the probability of each word occurring in spam and non-spam emails.

Estimate class prior probabilities (likelihood of an email being spam or non-spam).

**Classification Phase:**

Given a new email, extract words and compute the probability that the email belongs to either class (spam or not spam).

Use Bayes’ Theorem to compute the posterior probability:
![](/images/FG.png)

The email is classified as spam if 
𝑃
(
Spam
∣
Email
)
P(Spam∣Email) is higher than 
𝑃
(
Not Spam
∣
Email
)
P(Not Spam∣Email).

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
import time
import numpy as np # linear algebra
import pandas as pd # data processing
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


import os
```
Import data that contains spam and ham mails.

data1.head()
```python
Unnamed: 0                                               Body  Label
0           0  Subject: great part-time or summer job !\n \n ...      1
1           1  Subject: auto insurance rates too high ?\n \n ...      1
2           2  Subject: do want the best and economical hunti...      1
3           3  Subject: email 57 million people for $ 99\n \n...      1
4           4  Subject: do n't miss these !\n \n attention ! ...      1
```
data2.head()
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0.1</th>
      <th>Unnamed: 0</th>
      <th>Body</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2469</td>
      <td>2469</td>
      <td>Subject: stock promo mover : cwtd\n * * * urge...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5063</td>
      <td>5063</td>
      <td>Subject: are you listed in major search engine...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12564</td>
      <td>12564</td>
      <td>Subject: important information thu , 30 jun 20...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2796</td>
      <td>2796</td>
      <td>Subject: = ? utf - 8 ? q ? bask your life with...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1468</td>
      <td>1468</td>
      <td>Subject: " bidstogo " is places to go , things...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

data3.head()

In the above datasets spam's label is 1 and ham's labels is 0. Lets clean the data a bit
```python
data1.drop("Unnamed: 0",inplace=True,axis=1)
data2.drop(["Unnamed: 0","Unnamed: 0.1"],inplace=True,axis=1)
data3.drop("Unnamed: 0",inplace=True,axis=1)
```
Now we concatenate the datasets
```python
data = pd.concat([data1,data2,data3],axis=0)
data.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Body</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Subject: great part-time or summer job !\n \n ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Subject: auto insurance rates too high ?\n \n ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Subject: do want the best and economical hunti...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Subject: email 57 million people for $ 99\n \n...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Subject: do n't miss these !\n \n attention ! ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
Remove missing value

```python
data.dropna(inplace=True)
data.info()
```
In the dataset we have link that might cause problems when trying to predict.
```python
x = data["Body"]
x_clnd_link = [re.sub(r"http\S+", "", text) for text in x]

print(x_clnd_link[0])
```
We will use regex to remove digits because they don't have meaning.
```python
pattern = "[^a-zA-Z0-9]"
```
```python
x_cleaned = [re.sub(pattern," ",text) for text in x_clnd_link]
```
Next, we will lower the texts.
```python
x_lowered = [text.lower() for text in x_cleaned]
print(x_lowered[0])
```
Tokenizing

We will use a function in NLTK TO create a feature that shows whether the text includes the word or not, we need to split words into lists.
```python
x_tokenized = [nltk.word_tokenize(text) for text in x_lowered]
```
Lemmatizing and Removing Stopwords
```python
lemma = WordNetLemmatizer()

words = ["bats","removed","cheers","good","stopped","went","fired","cleaner","beers"]
for word in words:
    print(lemma.lemmatize(word),end=" ")
```
Next, we lemmatize our texts.
```python
x_lemmatized = [[lemma.lemmatize(word) for word in text] for text in x_tokenized]
```
Removing Stopwords

In natural languages there are words that not have a special meaning such as will, it is always a tense and such as and,or. In order to win from time and improve the model we should remove them.
```python
stopwords = nltk.corpus.stopwords.words("english")
x_prepared = [[word for word in text if word not in stopwords] for text in x_lemmatized]
```
We will then look at how many unique words we have in our dataset.
```python
len(np.unique([word for text in x_prepared for word in text]))
```
Bag of Words. 

Bag of Words is an easy approach to make sense of texts. In bag of words approach, each feature shows whether the text contains the word or not.
```python
vectorizer = CountVectorizer(max_features=20000)
x = vectorizer.fit_transform([" ".join(text) for text in x_prepared]).toarray()

x.shape
```
(10000, 20000)

Split data to train and test set
```python
x_train,x_test,y_train,y_test = train_test_split(x,np.asarray(data["Label"]),random_state=42,test_size=0.2)
x_train.shape
```
Naive Bayes Model

We will now fit our model and prepare our spam filter.
```python
start_time = time.time()
NB = GaussianNB()
NB.fit(x_train,y_train)
end_time = time.time()

print(round(end_time-start_time,2))
```
4.84

Accuracy and confusion matrix
```python
NB.score(x_test,y_test)
```
0.9125
```python
from sklearn.metrics import confusion_matrix
y_pred = NB.predict(x_test)

conf = confusion_matrix(y_pred=y_pred,y_true=y_test)
import seaborn
seaborn.heatmap(conf,annot=True,fmt=".1f",linewidths=1.5)
import matplotlib.pyplot as plt
plt.show()
```
![](/images/confusionmatrix.png)

A function to test the model

```python
def predict_mail(mail):
    
    model = pickle.load(open("model.pckl",mode="rb"))
    vectorizer = pickle.load(open("vectorizer.pckl",mode="rb"))
    
    lemma = WordNetLemmatizer()
    
    stopwords = nltk.corpus.stopwords.words('english')
    
    mail = re.sub(r"http\S+", "", mail)
    mail = re.sub("[^a-zA-Z0-9]"," ",mail)
    mail = mail.lower()
    mail = nltk.word_tokenize(mail)
    mail = [lemma.lemmatize(word) for word in mail]
    mail = [word for word in mail if word not in stopwords]
    mail = " ".join(mail)
    
    vector = vectorizer.transform([mail])
    decision = model.predict(vector.toarray())
    
    return decision[0]
```



The Naïve Bayes spam filter is a probabilistic model that classifies emails as spam or not spam using Bayes’ Theorem under the assumption that words occur independently. It learns from labeled emails by estimating the probability of words appearing in spam and non-spam messages. When a new email arrives, it calculates the likelihood of it being spam based on its words and assigns the category with the highest probability. The model is fast, simple, and interpretable, making it effective for real-time spam filtering. However, its independence assumption and inability to capture word order can limit accuracy. Despite these drawbacks, it remains a strong baseline for spam detection.