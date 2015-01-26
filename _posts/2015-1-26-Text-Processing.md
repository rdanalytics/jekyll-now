---
layout: post
title: "How to Predict High Salary Job Postings Using Scikit-Learn Text Processing"
categories: python pandas
author: Rodolfo Soto
keywords: Text Processing Scikit-Learn
---



In the past couple of years the use of unstructured data for data mining and machine learning has increased exponentially. This is due thanks to the continues improvement of tools such as NLTK, Scikit-Learn, Weka, etc. These applications have allowed to use text for prediction and classification problems, something that was considered a far fetched idea not so long ago. Today we will go over Scikit-Learn's text feauture extraction package the **tfidvectorizer** tool, and test it with a large data set of job posting which you can [download here](http://www.kaggle.com/c/job-salary-prediction/download/Train_rev1.zip). This dataset contains job postings that were scraped from the http://www.cv-library.co.uk/ website. All the numbers from the job descriptions have been erased and replaced by asterisks to allow the text mining tool to work to its full potential and to avoid any bias in the predictions.  




For this exercise we will use our job posting dataset to attempt to predict high salary job postings. In order to achieve this we will first dummy code our response variable, which is the normalized salary for each job. To dummy code the dependent variable we agreed to set the cutoff between a high salary job and the rest of the job postings as the 75th percentile, anything above this number will be given a one and anything below it will be given a zero.   

       
{% highlight python %}
In [1]:
import random
import pandas
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from operator import itemgetter
from sklearn.metrics import classification_report

io = pandas.read_csv('Train_rev1.csv',sep=',',usecols=(2,10))
io.describe()

Out [1]:
|-------------+-------------------|
|	      |	SalaryNormalized  |
|-------------|:------------------|
|	count |	244768.000000     |
|	mean  |	34122.577576      |
|	std   |	17640.543124      |
|	min   |	5000.000000       |
|	25%   |	21500.000000      |
|	50%   |	30000.000000      |
|	75%   |	42500.000000      |
|	max   |	200000.000000     |
|-------------+-------------------|

{% endhighlight %}




As we can see above the 75th percentile for the salaries is set at £42,500.00, which is approximately $63,794.18. Now we need to convert our pandas data frame into numpy array and then split our data into a training and a testing set. After performing these steps we will then dummy code our response variable to ones and zeroes respectively.   




{% highlight python %}
In [2]:
data = [np.array(x) for x in io.values]

random.shuffle(data)
size = int(len(data) * 0.6)
test_set, train_set = data[size:], data[:size]
train_set = np.array(train_set)
test_set = np.array(test_set)
x = train_set[:,1]
Sal75=np.percentile(x,75)
y = test_set[:,1]
Test75=np.percentile(y,75)

for i in range(len(train_set[:,1])):
    if train_set[i,1]>=Sal75:
        train_set[i,1]=1
    else:
        train_set[i,1]=0

for i in range(len(test_set[:,1])):
    if test_set[i,1]>=Test75:
        test_set[i,1]=1
    else:
        test_set[i,1]=0
train_setT = [tuple(x) for x in train_set]
test_setT = [tuple(x) for x in test_set]


train_set = np.array([''.join(el[0]) for el in train_setT])
test_set = np.array([''.join(el[0]) for el in test_setT])

y_train = np.array([el[1] for el in train_setT])
y_test = np.array([el[1] for el in test_setT])
{% endhighlight %}




After successfully shaping the data to our needs we will now use Sciki-Learn **tfidvectorizer**. This class converts a collection of raw documents to a matrix of TF-IDF features. TF-IDF stands for the "Term Frequency" or the the number of times that term t occurs in document d, and "Inverse Document Frequency" which divides the total number of documents by the number of documents containing the term t, and then taking the logarithm of that division. 

###TF

$$
\begin{align*}
& {tf}(t,d) = 0.5 + \frac{0.5 \times \mathrm{f}(t, d)}{\max\{\mathrm{f}(w, d):w \in d\}}
\end{align*}
$$

###IDF

$$
\begin{align*}
& {idf}(t, D) =  \log \frac{N}{|\{d \in D: t \in d\}|}
\end{align*}
$$
