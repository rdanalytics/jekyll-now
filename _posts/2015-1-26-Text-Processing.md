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


In [2]:
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
{% endhighlight %}


As we can see above the 75th percentile for the salaries is set at £42,500.00, which is approximately $63,794.18.   

$$
\begin{align*}
& {tf}(t,d) = 0.5 + \frac{0.5 \times \mathrm{f}(t, d)}{\max\{\mathrm{f}(w, d):w \in d\}}
\end{align*}
$$
