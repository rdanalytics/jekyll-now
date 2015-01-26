---
layout: post
title: "How to Predict High Salary Job Postings Using Scikit-Learn Text Processing"
categories: python pandas
author: Rodolfo Soto
keywords: Text Processing Scikit-Learn
---



In the past couple of years the use of unstructured data for data mining and machine learning has increased exponentially. This is due thanks to the continues improvement of tools such as NLTK, Scikit-Learn, Weka, etc. These applications have allowed to use text for prediction and classification problems, something that was considered a far fetched idea not so long ago. Today we will go over Scikit-Learn's text feauture extraction package the **tfidvectorizer** tool, and test it with a large data set of job posting which you can [download here](http://www.kaggle.com/c/job-salary-prediction/download/Train_rev1.zip). This dataset contains job postings that were scraped from the http://www.cv-library.co.uk/ website. All the numbers from the job descriptions have been erased and replaced by asterisks to allow the text mining tool to work to its full potential and to avoid any bias in the predictions.  


       



{% highlight python %}
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

$$
\begin{align*}
& {tf}(t,d) = 0.5 + \frac{0.5 \times \mathrm{f}(t, d)}{\max\{\mathrm{f}(w, d):w \in d\}}
\end{align*}
$$
