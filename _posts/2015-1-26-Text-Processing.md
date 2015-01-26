---
layout: post
title: "How to Predict High Salary Job Postings Using Scikit-Learn Text Processing"
categories: python pandas
author: Rodolfo Soto
keywords: Text Processing Scikit-Learn
---

In the past couple of years the use of unstructured data for data mining and machine learning has increased exponentially. This is due thanks to the continues improvement of tools such as NLTK as well as Scikit-Learn, which have allowed to use text for prediction and classification in easier way. Today we will review Scikit-Learn's text feauture extraction package, the **tfidvectorizer** to be more precise.    



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
