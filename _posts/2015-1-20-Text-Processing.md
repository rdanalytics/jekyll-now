---
layout: post
title: "How to Predict High Salary Job Postings using Scikit-Learn Text Processing"
categories: python pandas
author: Rodolfo Soto
keywords: Text Processing Scikit-Learn

---


First level header
==================

Second level header
------

   Other first level header
=





{% highlight Python %}
In [1]: new_svm_predicted
Out [1]: array([0, 1, 0])
{% endhighlight %}


{% highlight python %}

In [1]: import pandas as pd
In [2]: data = pd.read_csv('reed.csv')
In [3]: data.columns
Out[3]: Index([u'salary_min', u'description', u'title', u'salary_max', u'applications', u'page_number', u'location', u'published', u'link', u'found', u'id'], dtype='object')

{% endhighlight %}


|---
| Default aligned  |   Left aligned |   Center aligned |   Right aligned
|-|:-|:-:|-:
| First body part  |   Second cell |   Third cell |   fourth cell
| Second line |  foo |  **strong** |  baz
|  Third line |quux | baz | bar
|---
| Second body
| 2 line
|===
| Footer row






| header 1 | header 2 |
| -------- | -------- |
| cell 1   | cell 2   |
| cell 3   | cell 4   |


$$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$
