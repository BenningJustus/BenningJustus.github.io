---
layout: post
title: Density-Based Novelty Detection
date: 2018-11-12 00:00:00 +0900
published: false
# Version 0.1, Update Date before release, Author: Justus Aaron Benning
---

## Introduction and terminology
Novelty detection is a term describing different machine learning processes that try to teach a program to distinguish unusual or irregular measurements from regular data. These **outliers** are suspected to originate from different mechanisms than the usual data (in contrast to statistical noise that occurs due to environmental effects or measurement errors).

Novelty detection differs from other classifiers in regard to the data that is to be analyzed: Classifiers such as the support vector machine try to separate rather balanced data sets. The number of examples for each class has to be sufficiently high. When there is a **minority** class, of which there is only a handful of data points, novelty detection algorithms tend to produce better results. The decision process regarding the correct application could be modeled as follows:
<!-- Picture of application flow below (maybe replace /w own ppt design) -->
<p style="text-align:center;"><img src="/images/ND-Flowchart.png" alt="ND" height="250"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---
This post focuses on a certain field of novelty detection algorithms, namely the **density-based** approach. Generally speaking, these algorithms search for outliers by looking for distribution patterns in the data points and subsequently labeling the data points that do not fit the pattern as outliers.

## Gaussian Density Estimation (Unigaussian Method)
The first of the three code examples covers the most basic approach of density-based novelty detection. In the most simple form of density estimation, we assume that the data is distributed over a **single** Gaussian function. Based on this assumption, we can calculate the maximum likelihood estimators for the mean and the standard deviation of the data points. In a further assumption, we define every data point that has **low probability** according to the density function to be an **outlier**.

### The code

First, the libraries required for our program are imported. In this case, we use ```pyplot``` and ```seaborn``` for plotting purposes, ```numpy``` for our linear algebra calculations and and the ```stats``` module from ```scipy``` to facilitate handling our stochastic functions. The code is written in Python 3 and should run in any updated, solved environment with some additional scientific Python libraries installed. Additionally, when using Jupyter, ```matplotlib``` should be configured to the inline layout.
```python
# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
# Layout
sns.set_style("white")
%matplotlib inline
```

### Example of a single modal Gaussian distribution
Before we go deeper into the machine learning process of the topic, we take a look at a standard Gaussian distribution, as well as how to create and visualize it with our imported libraries: In step one, we create the x-values of 1000 theoretical samples using the ```linspace``` function. Step two is using the ```stats.norm``` function to create the y-values for each sample: For each instance of ```x```, we "measure" a normally distributed continuous random variable and save the value in ```y```.
```python
# Return evenly spaced numbers over a specified interval.
x = np.linspace(start=-10, stop=10, num=1000)
# Create a normal continuous random variable:
# The location (loc) keyword specifies the mean.
# The scale (scale) keyword specifies the standard deviation.
y = stats.norm.pdf(x, loc=0, scale=1.5)
# Plot it!
plt.plot(x, y)
```
The last line plots the result via ```pyplot```.
<p style="text-align:center;"><img src="/images/Gaussian-Bell.png" alt="Gauss" height="300"></p>

### Implementation of dataset
For the data we use an altered excerpt of the Iris data set. The 150 data points are taken and a part them are defined to be outliers. The other part is labeled as target samples. In the original dataset, four features were measured from each sample. Thus, we have a 150 by 4 input matrix, as well as a 150-entry output vector. An excerpt of the data is shown in the table below.
<!-- Excerpt of csv-table below (folded) -->
<style type="text/css">
.tg  {width: 50%;margin-left: auto;margin-right: auto;}
.tg  {border-collapse:collapse;border-spacing:0;border-color:#ccc;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;border-color:#ccc;color:#333;background-color:#fff;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;border-color:#ccc;color:#333;background-color:#f0f0f0;}
.tg .tg-21ub{background-color:#286594;color:#f0f0f0;text-align:left;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-21ub">sepallength</th>
    <th class="tg-21ub">sepalwidth</th>
    <th class="tg-21ub">petallength</th>
    <th class="tg-21ub">petalwidth</th>
    <th class="tg-21ub">label</th>
  </tr>
  <tr>
    <td class="tg-0pky">5.1</td>
    <td class="tg-0pky">3.5</td>
    <td class="tg-0pky">1.4</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">4.9</td>
    <td class="tg-0pky">3</td>
    <td class="tg-0pky">1.4</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">4.7</td>
    <td class="tg-0pky">3.2</td>
    <td class="tg-0pky">1.3</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">4.6</td>
    <td class="tg-0pky">3.1</td>
    <td class="tg-0pky">1.5</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">3.6</td>
    <td class="tg-0pky">1.4</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">5.4</td>
    <td class="tg-0pky">3.9</td>
    <td class="tg-0pky">1.7</td>
    <td class="tg-0pky">0.4</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">4.6</td>
    <td class="tg-0pky">3.4</td>
    <td class="tg-0pky">1.4</td>
    <td class="tg-0pky">0.3</td>
    <td class="tg-0pky">outlier</td>
  </tr>
  <tr>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">3.4</td>
    <td class="tg-0pky">1.5</td>
    <td class="tg-0pky">0.2</td>
    <td class="tg-0pky">outlier</td>
  </tr>
</table>
---

*Source: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.*
[Download from UCI machine learning repository](http://archive.ics.uci.edu/ml/datasets/Iris)

---
### Visualization
In order to understand better what density-based novelty detection does, we need to visualize our data. To do that, we first import the ```pandas``` data handling library, as it works well together with ```seaborn```. We let ```pandas``` read our CSV-file  (change the path if necessary) and store it under the name ```df```.
```python
import pandas as pd
df = pd.read_csv("../2.Data/1_iris_virginica.csv")
# Returns the first few lines of the table (see above)
df.head(n=5)
```
As it is impossible to visualize 4-dimensional data, we only choose the first column of the dataset right now. An alternative way to achieve this is with the command ```y = pd.DataFrame.as_matrix(df[['sepallength']])```, used in the original code from the tutorial. The function  ```.as_matrix()``` however, is deprecating, and will likely not be compiled in future versions.
We use a histogram to visualize the distribution of our variables. Because we do not want an estimate of the data's density yet, we switch off the kernel density estimate in the last argument of the function.
```python
y = df[['sepallength']].values
# These lines are just to show "y" in the Jupyter notebook.
print(y.shape)
print(y[0:5])
# Plot it!
sns.distplot(y, bins=20, kde = False)

```
<p style="text-align:center;"><img src="/images/Iris-Hist-Pure.png" alt="Hist1" height="250"></p>

Interestingly, ```.distplot()``` has an optional argument that can do exactly what we are trying to achieve: Fit a distribution based on the maximum likelihood estimators over the data. But this is cutting it short, and thus is only included for completeness' sake.
```python
# Fit a single-mode-gaussian over the dataset
sns.distplot(y, fit=stats.norm, bins=20, kde=False,)
```

<p style="text-align:center;"><img src="/images/Iris-Hist-Gauss1.png" alt="Hist2" height="250"></p>


### Estimating the Gaussian density function by ourselves
In order to understand our approach on a basic level, we make our own class for Gaussian probability density functions. This lets us create instances of Gaussian functions that can return their respective parameters to us, as well as calculate the probability for every point we input.
```python
class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, data):
        u = (data - self.mu) / abs(self.sigma)
        y = (1 / (sqrt(2 * pi) * abs(self.sigma))) * exp(-u * u / 2)
        return y
```

Now we create an instance of our function with the two parameters being the maximum likelihood estimators for mean and standard deviation. Luckily, ```numpy``` has functions to carry out this task for us.
```python
best = Gaussian(np.mean(y), np.std(y))
print("best mu=" , best.mu)
print("best sigma=", best.sigma)
```

The output gives us ```best mu= 5.8433[...]``` and ```best sigma= 0.8253[...]```. With these two parameters a Gaussian density function is well-defined, and we can let it calculate the respective y-values to the 900 points we create using the ```.linspace()``` function once again. We overlap the graphs and get a similar image to the plot above.
```python
x = np.linspace(3,9,200)
g_single = stats.norm(best.mu, best.sigma).pdf(x)
sns.distplot(y, bins=20, kde = False, norm_hist= True)
plt.plot(x,g_single, label = 'Single Gaussian')
plt.legend()
```
<p style="text-align:center;"><img src="/images/Iris-Hist-Gauss2.png" alt="Hist2" height="250"></p>


### Identifying the outliers
We now want to use the density function to identify outliers in the data. Like stated above, we can use the probability density as a measure: If the data point lies out of bounds of a pre-defined density region, we reject it. The image below illustrates the cutoff process:

<p style="text-align:center;"><img src="/images/Gauss-Cutoff.png" alt="Hist2" height="250"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

In our code, the process would look as follows:
```python
n = 0
b = 0
for i in range(0,y.shape[0]):
    if (stats.norm(best.mu, best.sigma).pdf(y[i])) >0.05 and (stats.norm(best.mu, best.sigma).pdf(y[i])) < 0.995:
        print(y[i],"= normal")
        n=n+1
    else:
        print(y[i],"=abnormal")
        b=b+1

print("normal=",n)
print("abnormal=",b)
```
The integers ```n``` and ```b``` are counters summing up the number of target and outlier samples. Note that in the ```if``` statement, we have to account for the symmetry of the Gaussian function by comparing the values to *p* and *1 - p*. The cutoff value used here is 0.05 and 0.995 respectively.

The output generated looks as follows:
```python
[...]
[7.2] = normal
[7.4] = normal
[7.9] =abnormal
[6.4] = normal
[6.3] = normal
[6.1] = normal
[7.7] =abnormal
[6.3] = normal
[6.4] = normal
[6.] = normal
[...]
normal= 145
abnormal= 5
```
Using this method, we would identify 5 instances as novel data; however, this method makes a lot of assumptions. To improve on this result we will now look at more sophisticated methods.

## Mixture of Gaussian Density Estimation (MoG)
The aforementioned assumptions (unimodal and convex distribution) are often too strong and do not hold up in the case of data points based on multiple underlying mechanisms. The MoG novelty detection is a way to remedy these strong assumptions: We allow a multi-modal distribution by estimating a linear combination of Gaussian density functions. This approach comes with its own set of challenges, such as the question "How many functions should we combine" (which is essentially choosing a hyper-parameter) and finding the correct defining features of our density functions.

<p style="text-align:center;"><img src="/images/MoG-Image.png" alt="Hist2" height="250"></p>

---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---
### Imports
Our imports are very similar to the ones before, the most important differences being ```scipy.linalg``` for matrix operations and another visualization tool from ```matplotlib``` called ```Ellipse```, whose use becomes clear when we plot our results.
```python
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
import scipy as sc
from scipy import random, linalg, stats, special
```

### Data Frame
This time we use all four independent columns from our dataset as input, meaning ```y.shape``` will tell us that we have a 150 by 4 matrix. This time, the ```.as_matrix()``` function is used to generate our matrix, but as stated in the "Data Frame" chapter above, in future versions, the statement ```.values``` should be used.
```python
df = pd.read_csv("../2.Data/1_iris_virginica.csv")
df.head(n=5)
y = pd.DataFrame.as_matrix(df[['sepallength','sepalwidth ','petallength','petalwidth ']])
y.shape
print(y[0:5])
```
The ```print```-statement gives us the first five lines of the ```ndarray``` that we just created:
```python
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
```

### Visualization
To get a better feel of our data, we visualize our input once again. For practicality reasons we only use the first two columns of our matrix. We map them on a scatter plot using ```matplotlib``` and mark the axes as "Property 1" and "Property 2".
```python
plotsize = 8
sizeMean =10
text_size = 16
axis_font = {'fontname':'Arial', 'size':'24'}
Title_font= {'fontname':'Arial', 'size':'28'}
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(y[:,0], y[:,1], 'k.', markersize= sizeMean)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Property 2', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('Classes based on the first two properties', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize, plotsize)
plt.show()
```
<p style="text-align:center;"><img src="/images/MoG-Scatter1.png" alt="Hist2" height="400"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

Even though there is only two dimensions of our four-dimensional data, we can already make a qualitative judgement about the distribution: The data points seem to arrange in clusters, and in this example we can discern at least two distinct groups. This supports our approach of mapping multiple convex density functions over the data.

### Assessing the input for our calculations
As our following operations span multiple dimensions and density functions, we can expect some linear algebra calculations for the remainder of the program. Keeping that in mind, it is useful to assess the dimensions of our input. This is done using the ```ndarray.shape``` method of ```y```.

Furthermore we need to make a decision about the number of density functions that we want to combine. In this model, the decision is to use six Gaussian density functions.
```python
# Number of instances (= 150)
NObjects = y.shape[0]
# Number of independent variables (= 4)
NProperties = y.shape[1]
# Hyper-parameter
NClasses = 6   
```
As with all hyper-parameters, this is a judgement call based on experience and a basic knowledge of the data. For example, look at the graphic below (taken from a different dataset). Given only the histogram, most people would intuitively choose to set ```NClasses``` to five for a first attempt. Given the right performance indicators for our model, we could assess this decision later (and maybe correct it).

<p style="text-align:center;"><img src="/images/MoG-Hist1.png" alt="Hist2" height="300"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

### Expectation Maximization Algorithm (Introduction)
By introducing multiple density functions we will gain the ability to analyze more complex data. However, we also introduced a challenge: there is no information about the assignment of our instances to the respective density functions in our dataset. That is, we do not know which data points to consider when estimating a certain density function. Which point "belongs" to which cluster? And how do we "weigh" each cluster in our linear combination of densities?
Mathematically speaking, we introduced so called "hidden variables" that we have to account for in our calculations.

For this kind of stochastic problem, we can use the Expectation-Maximization algorithm (EM algorithm). We will introduce weights for each of our six density functions and repeat a simple two-step algorithm until we are satisfied with the result. The formulae used in each iteration of the algorithm look like this:

<p style="text-align:center;"><img src="/images/MoG-EM1.png" alt="Hist2" height="300"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

In the **E-Step**, we calculate "responsibilities", that is, probabilities that take into consideration the weight of our density functions. To normalize the values, we divide it by the sum over all classes. Now we have specific values for each data point that give an impression how likely it is that our measurement is part of (or "responsible" for) a density function.

In the **M-Step**, the parameters of the density functions are improved with the new information gained by the E-Step: For multiple Gaussians, a single datapoint might contribute more to curve A rather than curve B. How much "more" it contributes is expressed by the responsibilities. Thus, if we integrate these values into our MLE calculations, we get estimates that account for the importance of each point for the curve.

With these new and improved parameters, we can again calculate responsibilities and repeat the E- and M-Step until we no longer improve significantly.

---
*Source 1: Expectation Maximization with Gaussian Mixture Models: http://aishack.in/tutorials/expectation-maximization-gaussian-mixture-model-mixtures/*
*Source 2: Gaussian mixture models and the EM algorithm, Ramesh Sridharan, https://people.csail.mit.edu/rameshvs/content/gmm-em.pdf*

---


### What do mean and covariance look like?
First, we create a 6 by 4 ```ndarray``` for storing our mean values. We need a matrix, as we want to store six mean vectors (for our six density functions), each with the dimension of the input space, which is four.
Following this logic, ```initCov``` should store six covariance matrices, each containing the entries for four variables and their relations. Thus, we need a 4 by 4 by 6 covariance matrix.
```python
initMu = np.empty([NClasses, NProperties])
initCov = np.empty([NProperties, NProperties, NClasses])
print("initMu size =" ,initMu.shape)
print("initCov size =", initCov.shape)
```

### Initialize the covariance (and mean)
For initialization, it would be useful to have a function at hand that can quickly create different instances of covariance matrices. Mathematically speaking, we want to create matrices that are:

- Quadratic.
- At least positive-semi definite.
- Symmetric.

The function ```numpy.random.rand()``` will give us a good starting point, as it creates a matrix with non-negative entries. The symmetry is achieved by the line ```M = 0.5*(M + M.T)```. Finally we can arbitrarily raise our standard deviation with the parameter ```sd```.
```python
def PosSymDefMatrix(n,sd):
    M = np.matrix(np.random.rand(n,n))
    M = 0.5*(M + M.T)
    M = M + sd*np.eye(n)
    return M
```
The feasible entries of the mean vector for each curve will not be negative, as we only have positive numbers in our dataset. Furthermore, they will not exceed the maximum value of said variable. To create vectors fulfilling these constraints, we multiply the maximum of our instances for every variable (```np.amax(y, axis=0)```) by a random vector with 4 entries between 0 and 1 (```np.random.random(NProperties)```). Doing this for every class yields a 4 by 6 matrix.

In ```Cov``` we store a 4 by 4 by 6 matrix, which will be our building block for the initial covariance matrix ```initCov```. Because of the ```.rand()```-function in our matrix generation function, the covariance for every curve would be different. In this case however, we want the same covariance matrix for every curve, and we want the entries to be between 2 and 3. The reasons for this are of empirical and practical nature (just like the choice of six classes). Other initial values could be found by pre-clustering and the k-means Algorithm.

The code for this is straightforward: Iteration of ```np.mean(np.array(Cov), axis=0)``` yields six identical 4 by 4 matrices. In the process, we add a "bias" of ```+ 2```.

```python
Cov = [PosSymDefMatrix(NProperties,i) for i in range(0,NClasses)]
for j in range(NClasses):
    initMu[j, :] = np.random.random(NProperties) * np.amax(y, axis=0)
    initCov[:, :, j] = np.mean(np.array(Cov), axis=0) + 2

print("initMu=" ,initMu) # average
print("initCov=" ,initCov) # sigma
print("Cov=", Cov)
```

### Initialize weight vector
The weight vector is what defines the linear combination of the single functions. The sum of its entries must be 1 to yield another viable probability density function. As we do not know (yet) how to combine the functions, it makes sense to initialize the entries equally to 1/n-th of the classes, so in our case 1/6th.

```python
#Initialize weight vector values:
theta = np.repeat(1.0/NClasses,NClasses)
print ('Weight for each class/curve (theta) '+str(NClasses))
print (theta)
# Weight vector: Uniform 6 entry vector
initW = theta
```

### E-Step
Quick reminder: For multiple Gaussians, a single datapoint might contribute more to curve A rather than curve B. How much "more" it contributes is expressed by the responsibilities. In the E-step we want to calculate our responsibilities (in the code they are called ```r_ij```).

<p style="text-align:center;"><img src="/images/MoG-EM2.png" alt="Hist2" height="150"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

The matrix ```r_ij``` has 150 rows for every data point (150 instances, iterated over by ```y.shape[0]```) and six columns for the curves (also called "classes", iterated over with ```mu.shape[0]```). To effectively deal with the sum in the denominator of the E-Step, we split the calculations into the column vectors. We do this by introducing the intermediate variable ```r_ij_Sumj```. The calculations are completely analogous to the formula above, using the weight ```w[jClass]``` and probability densities for each curve when calculating the entries. In the last step, we normalize the column entries and store them with ```r_ij[Object, jClass] = r_ij_Sumj[jClass] / np.sum(r_ij_Sumj)```.

```python
def EStep(y, w, mu, cov):
    r_ij = np.zeros((y.shape[0], mu.shape[0]))

    for Object in range(y.shape[0]):

        r_ij_Sumj = np.zeros(mu.shape[0])

        for jClass in range(mu.shape[0]):
            r_ij_Sumj[jClass] = w[jClass] * sc.stats.multivariate_normal.pdf(y[Object, :], mu[jClass, :],cov[:, :, jClass])

        for jClass in range(r_ij_Sumj.shape[0]):
            r_ij[Object, jClass] = r_ij_Sumj[jClass] / np.sum(r_ij_Sumj)

    return r_ij
```

### M-Step
The first formula of the M-Step defines the new entries of the weight vector. The operation boils down to "the higher the total responsibility for the curve, the higher the new weight". By this logic, curves containing a larger number of points are weighed stronger in the final linear combination.

<p style="text-align:center;"><img src="/images/MoG-EM3.png" alt="Hist2" height="150"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

```python
# Implementation of Maximization Step
def MStep(r, y, mu, cov):
    N = y.shape[0]
    # Normalize the weights
    w_j = np.sum(r, axis=0) / N
```

- Annotations
- numpy.outer(a, b, out=None) = Compute the outer product of two vectors.
- np.outer(r[Object, :], y[Object, :]) creates a 4 by 6 matrix for every "Object". Every instance vector is multiplied the 6 different class-specific responsibilities.
- ```Allmu_j``` stores these 150 matrices in a 3-dimensional ndarray
- When we sum over the instance axis of these products and divide by the total responsibilities, we get the new 6 mean vectors.

- Analog mit der kovarianz , nur eine dimension mehr.


<p style="text-align:center;"><img src="/images/MoG-EM4.png" alt="Hist2" height="130"></p>
---

*Source: Kang, Pilsung: “Novelty Detection.” Business Analytics. Korea University, Fall 2018. Lecture.*

---

```python
    Allmu_j = np.zeros((N, mu.shape[0], mu.shape[1]))
    Allcov_j = np.zeros((N, cov.shape[0], cov.shape[1], cov.shape[2]))

    # mean
    for Object in range(N):
        Allmu_j[Object, :, :] = np.outer(r[Object, :], y[Object, :])

    mu_j = np.zeros((mu.shape[0], mu.shape[1]))

    for j in range(cov.shape[2]):
        mu_j[j, :] = (1 / np.sum(r, axis=0)[j]) * np.sum(Allmu_j, axis=0)[j, :]

    # sd
    for Object in range(N):
        for j in range(cov.shape[2]):
            Allcov_j[Object, :, :, j] = r[Object, j] * np.outer((y[Object, :] - mu_j[j, :]),
                                                                (y[Object, :] - mu_j[j, :]))

    cov_j = np.zeros((cov.shape[0], cov.shape[1], cov.shape[2]))

    for j in range(cov.shape[2]):
        cov_j[:, :, j] = (1 / np.sum(r, axis=0)[j]) * np.sum(Allcov_j, axis=0)[:, :, j]

    return w_j, mu_j, cov_j
```

### Parameters for Algorithm
Annotation Text
```python
# implement EM algorithm
Inititeration = 100
EMiteration = 40
lookLH = 20
```

### Implementation
Annotation Text
```python
for init in range(Inititeration):

    # starting values
    initMu = np.empty([NClasses, NProperties])
    for j in range(NClasses):
        initMu[j, :] = np.random.random(NProperties) * np.amax(y, axis=0)

    r_n = EStep(y, initW, initMu, initCov)
    w_n, mu_n, cov_n = MStep(r_n, y, initMu, initCov)

    if init == 0:
        logLH = -1000000000000

    for i in range(EMiteration):

        # E step
        r_n = EStep(y, w_n, mu_n, cov_n)

        # M step
        w_n, mu_n, sigma_n = MStep(r_n, y, mu_n, cov_n)

        # log likelihood를 계산
        logLall = np.zeros((y.shape[0]))

        for Object in range(y.shape[0]):

            LH = np.zeros(NClasses)

            for jClass in range(NClasses):
                LH[jClass] = w_n[jClass] * sc.stats.multivariate_normal.pdf(y[Object, :], mu_n[jClass, :],
                                                                            cov_n[:, :, jClass])

            logLall[Object] = np.log(np.sum(LH))

        logL = np.sum(logLall)

        if i > EMiteration - lookLH:
            print(logL)

    if logL > logLH:
        logLH = logL
        print('found larger: ', logLH)
        w_p = w_n
        mu_p = mu_n
        sigma_p = sigma_n
        r_p = r_n
```

### Result
Annotation Text
```python
print("mu=", mu_n)
print("sigma=",cov_n)
```

### Cutoff
Annotation Text
```python
# 판정
mul_pdf =np.zeros(NClasses)
tot=0
for i in range(0, y.shape[0]):
    n = 0
    b = 0
    for jClass in range(0,NClasses):
        mul_pdf[jClass] = sc.stats.multivariate_normal.pdf(y[i, :], mu_p[jClass, :],sigma_p[:, :, jClass])
        if mul_pdf[jClass] > 0.025 and mul_pdf[jClass] < 0.975:
            n=n+1
        else:
            b =b+1
    if n==0:
        tot = tot+1
        print("abnormal =", y[i,:])
print("abnormal count=", tot)
print("normal count =" ,y.shape[0]-tot)
```

### Plot
Annotation Text
```python
plotsize = 8
sizeMean = 10
text_size = 16
axis_font = {'fontname': 'Arial', 'size': '24'}
Title_font = {'fontname': 'Arial', 'size': '28'}
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.plot(y[:, 0], y[:, 1], 'k.', markersize=sizeMean)

for i in range(NClasses):
    # the sd with ellipses
    # central point of the error ellipse
    pos = [mu_p[i, 0], mu_p[i, 1]]

    # for the angle we need the eigenvectors of the covariance matrix
    w, ve = np.linalg.eig(sigma_p[0:2, 0:2, i])

    # We pick the largest eigen value
    order = w.argsort()[::-1]
    w = w[order]
    ve = ve[:, order]

    # we compute the angle towards the eigen vector with the largest eigen value
    thetaO = np.degrees(np.arctan(ve[1, 0] / ve[0, 0]))

    # Compute the width and height of the ellipse based on the eigen values (ie the length of the vectors)
    width, height = 2 * np.sqrt(w)

    # making the ellipse
    ellip = Ellipse(xy=pos, width=width, height=height, angle=thetaO)
    ellip.set_alpha(0.5)
    ellip.set_facecolor(color[i])

    ax.add_artist(ellip)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontname('Arial')
    label.set_fontsize(text_size)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_ylabel('Property 2', **axis_font)
ax.set_xlabel('Property 1', **axis_font)
ax.set_title('The inferred classes based on the first two properties', y=1.08, **Title_font)
ax.figure.set_size_inches(plotsize, plotsize)

plt.show()
```
<p style="text-align:center;"><img src="/images/MoG-FinalPlot.png" alt="Hist2" height="450"></p>
