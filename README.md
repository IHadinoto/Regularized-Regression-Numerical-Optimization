# COMP9417 - Machine Learning

# Homework 1: Regularized Regression & Numerical

# Optimization

**Introduction** In this homework we will explore some algorithms forgradientbased optimization. These
algorithms have been crucial to the development of machine learning in the last few decades. The most
famous example is the backpropagation algorithm used in deep learning, which is in fact just an application
of a simple algorithm known as (stochastic) gradient descent. We will first implement gradient descent
from scratch on a deterministic problem (no data), and then extend our implementation to solve a real
world regression problem.
**Points Allocation** There are a total of 28 marks.

- Question 1 a): 5 marks
- Question 1 b): 1 mark
- Question 1 c): 1 mark
- Question 1 d): 4 marks
- Question 1 e): 1 mark
- Question 1 f): 1 mark
- Question 1 g): 1 mark
- Question 1 h): 2 marks
- Question 1 i): 5 marks
- Question 1 j): 5 marks
- Question 1 k): 2 marks

**What to Submit**

- A **single PDF** file which contains solutions to each question. For each question, provide your solution
    in the form of text and requested plots. For some questions you will be requested to provide screen
    shots of code used to generate your answer — only include these when they are explicitly asked for.
- **.py file(s) containing all code you used for the project, which should be provided in a separate .zip**
    **file.** This code must match the code provided in the report.
- You may be deducted points for not following these instructions.

## 1


- You may be deducted points for poorly presented/formatted work. Please be neat and make your
    solutions clear. Start each question on a new page if necessary.
- You **cannot** submit a Jupyter notebook; this will receive a mark of zero. This does not stop you from
    developing your code in a notebook and then copying it into a .py file though, or using a tool such as
    **nbconvert** or similar.
- We will set up a Moodle forum for questions about this homework. Please read the existing questions
    before posting new questions. Please do some basic research online before posting questions. Please
    only post clarification questions. Any questions deemed to befishingfor answers will be ignored
    and/or deleted.
- Please check Moodle announcements for updates to this spec. It is your responsibility to check for
    announcements about the spec.
- Please complete your homework on your own, do not discuss your solution with other people in the
    course. General discussion of the problems is fine, but you must write out your own solution and
    acknowledge if you discussed any of the problems in your submission (including their name(s) and
    zID).
- As usual, we monitor all online forums such as Chegg, StackExchange, etc. Posting homework ques-
    tions on these site is equivalent to plagiarism and will result in a case of academic misconduct.

**When and Where to Submit**

- Due date: Week 4, Monday **March 7th** , 2022 by **5pm**. Please note that the forum will not be actively
    monitored on weekends.
- Late submissions will incur a penalty of 5% per day **from the maximum achievable grade**. For ex-
    ample, if you achieve a grade of 80/100 but you submitted 3 days late, then your final grade will be
    80 − 3 ×5 = 65. Submissions that are more than 5 days late will receive a mark of zero.
- Submission must be done through Moodle, no exceptions.


**Question 1. Gradient Based Optimization**
The general framework for a gradient method for finding a minimizer of a functionf :Rn→Ris
defined by

```
x(k+1)=x(k)−αk∇f(xk), k= 0, 1 , 2 ,..., (1)
```
```
whereαk> 0 is known as the step size, or learning rate. Consider the following simple example of
minimizing the functiong(x) = 2
```
## √

```
x^3 + 1. We first note thatg′(x) = 3x^2 (x^3 + 1)−^1 /^2. We then need to
choose a starting value ofx, sayx(0)= 1. Let’s also take the step size to be constant,αk=α= 0. 1. Then
we have the following iterations:
```
```
x(1)=x(0)− 0. 1 ×3(x(0))^2 ((x(0))^3 + 1)−^1 /^2 = 0. 7878679656440357
x(2)=x(1)− 0. 1 ×3(x(1))^2 ((x(1))^3 + 1)−^1 /^2 = 0. 6352617090300827
x(3)= 0. 5272505146487477
..
.
```
```
and this continues until we terminate the algorithm (as a quick exercise for your own benefit, code
this up and compare it to the true minimum of the function which isx∗=− 1 ). This idea works for
functions that have vector valued inputs, which is often the case in machine learning. For example,
when we minimize a loss function we do so with respect to a weight vector,β. When we take the step-
size to be constant at each iteration, this algorithm is known as gradient descent. For the entirety of this
question, do not use any existing implementations of gradient methods, doing so will result in an
automatic mark of zero for the entire question.
(a) Consider the following optimisation problem:
```
```
min
x∈Rn
f(x),
```
```
where
```
```
f(x) =
```
## 1

## 2

```
‖Ax−b‖^22 +
```
```
γ
2
```
```
‖x‖^22 ,
```
```
and whereA∈Rm×n,b∈Rmare defined as
```
## A=

## 

## 

## 1 2 1 − 1

## − 1 1 0 2

## 0 − 1 − 2 1

## 

```
, b=
```
## 

## 

## 3

## 2

## − 2

## 

## ,

```
andγis a positive constant. Run gradient descent onfusing a step size ofα= 0. 1 andγ= 0. 2 and
starting point ofx(0)= (1, 1 , 1 ,1). You will need to terminate the algorithm when the following
condition is met: ‖∇f(x(k))‖ 2 < 0. 001. In your answer, clearly write down the version of the
gradient steps (1) for this problem. Also, print out the first 5 and last 5 values ofx(k), clearly
indicating the value ofk, in the form:
```
```
k= 0, x(k)= [1, 1 , 1 ,1]
k= 1, x(k)=···
k= 2, x(k)=···
..
.
```

```
What to submit: an equation outlining the explicit gradient update, a print out of the first 5 (k= 5inclusive)
and last 5 rows of your iterations. Use the round function to round your numbers to 4 decimal places. Include
a screen shot of any code used for this section and a copy of your python code in solutions.py.
(b) In the previous part, we used the termination condition‖∇f(x(k))‖ 2 < 0. 001. What do you think
this condition means in terms of convergence of the algorithm to a minimizer off? How would
making the right hand side smaller (say 0. 0001 ) instead, change the output of the algorithm? Ex-
plain.
What to submit: some commentary.
(c) Although we used gradient descent to find the minimizer offin part (a), we can also use calculus
to solve the problem directly. Show that the minimizer offis
```
```
xˆ= (ATA+γI)−^1 ATb,
```
```
whereIis the 4 × 4 identity matrix. What is the exact numerical value ofˆxfor the problem in (a)
and how does it compare to your result from gradient descent?
What to submit: your working out.
```
(d) In this question we’ll investigate the step-size parameter in more depth. Run the gradient descent
algorithm 9 times, each time with a different step-size, ranging over:

```
α∈{ 0. 0000001 , 0. 000001 , 0. 00001 , 0. 0001 , 0. 001 , 0. 01 , 0. 02 , 0. 1 , 0. 15 },
```
```
and withγ= 0. 2. For each choice ofα, plot the difference:‖x(k)−ˆx‖ 2 over all stepsk= 1, 2 ,...,
and wherexˆis the true solution from part (c). Use the same termination condition as before
(‖∇f(x(k))‖ 2 < 0. 001 ) with an additional termination constraint ofk < 10000 (the algorithm termi-
nates after 10,000 steps at most). Present your results as a 3 × 3 grid plot (one subplot for eachα),
and on each subplot also plot the liney= 0. 001 in red. Comment on your results. What effect does
changing the step-size have? What would you expect as you take the step-size to be increasingly
large (α= 10for example).What to submit: a single plot, some commentary. Include a screen shot of any
code used for this section and a copy of your python code in solutions.py.
In the next few parts, we will use gradient methods explored above to solve a real machine learning
problem. Consider the CarSeats data provided inCarSeats.csv. It contains 400 observations
with each observation describing child car seats for sale at one of 400 stores. The features in the
data set are outlined below:
```
- Sales: Unit sales (in thousands) at each location
- CompPrice: Price charged by competitor at each location
- Income: Local income level (in thousands of dollars)
- Advertising: advertising budget (in thousands of dollars)
- Population: local population size (in thousands)
- Price: price charged by store at each site
- ShelveLoc: A categorical variable with Bad, Good and Medium describing the quality of the
    shelf location of the car seat
- Age: Average age of the local population
- Education: Education level at each location
- Urban A categorical variable with levels No and Yes to describe whether the store is in an
    urban location or in a rural one


- US: A categorical variable with levels No and Yes to describe whether the store is in the US or
    not.
The target variable is Sales. The goal is to learn to predict the amount of Sales as a function of a
subset of the above features. We will do so by running Ridge Regression (Ridge) which is defined
as follows

```
βˆRidge= arg min
β
```
## 1

```
n
```
```
‖y−Xβ‖^22 +φ‖β‖^22 ,
```
```
whereβ∈Rp,X∈Rn×p,y∈Rnandφ > 0.
(e) We first need to preprocess the data. Remove all categorical features. Then use
sklearn.preprocessing.StandardScalerto standardize the remaining features. Print out
the mean and variance of each of the standardized features. Next, center the target variable (sub-
tract its mean). Finally, create a training set from the first half of the resulting dataset, and a test set
from the remaining half and call these objects Xtrain, Xtest, Ytrain and Ytest. Print out the first
and last rows of each of these.
What to submit: a print out of the means and variances of features, a print out of the first and last rows of
the 4 requested objects, and some commentary. Include a screen shot of any code used for this section and a
copy of your python code in solutions.py.
(f) Explain why standardization of the features is necessary for ridge regression. What issues might
you run into if you used Ridge without first standardizing?
What to submit: some commentary.
```
(g) It should be obvious that a closed form expression forβˆRidgeexists. Write down the closed form
expression, and compute the exact numerical value on the training dataset withφ= 0. 5.
What to submit: Your working, and a print out of the value of the ridge solution based on (Xtrain, Ytrain).
Include a screen shot of any code used for this section and a copy of your python code in solutions.py.
We will now solve the ridge problem but using numerical techniques. As noted in the lectures,
there are a few variants of gradient descent that we will briefly outline here. Recall that in gradient
descent our update rule is

```
β(k+1)=β(k)−αk∇L(β(k)), k= 0, 1 , 2 ,...,
```
```
whereL(β)is the loss function that we are trying to minimize. In machine learning, it is often the
case that the loss function takes the form
```
```
L(β) =
```
## 1

```
n
```
```
∑n
```
```
i=
```
```
Li(β),
```
```
i.e. the loss is an average ofnfunctions that we have lablledLi. It then follows that the gradient is
also an average of the form
```
```
∇L(β) =
```
## 1

```
n
```
```
∑n
```
```
i=
```
```
∇Li(β).
```
```
We can now define some popular variants of gradient descent.
(i) Gradient Descent (GD) (also referred to as batch gradient descent): here we use the full gradi-
ent, as in we take the average over allnterms, so our update rule is:
```
```
β(k+1)=β(k)−
```
```
αk
n
```
```
∑n
```
```
i=
```
```
∇Li(β(k)), k= 0, 1 , 2 ,....
```

```
(ii) Stochastic Gradient Descent (SGD): instead of considering allnterms, at thek-th step we
choose an indexikrandomly from{ 1 ,...,n}, and update
```
```
β(k+1)=β(k)−αk∇Lik(β(k)), k= 0, 1 , 2 ,....
```
```
Here, we are approximating the full gradient∇L(β)using∇Lik(β).
(iii) Mini-Batch Gradient Descent: GD (using all terms) and SGD (using a single term) represents
the two possible extremes. In mini-batch GD we choose batches of size 1 < B < nrandomly
at each step, call their indices{ik 1 ,ik 2 ,...,ikB}, and then we update
```
```
β(k+1)=β(k)−
```
```
αk
B
```
## ∑B

```
j=
```
```
∇Lij(β(k)), k= 0, 1 , 2 ,...,
```
```
so we are still approximating the full gradient but using more than a single element as is done
in SGD.
```
(h) The ridge regression loss is

```
L(β) =
```
## 1

```
n
```
```
‖y−Xβ‖^22 +φ‖β‖^22.
```
```
Show that we can write
```
```
L(β) =
```
## 1

```
n
```
```
∑n
```
```
i=
```
```
Li(β),
```
```
and identify the functionsL 1 (β),...,Ln(β). Further, show that
```
```
∇Li(β) =− 2 xi(yi−xTiβ) + 2φβ, i= 1,...,n.
```
```
What to submit: your working.
(i) In this question, you will implement (batch) GD from scratch to solve the ridge regression problem.
Use an initial estimateβ(0)= 1p(the vector of ones), andφ= 0. 5 and run the algorithm for 1000
epochs (an epoch is one pass over the entire data, so a single GD step). Repeat this for the following
step sizes:
```
```
α∈{ 0. 000001 , 0. 000005 , 0. 00001 , 0. 00005 , 0. 0001 , 0. 0005 , 0. 001 , 0. 005 , 0. 01 }
```
```
To monitor the performance of the algorithm, we will plot the value
```
```
∆(k)=L(β(k))−L(βˆ),
```
```
whereβˆis the true (closed form) ridge solution derived earlier. Present your results in a 3 × 3
grid plot, with each subplot showing the progression of∆(k)when running GD with a specific
step-size. State which step-size you think is best and letβ(K)denote the estimator achieved when
running GD with that choice of step size. Report the following:
(i) The train MSE:n^1 ‖ytrain−Xtrainβ(K)‖^22
(ii) The test MSE:^1 n‖ytest−Xtestβ(K)‖^22
What to submit: a single plot, the train and test MSE requested. Include a screen shot of any code used for
this section and a copy of your python code in solutions.py.
```

```
(j) We will now implement SGD from scratch to solve the ridge regression problem. Use an initial
estimateβ(0)= 1p(the vector of ones) andφ= 0. 5 and run the algorithm for 5 epochs (this means
a total of 5 nupdates ofβ, wherenis the size of the training set). Repeat this for the following step
sizes:
```
```
α∈{ 0. 000001 , 0. 000005 , 0. 00001 , 0. 00005 , 0. 0001 , 0. 0005 , 0. 001 , 0. 006 , 0. 02 }
```
```
Present an analogous 3 × 3 grid plot as in the previous question. Instead of choosing an index
randomly at each step of SGD, we will cycle through the observations in the order they are stored
in Xtrain to ensure consistent results. Report the best step-size choice and the corresponding
train and test MSEs. In some cases you might observe that the value of∆(k)jumps up and down,
and this is not something you would have seen using batch GD. Why do you think this might be
happening?
What to submit: a single plot, the train and test MSE requested and some commentary. Include a screen
shot of any code used for this section and a copy of your python code in solutions.py.
```
(k) Based on your GD and SGD results, which algorithm do you prefer? When is it a better idea to use
GD? When is it a better idea to use SGD?


