# Unit 1. Lecture 1

## 1. Unit 1 Overview

## 2. Objectives

## 3. What is Machine Learning?

## 4. Introduction to Supervised Learning

## 5. A Concrete Example of a Supervised Learning Task

## 6. Introduction to Classifiers: Let's bring in some geometry!

> Training data can be graphically depicted on a (hyper)plane. Classifiers are mappings that take feature vectors as input and produce labels as output. A common kind of classifier is the linear classifier, which linearly divides space(the (hyper)plane where training data lies) into two.

> Given a point $x$ in the space, the classifier $h$ outputs $h(x)=1$ or $h(x)=-1$, depending on where the point $x$ exists in among the two linearly divided spaces.

* Feature vectors
* Labels
* Training set $S_n$
  * Pair of $(X^i, y^i)$ in $ \Bbb{R}^2,[-1,1] $

### Linear Classifier

### Training error

$\\E_n(h) $  
Truth value
= 1 if error
= 0 otherwise

$ 1/n * \sum_{i=1}^n \llbracket h(x^{(i)}) \not= y^{(i)} \rrbracket$

Training error of classifier
Test error (of classifier)

Generalization

Hypothesis space: the set of possible classifier



## 7. Different Kinds of Supervised Learning: classification vs regression

> A supervised learning task is one
where you have specified the correct behavior.

Discret vs. Continuous

* Classification (Binary)
* Multi-way classification
* Regression
* Structure classification


* Supervised Learning
* Un-
* Semi-
* Active learning
  - Deciding which examples are needed to learn
* Transfer learning
  - apply a method (learn on a problem) on another problem
* Re-inforcement learning
  - optimise the outcome of the action


The role of training | test set
A classifier
A set of classifier
Error
Generalization

## 8.