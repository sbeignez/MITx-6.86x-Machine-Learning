# Unit 1. Lecture 2.


## 2.1. Objectives

Linear Classifiers and Perceptron Algorithm

* understand the concepts of
  * Feature vectors and labels
  * Training set and Test set
  * Classifier
  * Training error, Test error
  * and the Set of classifiers
* derive the mathematical presentation of linear classifiers
* understand the intuitive and formal definition of **linear separation**
* use the **perceptron algorithm** with and without offset

## 2.2. Review of Basic Concepts

* Feature vectors, labels
  * dimension of the vector = number of features
* Training set
* Classifier
* Training error
* Test error
* Generalization: the relation between the Tset and the Tset 

* The set of linear ..
* Linear separation
* Perceptron algorithm
  * a very simple online mistake driven algorithm that


## 2.3. Linear Classifiers Mathematically Revisited

Linear classifier
Decision boundary (hyper-plane)

Index and Parametize the LC

### Linear Through orign

$ \{ X : \theta_1 X_1+\theta_2 X_2 = 0 \}  $

$ \theta = \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix},\ and\ X = \begin{bmatrix} X_1 \\ X_2 \end{bmatrix} $

$ \{ X : \theta \cdot X = 0 \}  $

$ h(X; \theta) = sign (\theta \cdot X) $

multiple vector $\theta$ for the same decision boundary.  
The degree is the distance to the boundary

### Any linear (with offset)

$ \{ X : \theta \cdot X + \theta_o = 0 \}  $  
$ \theta_o $ is a scalar

2 paramters to define the classifier


### Exercices

Inner product of 2 vectors:

$
\begin{bmatrix} x_1 \\ .. \\ x_n \end{bmatrix} \cdot \begin{bmatrix} y_1 \\ .. \\ y_n \end{bmatrix} = x^Ty = \sum_{i=1}^n x_i y_i 
$

## 2.4. Linear Separation

For some cases, it's impossible to find a classifier that perfectly classify all the trainings points.

> **Linearly separarable**  
> A traning example set $S_n$ is linearly separable,
> * if it exists a linear classifier that correctly classifies all the training examples.
> * i.e. if it exists a parameter vector $\hat\theta$ and offset parameter $\hat\theta_0$, such that $y^{(i)} ( \hat\theta \cdot x^{(i)} + \hat\theta_0) >0$, for all $i = 1,..,n$


By definition: The linear classifier $h : X \mapsto \{ -1,0,1\}$, 

But conventionally, 0 is not a possible label. 0 is counted as an error.

$ y^{(i)} ( \theta \cdot x^{(i)}) > 0 $ when $y^{(i)}$ and $( \theta \cdot x^{(i)})$ are of the same sign  
  

  
## 2.5. The Perceptron Algorithm

Training error for linear classifier

Linear classifier: $ h() $ 

$ E_n(\theta,\theta_o) = 1/n * \sum_{i=1}^n( h(x^{(i)}) <> y^{(i)}) ..$

sum of mis-classified points

### Perceptron through orgin

* Initialize with $ \theta = 0 $
* Loop T times: For $ i = [1,T] $ 
  * For each data point: For $ i = [1,n] $ 
    * If: $ y^{(i)}(\theta \cdot x^{(i)}) \le 0 $
    * then update: $\theta^{(i)} = \theta^{(i-1)} + y^{(i)}x^{(i)} $ 

Notes:
* (T is the number of epoch)  
* (n is the size of the training set)
* Epoch: the number of complete passes through the training dataset.
* Batch: number of training samples to work through before the model’s internal parameters are updated.
* Both are Hyperparameter of the gradient descent algo.

### Perceptron with offset  

* if: $ y^{(i)}(\theta \cdot x^{(i)} + \theta_0) \le 0 $ 
* then: $\theta = \theta + y^{(i)}x^{(i)} $
* and $\theta_0 = \theta_0 + y^{(i)}$
