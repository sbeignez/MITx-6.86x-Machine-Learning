Unit 1. Lecture 2.


# 1. Objectives

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

# 2. Review of Basic Concepts

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


# 3. Linear Classifiers Mathematically Revisited

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

# 4. Linear Separation

For some cases, it's impossible to find a classifier that perfectly classify all the trainings points.

> **Linearly separarable**  
> A traning example set $S_n$ is linearly separable, if there exist a parameter vector $\theta$ and offset parameter $\theta_o$.. such as ... >0 for all i = 1,..,n


By definition: The linear classifier $h : X \mapsto \{ -1,0,1\}$, 

But conventionally, 0 is not a possible label

$ y^{(i)} ( \theta \cdot x^{(i)}) > 0 $ when $y^{(i)}$ and $( \theta \cdot x^{(i)})$ are of the same sign 


# 5. The Perceptron Algorithm

Training error for linear classifier

Linear classifier: $ h() $ 

$ E_n(\theta,\theta_o) = 1/n * \sum_{i=1}^n( h(x^{(i)}) <> y^{(i)}) ..$

sum of mis-classified points

## Perceptron through orgin

* Start with $ \theta = 0 $
* For $ i = [1,T] $
* For $ i = [1,n] $

$ if: \ y^{(i)}(\theta \cdot x^{(i)}) <= 0 \\
then: \theta = \theta + y^{(i)}x^{(i)}
$ 

## Perceptron with offset


$ if: \ y^{(i)}(\theta \cdot x^{(i)} + \theta_0) <= 0 \\  
then: \theta = \theta + y^{(i)}x^{(i)} \\
\theta_0 = \theta_0 + y^{(i)}
$ 
