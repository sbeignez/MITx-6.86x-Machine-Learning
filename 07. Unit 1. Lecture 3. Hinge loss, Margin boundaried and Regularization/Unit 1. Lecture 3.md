# Unit 1. Lecture 3. Hinge loss, Margin boundaries and Regularization

## 3.1. Objective

Hinge loss, Margin boundaries, and Regularization

* understand the need for maximizing the margin
* pose linear classification as an optimization problem
* understand hinge loss, margin boundaries and regularization

## 3.2. Introduction

Large margin classifier

Margins boundaries (positive and negative)

Regularization ??

Loss 

Objective function = min (loss + )


## 3.3. Margin Boundary

### Decision boundary

> The decision boundary is the set of points  which satisfy $ \theta \cdot x + \theta_0 = 0 $

### Margin boundary  
> The Margin Boundary is the set of points  which satisfy $ \theta \cdot x + \theta_o = +1\ or -1 $

+1 on the side teh vector points, -1 on the other side.

$ \theta \cdot x + \theta_0 $ is the signed distance.


Margins boundaries are 2 lines, equidistant from the decision boundary

 > So, the distance from the decision boundary to the margin boundary is $ 1 \over ||\theta||$ .



For all points, correctly classfied, on the margin boundary: $ y_i*(\theta \cdot x_i + \theta_0) = 1 $

As $ ||\theta|| $ increase,  $ 1 \over ||\theta||$ decrease and the distance from the decision boundaries to the margins boundaries decrease too.

Illustration: #TODO  
Large theta, small theta





## 4. Hinge Loss and Objective function

### Loss function

Loss function

> In supervised machine learning algorithms, we want to minimize the error for each training example. This is done using some optimization strategies like gradient descent. And this error comes from the loss function.

Cost function and loss function 

> A **Loss function** (or error funciton) is for a single training example.  
> A **Cost function**, on the other hand, is the average loss over the entire training dataset.

Type of loss functions
* Regression Loss Functions
  * Squared Error Loss
  * Absolute Error Loss
  * Huber Loss
* Binary Classification Loss Functions
  * Binary Cross Entropy Loss
  * Hinge Loss
* Multi-Class Classification Loss Functions
  * Multi-class Cross entropy Loss
  * KL-Divergence
  * etc..


### Hinge Loss function

$ L = max(0,1-y*f(x))$


### Regularization


### Objective function

> Objective function = average loss + regularization

$ J(\theta, \theta_0) = [ {1 \over n} \sum_{i=1}^n Loss] + [ ]$



Notes:
https://www.analyticsvidhya.com/blog/2017/09/understaing-support-vector-machine-example-code/