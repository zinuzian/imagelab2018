# PE Program Report - Image Lab 2018 Summer 

## Goals

Our project is about recognizing the bone fracture using pytorch. 

### 1. Get used to pytorch
we first design **autoencoder** of our own. We must produce meaningful results within 2 weeks. 

My job is to make an encoder part of autoencoder.
# Daily Report
### 14th, May
### 1. To Create Tensor
To create a random matrix that has "dynamic" dimension, so called **tensor**, we can use rand function, which needs **two parameters**. 

And we assign this return value(matrix) to the variable.

<pre><code>import torch

randomMat = torch.rand(3,4)
</code></pre>

### 2. Various Functions
Generating tensor can be subdivided into several functions with various roles.

* torch.randn(a,b)
  * This method creates random matrix with normal distribution with size (a,b).
  
* torch.randperm(n)
  * This method creates random matrix with permutation of 0~n.

* torch.zeros(a,b)
  * This method creates matrix that is filled with 0s and size (a,b).
  
* torch.ones(a,b)
  * Same as zeros but filled with 1s.
  
* torch.arange(start, end, step=1)
  * Greater than or equal to start and less than end, step by step, makes a **list** of numbers. Default step is 1.
  
### 3. Data Types
Tensors are represented as a list of numbers enclosed in square brackets. 

* torch.FloatTensor(size | list)
  * This method creates float type tensor that has given size or list. 
* torch.LongTensor(size | list)
  * This method creates long type tensor that has given size of list. 
  
We can change Numpy to Tensor, vice versa.
* x2 = torch.from_numpy(x1) is for **numpu -> tensor**
* x3 = x2.numpy() is for **numpy <- tensor**

***

### 15th, May
### 1. Tensor Operations
Parameter dim means row when it is set to 0, column when it is set to 1. 

> This is not a correct sentence. "dim" means dimension of tensor since tensor is not just 2-D array.  

* Indexing
  * We can do indexing by torch.index_select(inputTensor, dim, index(may be list)) method.
  
* Masking
  * We can get masked tensor by torch.masked_select(inputTensor, mask(list)) method. This method returns masked tensor.
  
* Joining
  * We can concatenate two tensors by torch.cat([tns1, tns2],dim), If dim == 0, second one would be concatenated into row, otherwise, into column.
  
* Stacking
  * We can use stack of tensors by torch.stack(sequence, dim(=new dim)). This results an increase of dimension of tensor.
  
* Slicing
  * If you need a part of tensor, you can use torch.chunk(tensor, numOfChunks, dim). If dim = 0, tensor will be sliced by row, otherwise, to column.
  * Since result is multiple tensors, we can store each of them in different variables by positioning multiple variables to the left of the assignment.
  * **split()** method can do the same but its result is little bit different. We can think it as quotient and remainder.
  
* Squeezing
  

  
