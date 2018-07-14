# PE Program Report - Image Lab 2018 Summer 

## Goals

Our program is about 
* Recognizing the bone fracture using pytorch. 
* Understand basic knowledges of Machine Learning and Neural Network.
* Make ability to read papers about Deep-Learning in now days.
* Find my future plan in Image processing

### 1. Get used to pytorch
* We first design **autoencoder** of our own. We must produce meaningful results within 2 weeks. 

* My job is to make an encoder part of autoencoder.

***

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
  * ` torch.squeeze(tensor) ` deletes dimesion whose size is **1**.
  * ` torch.unsqueeze(tensor, dim) ` adds dimension to "dim" dimension with size 1.
 
* Initializing
To use initialiing method, you should import torch.nn.init (e.g. as init)
  * ` init.uniform(tensor, a, b) ` fills tensor with values drawn from uniform distribution from a(lb) to b(ub).
  * ` init.normal(tensor, std) ` fills tensor with values drawn from normal distribution where standard deviation is std.
  * ` init.constant(tensor, val) ` fills tensor with constant(val).
  

### 2. Arithmetic Operation
You can add, multiply, divide tensors. Subtraction is proceeded by adding negative value or operator **-**.

Broadcasting(do the same thing to all elemnets) is supported.


* Addition is just as same as matrix addition.
* Multiplication and Division is performed element by element, not like matrix multiplication.
* If you want to do matrix multiplication, you should use torch.mm(tensor1, tensor2).

  
### 3. Matrix Operation
Matrix multiplication is not performed by star operator. To do that, you should use another method.
    x1 = torch.FloatTensor(3,4)
    x2 = torch.FloatTensor(4,5)
    torch.mm(x1,x2)

This torch.mm() method performs matrix multiplication which results 3 X 5 matrix

* Dot
  * You can do dot product by using ` torch.dot(x1, x2) `

* Transpose
  * You can transpose tensor by using ` (tensor obj).t() ` method

* Eigen Vector
  *  ` torch.eig(x1, True) `

* Eigen Value
  * ` torch.eig(x1, False) `
  
***

### 16th, May
### 1. Tensor? Variable?
Basically, tensor == variable. Variable is just a wrapper of tensor. Tensors are the actual data.

You can easily auto-compute the gradient of **variable**.

* Creator property
  * e.g.) You create a variable A, then add 1 to get B. Now there's a link stored between A and B, in the **creator** property.

Variable contains some variables in it. 
* data : wrapped tensor(actual data)
* grad : gradient of the variable
* requires_grad : for fine grained exclusion of subgraphs from gradient computation, increase efficiency. Only if all inputs don’t require gradient, the output also won’t require it. 
  * Backward computation is never performed in the subgraphs, where all Tensors didn’t require gradients.
* volatile : makes whole graph not requiring gradient.

### 2. Gradient Calculation
What is Gradient?
  * Gradient is a multi-variable generalization of the derivative. While a **derivative** can be defined on functions of a **single** variable, for functions of **several** variables, the gradient takes its place. 
  * The gradient is a vector-valued function, as opposed to a derivative, which is scalar-valued.
  * Gradient results vector from scalar.

To calculate gradient, you should import **Variable** module from **torch.autograd** 

```python
    import torch
    from torch.autograd import Variable
    x=Variable(torch.FloatTensor(3,4),requires_grad=True)
    y=x**2 + 4*x
    z=2*y + 3
    
    gradient = torch.FloatTensor(3,4)
    z.backward(gradient)
    
    print(x.grad)
    y.grad, z.grad
```
Since y is made of x, and z is made of y, when we need to find the gradient of z, we should differentiate y and x too by **chain rule**.

This code calculates the gradient of x. **backward** method accumulates gradients in the leaves - you might need to zero them before calling it.
    
***

### 17th, May
### 1. Linear Regression
Procedure : Gen data -> model -> optimize -> train -> check param

* Required Libraries
```python 
    import numpy as np 
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.init as init
    from torch.autograd import Variable
```
* Generate Data
```python
    numOfData = 1000
    noise = init.normal(torch.FloatTensor(numOfData,1),std=0.2)
    x=init.uniform(torch.Tensor(numOfData,1),-10,10)
    y=2*x+3 #relation, the actual parameter values(answers; 2 and 3)
    y_noise = 2*(x+noise)+3 #same relation
```
* Model and Optimizer
```python
    model = nn.Linear(1,1) # bias is true (default), input size and ouput size of each sample is 1
    ouput = model(Variable(x)) # ouput 
    
    loss_func = nn.L1Loss() # Creates a criterion that measures the mean absolute value of the element-wise difference between input x and target y
    optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent with learning rate = 0.01
```
* Train
```python
    loss_array = []
    lable = Variable(y_noise)
    num_epoch = 1000 # number of loop
    for i in range(num_epoch):
        ouput = model(Variable(x))
        optimizer.zero_grad()
        
        loss = loss_func(ouput,label)
        loss.backward()
        optimzer.step()
        if i%10 == 0: # print loss per 10 loops
            print(loss)
        loss_arr.append(loss.data.numpy()[0])
```
***

### 18th, May
### 1. Visdom
Visdom broadcasts visualizations of plots, images, and text for yourself and your collaborators.

`pip install visdom` to setup visdom, `python -m visdom.sever` to start server

#### Test Code
```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.init as init
    from torch.autograd import Variable

    from visdom import Visdom
    viz = Visdom()

    num_data  = 1000
    num_epoch = 1000

    noise = init.normal(torch.FloatTensor(num_data,1),std=0.2)
    x = init.uniform(torch.Tensor(num_data,1),-10,10)
    y = 2*x+3
    y_noise = 2*(x+noise)+3

    input_data = torch.cat([x,y_noise],1)

    win=viz.scatter(
        X = input_data,
        opts=dict(
            xtickmin=-10,
            xtickmax=10,
            xtickstep=1,
            ytickmin=-20,
            ytickmax=20,
            ytickstep=1,
            markersymbol='dot',
            markersize=5,
            markercolor=np.random.randint(0, 255, num_data),
        ),
    )

    viz.updateTrace(
        X = x,
        Y = y,
        win=win,
    )

    model = nn.Linear(1,1)
    output = model(Variable(x))

    loss_func = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    loss_arr =[]
    label = Variable(y_noise)
    for i in range(num_epoch):
        output = model(Variable(x))
        optimizer.zero_grad()

        loss = loss_func(output,label)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(loss)
        loss_arr.append(loss.data.numpy()[0])

    param_list = list(model.parameters())
    print(param_list[0].data,param_list[1].data)

    win_2=viz.scatter(
        X = input_data,
        opts=dict(
            xtickmin=-10,
            xtickmax=10,
            xtickstep=1,
            ytickmin=-20,
            ytickmax=20,
            ytickstep=1,
            markersymbol='dot',
            markercolor=np.random.randint(0, 255, num_data),
            markersize=5,
        ),
    )

    viz.updateTrace(
        X = x,
        Y = output.data,
        win = win_2,
        opts=dict(
            xtickmin=-15,
            xtickmax=10,
            xtickstep=1,
            ytickmin=-300,
            ytickmax=200,
            ytickstep=1,
            markersymbol='dot',
        ),
    )

    x = np.reshape([i for i in range(num_epoch)],newshape=[num_epoch,1])
    loss_data = np.reshape(loss_arr,newshape=[num_epoch,1])

    win2=viz.line(
        X = x,
        Y = loss_data,
        opts=dict(
            xtickmin=0,
            xtickmax=num_epoch,
            xtickstep=1,
            ytickmin=0,
            ytickmax=20,
            ytickstep=1,
            markercolor=np.random.randint(0, 255, num_epoch),
        ),
    )
```
![ex_screenshot](./img/vizTest.PNG)


***

### 21st, May
### 1. CNN
#### What is CNN ? 
* CNN is abbreviation of Convolutional-Neural-Network. 
* Convolution : Combine two data into one datum.
* Neural Network : Such systems "learn" (i.e. progressively improve performance on) tasks by considering examples, generally without task-specific programming.
* Input image and Convolution kernel makes Featured image. 
* The point is that CNN makes this convolution kernel itself. Kernel does not have fixed values. Rather, its values are modified by conv net.
  
#### Test Code
```python
    import torch
    import torch.nn as nn
    import torch.utils as utils
    from torch.autograd import Variable
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    %matplotlib inline


    # Set Hyperparameters

    epoch = 100
    batch_size =16
    learning_rate = 0.001


    # Download Data

    mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
    mnist_test  = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
    
    
    # Check the datasets downloaded

   print(mnist_train.__len__())
    print(mnist_test.__len__())
    img1,label1 = mnist_train.__getitem__(0)
    img2,label2 = mnist_test.__getitem__(0)

    print(img1.size(), label1)
    print(img2.size(), label2)


    # Set Data Loader(input pipeline)

    train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=mnist_test,batch_size=batch_size,shuffle=True)
    
    
    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
    #                 padding=0, dilation=1, groups=1, bias=True)
    # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
    #                    return_indices=False, ceil_mode=False)
    # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,affine=True)
    # torch.nn.ReLU()
    # tensor.view(newshape)

    class CNN(nn.Module):
        def __init__(self):
            super(CNN,self).__init__()
            self.layer1 = nn.Sequential(
                            nn.Conv2d(1,16,5),   # batch x 16 x 24 x 24
                            nn.ReLU(),
                            nn.BatchNorm2d(16),
                            nn.Conv2d(16,32,5),  # batch x 32 x 20 x 20
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.MaxPool2d(2,2)   # batch x 32 x 10 x 10
            )
            self.layer2 = nn.Sequential(
                            nn.Conv2d(32,64,5),  # batch x 64 x 6 x 6
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.Conv2d(64,128,5),  # batch x 128 x 2 x 2
                            nn.ReLU()
            )
            self.fc = nn.Linear(2*2*128,10)

        def forward(self,x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(batch_size, -1)
            out = self.fc(out)
            return out

    cnn = CNN()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    
    # Train Model with train data
    # In order to use GPU you need to move all Variables and model by Module.cuda()

    for i in range(epoch):
        for j,[image,label] in enumerate(train_loader):
            image = Variable(image)
            label = Variable(label)

            optimizer.zero_grad()
            result = cnn.forward(image)
            loss = loss_func(result,label)
            loss.backward()
            optimizer.step()

            if j % 100 == 0:
                print(loss)
              
              
    # Test with test data
    # In order test, we need to change model mode to .eval()
    # and get the highest score label for accuracy

    cnn.eval()
    correct = 0
    total = 0

    for image,label in test_loader:
        image = Variable(image)
        result = cnn(image)

        _,pred = torch.max(result.data,1)

        total += label.size(0)
        correct += (pred == label).sum()

    print("Accuracy of Test Data: {}".format(correct/total))
```
***
### 22nd, May
### CNN
#### Analysis of CNN 

CNN is made up of several layers. There are three layers in addition to the layers for classification: convolutional layer, relu layer, and pooling layer.

**First**, the convolutional layer is the part that scans the image, simply repeating multiplication and addition. The kernel used at this time is called the convolution kernel. Multiply the value of the corresponding kernel element for each pixel, and add all of them to the value of the target pixel. This will result in a `d * d` matrix for the `d * d` input.
![ex_screenshot](./img/conv.PNG)

**Second**, the ReLU layer proceeds according to the following definition. `ReLU (x) = max (0, x)` That is, keeps its value for positive numbers and makes negative numbers zero.

**Third**, the pooling layer summarizes the `d * d` input as `k * k`. Where `d > k`. As a representative pooling method, max pooling is used to summarize the largest value of the input data.

After passing through these three layers, the input image can be said to pass through one filter. Pixel values ​​were modified by the kernel and reduced in size from `d * d` to `k * k`. This is called **convolve**.
![ex_screenshot](./img/cnn.PNG)

Now, if there are N of these filters, scan N times, relu application N times, and pooling N times, resulting in N results. But this is only half of CNN's role. CNN's ultimate goal is to extract the right features from the image. Therefore, in order **to extract the feature**, the **classification of the input image should proceed first**.

![ex_screenshot](./img/graph.PNG)
This is done at the Fully Connected Layer. The N (assumed to be feature) outputs obtained in the previous process are classified by passing through MLP. Then modify the values of the convolution kernel based on the output layer results.
Yes. The initial values of the convolution kernel are initially specified **randomly**. And, by learning map, we find the values necessary for classification by oneself. A person does not need to set a price in advance.

Let's say we have a photo of the dog, a label of Dog, a picture of the cat, and a label of Cat. After learning these pictures, if you give them the first picture of the dog that is not in the learning set, CNN can classify it as Dog.

***

### 23rd, May
### 1. Machine Learning Algorithms
#### How Machine Learns?


The learning algorithm of machine learning is largely divided into supervised learning and unsupervised learning.

It can be divided into Deep or Shallow according to the number of layers. Ranzato classifies a number of learning algorithms as follows.

![ex_screenshot](./img/ml_algos.PNG)

Learning algorithms generally include 'reinforcement learning' in addition to the two 'supervised learning' and 'unsupervised learning'.



### 2. Supervised Learning

Supervised learning is a way of learning that knowing the answer of given input. It is a method to make a model based on the data with the learning data and the correct label for it and to estimate the data of the new verification set.

To get the right learning outcomes, lots of learning data that has good quality should be prepared. Nowadays, because there is a database like ImageNet, it is easy to get data for learning, but it is rare that it contains the correct answer, so the painful process must be preceded by giving expectations one by one. 

Typically, pattern recognition belongs to supervised learning.


### 3. Unsupervised Learning


Unsupervised learning is a learning method in which a student finds patterns or features in data through algorithms.

Although it can be thought that it is convenient to learn by itself if we put only learning data, it is difficult to implement, and in the process of inferring the criterion for the feature itself, it is possible to return different result than expected. 

Typically, data mining belongs to unsupervised learning.


### 4. Reinforcement Learning

Reinforcement learning is an algorithm that rewards and punishes as if training a pet. 

This can be used when the relationship between input and output can not be clearly described, as opposed to superviesed learning. This method is effective when the number of cases is too large to judge right or wrong. 

For example, the AlphaGo is famous for its confrontation with Lee Sedol in Korea. 

These features can be useful for real-life problems, which are suitable for autonomous driving or strategic simulation problems.

***
### 24th, May
### 1. Softmax Classification
#### Multinomial classification


The multinomial classification is a classification that classifies the most probable class in several labels. There is nothing special about multinomial, but it's all about doing binary classification many times like an if statement in programming and calculating the probability that belongs to each class accordingly.

In the ordinary expression `Wx + b`, W was `1 * N`. For multiple binary classifications, you can write W in `K * N`.
Where N is the number of elements in the input vector and K is the number of classes to classify. For example, if W is 3 * 3, then three classes are distinguished and the input vector will be given as 3 * 1.


The result of Wx + b is hatched to y. This value is the predicted value, not the probability value yet. Adjust this value between 0 and 1 using the sigmoid function. After that, one-hot encoding is done through the arg max function to clearly indicate which one to select.

The cost function is obtained by using the probability value obtained in the previous step and the one-hot encoding value. The log function is used and the `-log(S(Yi))` value of the probability value of the class selected by one-hot encoding is used as cost.

Finally, the gradient decent method is used to find the minimum value and the learning proceeds. This part is too complicated, so it only understands the function call.

***


### 25th, May
### 1. AutoEncoder
#### What is AutoEncoder?

![ex_screenshot](./img/ae01.PNG)

The structure is similar to that of MLP, but their purpose is quite different.
MLP aims to classify the input vector into one of the given classes, but AE aims to approximate the output to the input.

AE is a kind of compression because the number of neurons in the hidden layer is generally smaller than that of the input layer. Also, since the output layer must be equal in number to the input layer and the number of neurons, decoding is performed from the hidden layer to the output layer.

When using AE to extract information from an input vector, if the number of neurons in the hidden layer is larger than the number of neurons in the input layer, the weight of AE is simply calculated immediately. If we make the weight of the remaining neurons zero, it is over. Therefore, the number of neurons in the hidden layer must be smaller than the number of neurons in the input layer. Putting a number of constraints on it gives you an excellent ability to reduce dimensions.

![ex_screenshot](./img/ae.PNG)

Hidden layers can be multiple layers like MLP. Therefore, AE usually has an hourglass shaped graph. As the bottleneck reaches the hidden layer, AE can express the features from the front more compactly.

***


### 28th, May
### 1. Basics of Deep-learning
#### 5-steps of learning

* Choose Proper Network

* Check Gradient 

* Parameter Initialization

* Parameter Optimization

* Prevent Overfitting



#### Choose Proper Network - non-linearity

To choose network, we should know the structures of each networks and method about acquiring non-linearity. 

Structures of networks will be handled in July.

Before that, we should know how to acquire non-linearity.

#### What is Non-linearity?

By using a linear function as an activation function, when the layer is stacked, the network can be expressed again as a single equation. This is no different from working without a hidden layer.

In this way, only very simple problems can be solved. classification is rarely divisible by a straight line. Therefore, performance is very bad when using linear functions. It is therefore essential to use non-linear functions.

Non-linear functions include the sigmoid function and the tanh function. The s-shaped function represents a value between 0 and 1 or -1 and 1.

### 2. Non-linearity Functions
#### Sigmoid

Sigmoid function is also called as logistic function. The equation is like this:

\sigma (x)=\frac { 1 }{ 1+{ e }^{ -x } } \\ \sigma '\left( x \right) =\sigma (x)(1-\sigma (x))

![ex_screenshot](./img/sigmoid.PNG)

We usually set threshold as 0.5, which means the neuron will be activated when `x > 0`.

But this function has a disadvantage that is fatal to Deep-Neural-Network. The graph that differentiates sigmoid function is as follows.

![ex_screenshot](./img/derivativeSigmoid.PNG)

As you can see, when input value is greater than 5 or less than -5, the result become very close to 0. And also, gradient of this function is always greater than 0.

#### Hyperbolic tangent

As mentioned above, the gradient of sigmoid function is positive for all integers. This slows down the speed of learning or makes it impossible to learn. 

![ex_screenshot](./img/weightedSum.PNG)

In process of back propagation, if we differentitate Loss for every w_i, all gradients are positive or all negative since every input is postive because of sigmoid function.

This phenomenon is lead to a problem shown below, which is slowing down the learning speed.

![ex_screenshot](./img/zigzag.PNG)

If we say blue arrow is an optimal way, network only can reach optimal point by zigzag way(red arrows).

Therefore, we decided to use a new activation function that is hyperbolic tangent. The equation is like this:

% <![CDATA[
\begin{align*}
tanh(x)&=2\sigma (2x)-1\\ &=\frac { { e }^{ x }-{ e }^{ -x } }{ { e }^{ x }+{ e }^{ -x } } 
\\tanh'\left( x \right) &=1-tanh^{ 2 }\left( x \right) 
\end{align*} %]]>

The function has same shape of sigmoid, but it is scaled and shifted. We can see that function is symmetric with respect to zero, unlike the sigmoid function. 

![ex_screenshot](./img/ht.PNG)

But this function has same problem that sigmoid has; The Gradient Vanishing Problem.

![ex_screenshot](./img/diffHT.PNG)

### 3. Problems
#### Gradient Vanishing/Exploding Problem

As the neural network becomes deeper and deeper, it faces the **Gradient Vanishing / Exploding Problem**. It is a phenomenon that, in the process of optimizing through the derivative using the SGD(Stochatstic Gradient Descent) method, when the input value is out of a certain range, the slope becomes close to 0, and as it passes through the hidden layers, it gradually converges to zero, that makes network can't learn as result can't affect parameters. This led a deep, cold winter of Neural Network for 20 years(1986~2006).

![ex_screenshot](./img/GVP.PNG)

So we use the function which is called ReLU or function called Maxout to solve this problem.
The use of ReLU solves the problem, but at the same time it is easier to differentiate and reduce computational complexity. Compared to sigmoid, ReLU converges about 6 times faster.

### 4. Solution
#### ReLU(Rectified Linear Unit)

Rectified Linear Unit is first invented by Geoffrey Hinton in 2006. He pointed that we used wrong type of non-linearity and suggested ReLU as solution. It's equation is like this:

f(x)=max(0,x)

![ex_screenshot](./img/relu.PNG)

This simple function solved gradient vanishing problem by making it's gradient as 1 for all positive x. It is easier than sigmoid or hyperbolic tangent to differentiate. This made NNs learn faster than before. 

![ex_screenshot](./img/reluIsBetter.PNG)

### 5. Next Step
#### Parameter Initialization

What happens if we initialize all the weights of the network to zero? Whatever the input is, if the weight is zero, then the same value is passed to the next layer as `k * 0 = 0` for all k.
If all nodes have the same value when back propagation, the weights are all updated equally. That is, even if the number of neurons is increased for hidden layers, the expression power of the network is limited.

Therefore, an initialization methodology emerges.

***

### 29th, May
### 1. Parameters
#### Parameter Initialization

To set initial values of the weights and bias of each layer are very important. Since the problems we try to solve by neural network are optimization problems in non-convex condition, it may not be possible to find the optimum point if the starting point is mis-caught. (networks stops at local minima)

![ex_screenshot](./img/localminima.PNG)


If the initial value of the parameter is set appropriately, it is effective also for gradient adjustment. If we initialize the weight vector W of the input layer to all 0s, the weight is zero at the first forward propagation, so the same value is transmitted to the second layer (bias value). In other words, the neural network can not operate properly because the same value is transmitted to all nodes.

In 2006, Professor Geoffrey Hinton realized that there was a problem with the parameter initialization method at the time of "A Fast Learning Algorithm for Deep Belief Nets" and proposed a new method called Restricted Boltzmann Machine. This paper introduces the concept of "pre-train".

#### Restricted Bolzmann Machine(Hinton et al. 2006)

![ex_screenshot](./img/RBM.PNG)

Weights are forwarded to the next layer for the value of x in the current layer(**forward**). This time, we pass backwards the value to the previous layer and weave it backward(**backward**). By doing this **forward** and **backward** repeatedly, we can find the weight that minimizes the difference between the first input x and the predicted x value x_hat .

Applying this between every layer will properly initialize the weights between all layers. This pre-trained (or fine-tuned) network does not take a long time for learning. Although we benefited from this learning speed, the RBM is difficult to implement because of its complicated structure. A paper published in 2010 suggests that similar results can be achieved without this complex initialization.

### 2. Better Initialization
#### Xavier/He Initialization (Glorot and Bengio, 2010) (He,Zhang, Ren and Sun, 2015)

The xavier initialization released in 2010 is incredibly simple, but at the same time it shows incredibly good performance. Xavier initialization selects a random number between the input and output values and divides it by the square root of the input value.

//Xavier Initialization (LeCun Initialization)

W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 1}{ { n }_{ in } }

//Glorot Initialization

W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 2 }{ { n }_{ in }+{ n }_{ out } }

He initialization which applied xavier initialization, uses the square root of the input value divided by half to generate a wider range of random numbers than the xavier.

//He Initialization

W\sim Uniform({ n }_{ in },{ n }_{ out })\\ Var(W)=\frac { 2 }{ { n }_{ in } }

Here is the code :

```python
#xavier initialization
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)

#He initialization
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in/2)
```

***

### 30th, May
### Pretrained Models
#### Using ResNet50 by Pytorch 

```python
# How to use Pretrained models with PyTorch
# Simple Classifier using resnet50

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable

batch_size = 3
learning_rate =0.0002
epoch = 50

resnet = models.resnet50(pretrained=True)

# Input pipeline from a folder containing multiple folders of images
# we can check the classes, class_to_idx, and filename with idx

img_dir = "./images"
img_data = dset.ImageFolder(img_dir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ]))

print(img_data.classes)
print(img_data.class_to_idx)
print(img_data.imgs)

# After we get the list of images, we can turn the list into batches of images
# with torch.utils.data.DataLoader()

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)

for img,label in img_batch:
    print(img.size())
    print(label)

# test of the result coming from resnet model

img = Variable(img)
print(resnet(img))

# we have 2 categorical variables so 1000 -> 500 -> 2
# test the whole process

model = nn.Sequential(
            nn.Linear(1000,500),
            nn.ReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500,2),
            nn.ReLU()
            )

print(model(resnet(img)))

# define loss func & optimizer

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

# In order to train with GPU, we need to put the model and variables
# by doing .cuda()

resnet.cuda()
model.cuda()

for i in range(epoch):
    for img,label in img_batch:
        img = Variable(img).cuda()
        label = Variable(label).cuda()

        optimizer.zero_grad()
        output = model(resnet(img))
        loss = loss_func(output,label)
        loss.backward()
        optimizer.step()

    if i % 10 ==0:
        print(loss)


# Check Accuracy of the trained model# Check
# Need to get used to using .cuda() and .data

model.eval()
correct = 0
total = 0

for img, label in img_batch:
    img = Variable(img).cuda()
    label = Variable(label).cuda()

    output = model(resnet(img))
    _, pred = torch.max(output.data, 1)

    total += label.size(0)
    correct += (pred == label.data).sum()

print("Accuracy: {}".format(correct / total))
```

***

### 31st, May
### 1. Parameters
#### Parameter Optimization

The parameter optimization is basically based on the gradient descent method. Most papers are learning neural networks through this method. There are various detailed methods in the descending method. Among them, Batch Gradient Descent, Stochastic Gradient Descent, and Mini-batch Gradient Descent are discussed.

For the gradient parameter **θ**, **dL/dθ** is the gradient of θ against Loss, and **η** is the leaning rate. In other words, the following equation means that **θ** should be updated little by **η** in the **opposite direction** of θ to Loss.

`θ←θ−η∂L/∂θ`

If this function is continued, if the Loss function is convex, the network approaches critical point and the learning ends.


### 2. Optimization Methods
#### Batch Gradient Descent

The gradient of each parameter for the loss of the entire learning data is obtained at once and all parameters are updated once during one epoch. It is very slow and requires a lot of memory. However, in the case of convex, it is guaranteed that the optimal solution can be obtained. If non-convex, converge to local minimum.

```python
for i in range(nb_epochs):
    params_grad = evaluate_gradient(loss_function, data, params)
    params = params - learning_rate * params_grad
```

#### Stochastic Gradient Descent

The sequence of training data is randomly mixed, and then the loss and gradient are obtained for each individual record. This updates the learning parameters little by little. Updating is performed as many times as the number of learning data for one epoch. It is much faster than BGD and the converge result is consistent with BGD. Of course, learning rate should be reduced.

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
    	params_grad = evaluate_gradient(loss_function, example, params)
    	params = params - learning_rate * params_grad
```

#### Mini-batch Gradient Descent

This is the same way as SGD, except that it learns batch_size instead of individual records. It is used in many experiments in a stable manner compared to SGD. Because the data is in batches, you can use matrix operations, which leads to the advantage of being able to take advantage of powerful libraries that are available on the market.

```python
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
    	params_grad = evaluate_gradient(loss_function, batch, params)
    	params = params - learning_rate * params_grad
```

***

### 25th, June
### 1. Overfitting
#### Prevent Overfitting

Overfitting is when the model becomes too adaptive to the learning data and generalization performance drops. Because machine learning is aimed at general-purpose performance, it must have the ability to judge correctly, even if it is given the new data in addition to the learning data. 

There's some techniques to prevent overarching in neural network learning.

#### Reduce Model Size

This is the simplest way to prevent overfitting. Reduces the number of parameters you need to learn, such as layers and neurons, to avoid overfitting. The more parameters there are, the more tendency to learn the characteristics of given learning data excessively.

#### Early Stopping

Prevent overfitting by stop learning early. Stop learning before NN being conquered by learning data.

#### Decay Weights

It is said that overfitting is often caused by large learning parameters. We use **Weight decay** to prevent this. It is a technique that gives a corresponding large penalty if the value of a learning parameter is large. We mainly use **L2 Regularization**.

#### L2 Regularization

L2 Regularization uses new loss function that is sumation of square of parameters in original loss function. 

The equation is below :

\\ { L }_{ new }={ L }_{ old }+\frac { \lambda  }{ 2 } { (w }_{ 1 }^{ 2 }+{ w }_{ 2 }^{ 2 }+...+{ w }_{ n }^{ 2 }) %]]>

Where 1/2 is taken into account for differential convenience, and λ is a user-specified hyperparameter that determines the strength of the penalty. This technique has the effect of restricting weights with large values and spreading the weight values as much as possible.

#### Dropout

Dropout is a way of learning by turning off some neurons. At the time of learning, the neurons to be deleted are randomly turned off, and all neurons are used when testing.

![ex_screenshot](./img/dropout.jpeg)

***

### 26th, June
### 1. Learning Rate
#### Decaying learning rate

The learining rate plays an important role in the learning process as shown in the figure below. If it is too large, it can not be diverted and can not be learned properly. If it is small, learning time becomes too long.

![ex_screenshot](./img/LR.PNG)

While learning can be done with fixed learning rates from the beginning to the end of learning, the more models you learn, the more likely the model will converge to the optimal point, so you might want to fine-tune the parameters by lowering the learning rate at the end.

**η** is the learning rate, **t** is the number of steps, and **k** is a user-specified hyperparameter.

#### Step Decay

This technique reduces the learning rate by a certain amount every step. Each 5epoch is reduced by half or every 20epoch by 1/10, but it is difficult to apply uniformly depending on data or network structure.

#### Exponential Decay

η=η_0e^{-kt}

#### 1/t Decay

η=η_0/(1+kt)

