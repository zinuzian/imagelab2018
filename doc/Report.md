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

**First**, the convolutional layer is the part that scans the image, simply repeating multiplication and addition. The kernel used at this time is called the convolution kernel. Multiply the value of the corresponding kernel element for each pixel, and add all of them to the value of the target pixel. This will result in a d * d matrix for the d * d input.
![ex_screenshot](./img/conv.PNG)

**Second**, the ReLU layer proceeds according to the following definition. ReLU (x) = max (0, x) That is, keeps its value for positive numbers and makes negative numbers zero.

**Third**, the pooling layer summarizes the d * d input as k * k. Where d> k. As a representative pooling method, max pooling is used to summarize the largest value of the input data.

After passing through these three layers, the input image can be said to pass through one filter. Pixel values ​​were modified by the kernel and reduced in size from **d * d** to **k * k**. This is called **convolve**.
![ex_screenshot](./img/cnn.PNG)

Now, if there are N of these filters, scan N times, relu application N times, and pooling N times, resulting in N results. But this is only half of CNN's role. CNN's ultimate goal is to extract the right features from the image. Therefore, in order **to extract the feature**, the **classification of the input image should proceed first**.

![ex_screenshot](./img/graph.PNG)
This is done at the Fully Connected Layer. The N (assumed to be feature) outputs obtained in the previous process are classified by passing through MLP. Then modify the values of the convolution kernel based on the output layer results.
Yes. The initial values of the convolution kernel are initially specified **randomly**. And, by learning map, we find the values necessary for classification by oneself. A person does not need to set a price in advance.

Let's say we have a photo of the dog, a label of Dog, a picture of the cat, and a label of Cat. After learning these pictures, if you give them the first picture of the dog that is not in the learning set, CNN can classify it as Dog.

***

