---
## Deep Learning Context and PyTorch Basics

---

Hello and welcome! 🎉

In this notebook, we will start our exciting journey into the world of **Deep Learning**.

**What you will do in this notebook:**
- Understand what Deep Learning is and why it matters.
- Learn about basic mathematical structures (scalars, vectors, matrices, tensors).
- Explore **PyTorch**, the powerful deep learning library.
- Perform tensor operations and prepare real-world data for modeling.
- Visualize real data and build intuition about learning problems.

✅ **Pro Tip:**  
Take your time to read the Markdown explanations carefully, and try **running and modifying** code cells.  
**Learning happens when you experiment!**

Ready? 🚀 Let's begin!

---


## 1. Introduction: Why Deep Learning?

### 1.1 What is Deep Learning? 

**Deep Learning** is a branch of machine learning that focuses on using **neural networks with many layers** (hence the term 'deep') to model complex patterns in data.

At its core:
- A **neural network** is a collection of simple units (called **neurons**) that process information.
- **Deep** means stacking many layers of these units together, allowing the network to learn **highly complex mappings** from inputs to outputs.

> **Analogy:**  
Just like the human brain is made up of many layers of neurons working together to recognize objects, sounds, and patterns, deep learning models use artificial layers to process and learn from data.

Deep learning has proven especially powerful because:
- It **automatically learns features** from raw data (no manual feature engineering needed).
- It can handle **very large and complex datasets** like images, videos, and text.
- It continues to improve as more data and computing power become available.

---


### 1.2 Real-World Applications of Deep Learning

Deep Learning powers many technologies you interact with every day! Some exciting examples include:

- **Image Classification:**  
  - Systems that can detect whether an image contains a **cat**, **dog**, or **car**.  
  - Used in apps like Google Photos and Instagram.

- **Speech Recognition:**  
  - Virtual assistants like **Siri**, **Alexa**, and **Google Assistant** use deep learning to understand spoken commands.

- **Text Generation:**  
  - Language models such as **ChatGPT** can generate text, answer questions, or even write poetry.

- **Medical Diagnostics:**  
  - Deep learning models can help **detect cancer** or **analyze X-ray images** with high accuracy, assisting doctors.
  


### 1.3 Short Timeline of Deep Learning

Here is a brief history of some **milestones** in the development of Deep Learning:

- **1958:**  
  - **Perceptron** introduced by Frank Rosenblatt.  
  - The first model simulating a neuron that could learn simple patterns.

- **1986:**  
  - **Backpropagation algorithm** popularized by Geoffrey Hinton and others.  
  - This allowed **multi-layer** networks to be trained effectively.

- **2012:**  
  - **AlexNet** (by Alex Krizhevsky) wins the ImageNet competition by a huge margin using a deep convolutional neural network (CNN).  
  - Sparked a revolution in computer vision and deep learning adoption.

- **2017:**  
  - **Transformers** architecture introduced in the paper *"Attention is All You Need"*.  
  - Replaced recurrence with **self-attention**, enabling parallelization and scaling to very large models.  
  - Became the foundation for models like BERT, GPT, and ultimately LLMs.

- **2022:**  
  - **ChatGPT** (by OpenAI) demonstrated the incredible power of large deep language models.  
  - Widely adopted for text generation, coding assistance, and more.

These milestones show how deep learning has evolved from a theoretical idea to **a central technology shaping our world today**!

---


## 2. Introduction to Tensors

### 2.1 Scalars, Vectors, Matrices, and Tensors

In Deep Learning, we work with **tensors** — a generalization of scalars, vectors, and matrices.  
Tensors are the **building blocks** of data representation in modern neural networks.

---

🔢 Object Types and Dimensions

| Object   | Example                         | Dimension | Description                          |
|:--------:|:--------------------------------|:---------:|:-------------------------------------|
| Scalar   | $x = 5$                         | 0D        | A single number                      |
| Vector   | $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ | 1D | A list of numbers (1 axis)           |
| Matrix   | $\mathbf{X} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ | 2D | A grid/table of numbers (2 axes)     |
| Tensor   | e.g., $\mathcal{T} \in \mathbb{R}^{2 \times 2 \times 2}$ | 3D+       | A cube or higher-dimensional array   |

---

### 📏 Visual Intuition

**Scalar (0D):**  
$$
x = 5
$$

**Vector (1D):**  
$$
\mathbf{x} =
\begin{bmatrix}
1 \\
2 \\
3
\end{bmatrix}
$$

**Matrix (2D):**  
$$
\mathbf{X} =
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

**Tensor (3D Example):**  
Think of this as multiple matrices stacked together:
$$
\mathcal{T} =
\begin{bmatrix}
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix},
\begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix}
\end{bmatrix}
$$

*This is a $2 \times 2 \times 2$ tensor.*


🔵 **Key takeaways:**  
- A scalar is a single value
- A vector is a list
- A matrix is a grid
- A tensor can be a cube or even higher-dimensional structure!


#### 🖥️ Code Cell 2.1: Create Examples of Each


```python
import torch

# Scalar: 0D tensor
scalar = torch.tensor(5)
print("Scalar:", scalar)
print("Shape:", scalar.shape)

# Vector: 1D tensor
vector = torch.tensor([1, 2, 3])
print("\nVector:", vector)
print("Shape:", vector.shape)

# Matrix: 2D tensor
matrix = torch.tensor([[1, 2], [3, 4]])
print("\nMatrix:", matrix)
print("Shape:", matrix.shape)

# 3D Tensor
tensor_3d = torch.randn(2, 3, 4)  # random numbers
print("\n3D Tensor:", tensor_3d)
print("Shape:", tensor_3d.shape)

```

    Scalar: tensor(5)
    Shape: torch.Size([])
    
    Vector: tensor([1, 2, 3])
    Shape: torch.Size([3])
    
    Matrix: tensor([[1, 2],
            [3, 4]])
    Shape: torch.Size([2, 2])
    
    3D Tensor: tensor([[[-1.1431, -0.0169,  1.0002,  1.0447],
             [-0.8077, -1.6285, -1.0354, -0.9119],
             [-0.5886,  1.2081,  0.0288,  1.4786]],
    
            [[ 0.0623, -1.8246, -1.7332, -0.2198],
             [ 1.0742,  0.0484, -0.4647,  0.3443],
             [-0.1478, -2.0239,  0.5249, -0.7989]]])
    Shape: torch.Size([2, 3, 4])


#### 🛠️ Code Task 1.1.2.1

Create the following PyTorch tensors:

- A scalar tensor with value `10`, stored in a variable named `my_scalar`
- A vector tensor with values `[5, 10, 15]`, stored in `my_vector`
- A matrix tensor with shape **(2, 3)** (2 rows, 3 columns), using any values, stored in `my_matrix`

**Steps:**
1. Use `torch.tensor()`.
2. Print the tensor.
3. Print its `.shape` attribute.

🔔 *Hint:* Scalars have no shape dimensions like ( ); vectors have (N,); matrices have (N, M).

---



```python
# 1. Scalar tensor
my_scalar = torch.tensor(10)

# 2. Vector tensor
my_vector = torch.tensor([5, 10, 15])

# 3. Matrix tensor (2×3)
my_matrix = torch.randn(2, 3)

# Print tensors and shapes
print("Scalar:", my_scalar, "Shape:", my_scalar.shape)
print("Vector:", my_vector, "Shape:", my_vector.shape)
print("Matrix:\n", my_matrix, "Shape:", my_matrix.shape)
```

    Scalar: tensor(10) Shape: torch.Size([])
    Vector: tensor([ 5, 10, 15]) Shape: torch.Size([3])
    Matrix:
     tensor([[0.7650, 0.1896, 0.4550],
            [1.0653, 1.3493, 0.6840]]) Shape: torch.Size([2, 3])


### 2.2 Why Tensors? 

You might wonder: why do we need **tensors** in Deep Learning?  
Here are the key reasons:

---

🔹 **Compact Data Representation**  
Tensors allow us to organize and manage large datasets (images, audio, text, etc.) efficiently and consistently.

🔹 **Optimized for GPUs**  
Tensors are designed to work seamlessly on both CPUs and GPUs.  
PyTorch can easily move tensors to GPUs for **accelerated computation**, making it ideal for training large models.

🔹 **Mathematical Structure**  
Tensors allow operations like addition, multiplication, dot products, and more — enabling us to define and optimize complex neural network models through clear algebraic operations.

---

🚀 **In short:**  
Tensors = the universal language of Deep Learning!




## 3. PyTorch Setup

### 3.1 Importing Libraries

Before we can start working with tensors and deep learning models, we need to import **PyTorch** — the popular deep learning library.

```python
import torch


### 🔵 Why use PyTorch?

   🔹 **Flexibility:**
    PyTorch uses a dynamic computation graph, meaning you can modify your model structure on-the-fly. It's very developer-friendly!

   🔹 **Ease of Learning:**
    PyTorch code looks very similar to NumPy and basic Python, making it intuitive if you already know basic Python programming.

   🔹 **GPU Acceleration:**
    PyTorch can easily move computations to a GPU, making training deep learning models much faster.

    
<b>Fun Fact:</b> Major tech companies like Facebook (Meta), Tesla, and OpenAI use PyTorch for building cutting-edge AI systems!



#### 🖥️ Code Cell 3.1: Import PyTorch


```python
# Import the torch library
import torch

# Check PyTorch version
print(torch.__version__)
```

    2.7.0+cpu





    '2.7.0+cpu'



### 3.2 Creating Tensors in PyTorch

Once PyTorch is imported, we can create **tensors** using different methods:

- **torch.tensor():** Create tensor from a Python list or number.
- **torch.zeros():** Create tensor filled with zeros.
- **torch.ones():** Create tensor filled with ones.
- **torch.rand():** Create tensor filled with random numbers between 0 and 1.

---

#### 📐 Understanding Dimensions (Shape)

- A **scalar** has **0 dimensions**: just a number.
- A **vector** has **1 dimension**: a list of numbers.
- A **matrix** has **2 dimensions**: rows and columns.
- A **3D tensor** is like a stack of matrices.

Knowing the **shape** of your tensors is very important in deep learning!  
(Mismatched shapes cause errors.)

---

#### 🖥️ Code Cell 3.2: Examples


```python
# Scalar
scalar = torch.tensor(7)
print("Scalar:")
print(scalar)
print("Shape:", scalar.shape)

# 1D Vector
vector = torch.tensor([2, 4, 6])
print("\nVector:")
print(vector)
print("Shape:", vector.shape)

# 2D Matrix
matrix = torch.tensor([[1, 2], [3, 4]])
print("\nMatrix:")
print(matrix)
print("Shape:", matrix.shape)

# 3D Tensor
tensor3d = torch.rand((2, 3, 4))
print("\n3D Tensor:")
print(tensor3d)
print("Shape:", tensor3d.shape)

```

    Scalar:
    tensor(7)
    Shape: torch.Size([])
    
    Vector:
    tensor([2, 4, 6])
    Shape: torch.Size([3])
    
    Matrix:
    tensor([[1, 2],
            [3, 4]])
    Shape: torch.Size([2, 2])
    
    3D Tensor:
    tensor([[[0.1584, 0.7851, 0.9465, 0.7914],
             [0.2408, 0.6808, 0.7908, 0.8516],
             [0.3930, 0.3256, 0.3544, 0.4228]],
    
            [[0.9861, 0.9854, 0.6047, 0.4814],
             [0.6169, 0.0611, 0.2029, 0.4251],
             [0.1533, 0.3092, 0.7174, 0.3778]]])
    Shape: torch.Size([2, 3, 4])


### 3.3 Tensor Attributes

Every tensor in PyTorch comes with important **attributes** that describe its properties.

| Attribute | Description | Example |
|:---|:---|:---|
| `.dtype` | Data type of tensor elements (e.g., float32, int64) | `torch.float32` |
| `.shape` | Dimensions of the tensor (rows, columns, etc.) | `(2, 3)` |
| `.device` | Where the tensor is stored (CPU or GPU) | `cpu` or `cuda` |

Knowing these attributes helps you **debug** and **optimize** your deep learning models easily.

---


#### 🖥️ Code Cell 3.3: Display Properties


```python
# Checking properties
print("Data type:", matrix.dtype)
print("Shape:", matrix.shape)
print("Device:", matrix.device)

```

    Data type: torch.int64
    Shape: torch.Size([2, 2])
    Device: cpu


## 4. Tensor Operations

### Video: Tensors & PyTorch


```python
from IPython.display import VimeoVideo

VimeoVideo("1091215607", h="3298dbabb7", width=700, height=450)
```

### 4.1 Basic Tensor Operations

Tensors are not just storage for numbers — we also **perform operations** on them, like addition, multiplication, and matrix multiplication.

Here are some basic operations you should know:

---

#### ➡️ Element-wise Addition

Given two vectors:

$$
\mathbf{a} = [1, 2, 3], \quad \mathbf{b} = [4, 5, 6]
$$

Element-wise addition:

$$
\mathbf{a} + \mathbf{b} = [1+4, 2+5, 3+6] = [5, 7, 9]
$$

---

#### ➡️ Element-wise Multiplication

Element-wise multiplication (also called Hadamard product):

$$
\mathbf{a} \circ \mathbf{b} = [1 \times 4, 2 \times 5, 3 \times 6] = [4, 10, 18]
$$

---

#### ➡️ Dot Product (Inner Product)

The dot product of two vectors:

$$
\text{dot}(\mathbf{a}, \mathbf{b}) = (1 \times 4) + (2 \times 5) + (3 \times 6) = 4 + 10 + 18 = 32
$$

---

#### ➡️ Matrix Multiplication

Given two matrices:

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Matrix multiplication:

$$
\mathbf{A} \times \mathbf{B} =
\begin{bmatrix}
1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\
3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8
\end{bmatrix}
=
\begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}
$$

---

🔁 **Tensor operations** allow deep learning models to process inputs, update weights, and make predictions.


#### 🖥️ Code Cell 4.1: Basic Operations


```python
import torch

# Element-wise addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print("Element-wise addition:")
print(a + b)

# Element-wise multiplication
print("\nElement-wise multiplication:")
print(a * b)

# Dot product
print("\nDot product:")
print(torch.dot(a, b))

# Matrix multiplication
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print("\nMatrix multiplication:")
print(torch.matmul(A, B))

```

    Element-wise addition:
    tensor([5, 7, 9])
    
    Element-wise multiplication:
    tensor([ 4, 10, 18])
    
    Dot product:
    tensor(32)
    
    Matrix multiplication:
    tensor([[19, 22],
            [43, 50]])


#### ✍️ Practice!

#### 🛠️ Code Task 1.1.4.1

- Create two vectors:
  - $\mathbf{a} = [1, 3, 5]$
  - $\mathbf{b} = [2, 4, 6]$

Then perform the following operations:

- ✅ Element-wise addition → store in `add_result`
- ✅ Element-wise multiplication → store in `mult_result`
- ✅ Dot product → store in `dot_result`

Use PyTorch for all operations.

---

🔔 *Hint:* Use PyTorch functions:
- `torch.tensor()`
- `+` for addition
- `*` for element-wise multiplication
- `torch.dot()` for dot product



```python
# Create the vectors
a = torch.tensor([1,3,5])
b = torch.tensor([2, 4, 6])

# Operations
add_result = a + b
mult_result = a * b
dot_result = torch.dot(a,b)

# Display results
print("Addition:", add_result)
print("Multiplication:", mult_result)
print("Dot product:", dot_result)
```

    Addition: tensor([ 3,  7, 11])
    Multiplication: tensor([ 2, 12, 30])
    Dot product: tensor(44)


### 4.2 What is Broadcasting?

**Broadcasting** is a powerful concept in PyTorch that allows tensors with **different shapes** to be combined automatically during operations.

---

#### ➡️ Example:

Suppose we have:

- A matrix $X$ of shape $(2, 3)$:

$$
X =
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

- A vector $b$ of shape $(3,)$:

$$
b = [7, 8, 9]
$$

When we do:

$$
X + b
$$

The vector $b$ is **automatically expanded** (broadcasted) to match the shape of $X$.

**Result:**

$$
\begin{bmatrix}
1+7 & 2+8 & 3+9 \\
4+7 & 5+8 & 6+9
\end{bmatrix}
=
\begin{bmatrix}
8 & 10 & 12 \\
11 & 13 & 15
\end{bmatrix}
$$

---

#### 🔥 Why is Broadcasting Useful?

- Saves memory (no need to manually copy or reshape tensors)  
- Makes code **shorter**, **cleaner**, and more **efficient**  
- Often used to add **bias terms** to each row in neural networks


#### 🖥️ Code Cell 4.2: Broadcasting Example


```python
# Broadcasting example
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
bias = torch.tensor([10, 20, 30])

# Broadcasting addition
print("Broadcasting result:")
print(X + bias)
```

    Broadcasting result:
    tensor([[11, 22, 33],
            [14, 25, 36]])


#### 🛠️ Code Task 1.1.4.2

✅ **Your Task:**

- Create a **matrix** of size $(2, 3)$, named `my_matrix`
- Create a **vector** of size $(1, 3)$, named `my_vector` 
- Add them using **broadcasting** and store the result in a variable named `broadcast_sum`

---

**Use these values:**

Matrix:

$$
\begin{bmatrix}
7 & 8 & 9 \\
10 & 11 & 12
\end{bmatrix}
$$

Vector:

$$
[1, 1, 1]
$$

---

**Steps:**
- Use `torch.tensor()` to define both tensors  
- Add them together and print the result

🔔 *Hint:* You don't need to reshape anything manually — **broadcasting takes care of it**!



```python
# Define matrix and vector
my_matrix = torch.tensor([
    [7, 8, 9],
    [10, 11, 12]
])
my_vector = torch.tensor([1, 1, 1])

# Perform addition with broadcasting
broadcast_sum = my_matrix + my_vector

# Display result
print("Result of broadcasting addition:")
print(broadcast_sum)
```

    Result of broadcasting addition:
    tensor([[ 8,  9, 10],
            [11, 12, 13]])


## 5. Loading Real Data: Concrete_Data.csv

### 5.1 Introduction to the Concrete Dataset

For our deep learning journey, we will work with a real-world dataset:  
**Concrete Compressive Strength** dataset.

🔹 **What's inside this dataset?**
- **8 input features** related to the material composition and age of concrete:
  - Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, and Age (days).
- **1 output feature** (target):
  - Concrete Compressive Strength (in Megapascals, MPa).

---

🔵 **Why is this dataset important?**

- Concrete is the **most important construction material** in civil engineering.
- The strength of concrete depends on **complex, nonlinear relationships** between ingredients and curing time.
- Predicting compressive strength accurately helps in **designing stronger, safer structures**.

---

✅ **Goal:**  
Learn how to load, inspect, and work with real-world tabular data in preparation for building predictive models!

---
**Reference:**  
- [UCI Machine Learning Repository - Concrete Data Set](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)


#### 🖥️ Code Cell 5.1: Load the CSV File


```python
# Import pandas
import pandas as pd

# Load the Concrete_Data.csv dataset
data = pd.read_csv("Concrete_Data.csv")

# Show shape of the dataset
print("Shape of the dataset:", data.shape)

# Display first few rows
data.head()
```

    Shape of the dataset: (1030, 9)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cement (component 1)(kg in a m^3 mixture)</th>
      <th>Blast Furnace Slag (component 2)(kg in a m^3 mixture)</th>
      <th>Fly Ash (component 3)(kg in a m^3 mixture)</th>
      <th>Water  (component 4)(kg in a m^3 mixture)</th>
      <th>Superplasticizer (component 5)(kg in a m^3 mixture)</th>
      <th>Coarse Aggregate  (component 6)(kg in a m^3 mixture)</th>
      <th>Fine Aggregate (component 7)(kg in a m^3 mixture)</th>
      <th>Age (day)</th>
      <th>Concrete compressive strength(MPa, megapascals)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1040.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>79.99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>540.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>162.0</td>
      <td>2.5</td>
      <td>1055.0</td>
      <td>676.0</td>
      <td>28</td>
      <td>61.89</td>
    </tr>
    <tr>
      <th>2</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>270</td>
      <td>40.27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>332.5</td>
      <td>142.5</td>
      <td>0.0</td>
      <td>228.0</td>
      <td>0.0</td>
      <td>932.0</td>
      <td>594.0</td>
      <td>365</td>
      <td>41.05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>198.6</td>
      <td>132.4</td>
      <td>0.0</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>978.4</td>
      <td>825.5</td>
      <td>360</td>
      <td>44.30</td>
    </tr>
  </tbody>
</table>
</div>



### 5.2 Checking Dataset Structure

Once a dataset is loaded, the **first step** is to explore its structure:

- **`.shape`** → Tells us the number of rows and columns.
- **`.head()`** → Shows the first 5 rows to get a quick sense of the data.
- **`.columns`** → Lists all feature names.

✅ This helps us **understand what features** are available and **how the data looks** before any modeling.

---


#### 🖥️ Code Cell 5.2: Exploring Columns


```python
# Print column names
print("Column names:")
print(data.columns.tolist())
```

    Column names:
    ['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)', 'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)', 'Concrete compressive strength(MPa, megapascals) ']


#### 🛠️ Code Task 1.1.5.1

✅ Your Task:

You are working with the previous dataset stored in a pandas DataFrame named `data`

* Store the shape of the dataset in a variable named `dataset_shape`.
* Store the list of column names in a variable named `column_names`.

🔔 *Hint:* Use `.shape` and `.columns.tolist()` functions.

---



```python
# Dataset info
dataset_shape = data.shape
column_names = data.columns.tolist()

# Display results
print("Dataset shape:", dataset_shape)
print("Column names:", column_names)
```

    Dataset shape: (1030, 9)
    Column names: ['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)', 'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)', 'Concrete compressive strength(MPa, megapascals) ']


### 5.3 Checking for Missing Data

Before using the dataset to train models, we must **ensure there are no missing values**.

🔵 **Why is this important?**
- Neural networks expect **complete** data.
- Missing values can cause errors or reduce model accuracy.
- If missing values exist, we may need to handle them through **imputation** (filling missing values) or **removal**.

---

✅ Let's check if our Concrete dataset has any missing data!

---


#### 🖥️ Code Cell 5.3: Missing Values Check


```python
# Check for missing values
print(data.isnull().sum())
```

    Cement (component 1)(kg in a m^3 mixture)                0
    Blast Furnace Slag (component 2)(kg in a m^3 mixture)    0
    Fly Ash (component 3)(kg in a m^3 mixture)               0
    Water  (component 4)(kg in a m^3 mixture)                0
    Superplasticizer (component 5)(kg in a m^3 mixture)      0
    Coarse Aggregate  (component 6)(kg in a m^3 mixture)     0
    Fine Aggregate (component 7)(kg in a m^3 mixture)        0
    Age (day)                                                0
    Concrete compressive strength(MPa, megapascals)          0
    dtype: int64


## 6. Concrete Data as Tensors

### 6.1 Preparing Data for Deep Learning

In Deep Learning, models expect inputs and outputs to be represented as **tensors** — not as pandas DataFrames.

---

🔵 **Why do we need to convert data into tensors?**

- **PyTorch models** are built to work with **torch.Tensor** types.
- Tensors allow operations like:
  - Matrix multiplication
  - Gradient calculations
  - Automatic differentiation
- **Efficiency:**  
  Tensors can be processed much faster on **GPUs** compared to plain data structures like lists or pandas DataFrames.

---

✅ So, we must **split** our data into:
- **Inputs:** the 8 material/age features
- **Targets:** the concrete compressive strength (what we want to predict)

Then we will **convert** them into **torch.Tensor** format!

---


#### 🖥️ Code Cell 6.1: Splitting Inputs and Outputs 


```python
# Inputs: all columns except the last one
inputs = data.iloc[:, :-1]

# Targets: only the last column
targets = data.iloc[:, -1]

print("Input shape:", inputs.shape)
print("Target shape:", targets.shape)

```

    Input shape: (1030, 8)
    Target shape: (1030,)


### 6.2 Converting Data to Tensors

Now that we have separated **inputs** from **targets**, we will **convert** them to PyTorch tensors.

---

- The **inputs** should become a tensor of shape:

$$
(1030, 8)
$$

*(1030 examples, each with 8 features)*

- The **targets** should become a tensor of shape:

$$
(1030,)
$$

*(One target value for each example)*

---

📌 **Note:**  
When creating tensors from a NumPy array or DataFrame, we often need to specify:

```python
dtype=torch.float32


#### 🖥️ Code Cell 6.2: Convert to Tensors


```python
import torch

# Convert inputs and targets to PyTorch tensors
inputs_tensor = torch.tensor(inputs.values, dtype=torch.float32)
targets_tensor = torch.tensor(targets.values, dtype=torch.float32)

# Check their shapes
print("Inputs Tensor Shape:", inputs_tensor.shape)
print("Targets Tensor Shape:", targets_tensor.shape)

```

    Inputs Tensor Shape: torch.Size([1030, 8])
    Targets Tensor Shape: torch.Size([1030])


#### ✍️ Practice Cell 6.1 (Small Exercise)

#### 🛠️ Code Task 1.1.6.1

✅ Your Task:

- Create `inputs_100` for the **first 100 rows** of the `input_tensor`.
- Create `target_100` for the **first 100 rows** of the `target_tensor`.
- Print:
  - Shape of the inputs tensor
  - Shape of the targets tensor

🔔 *Hint:* Use slicing like `inputs_tensor[:100]`.



```python
# Slice the first 100 rows
inputs_100 = inputs_tensor[:100]
targets_100 = targets_tensor[:100]

# Print shapes
print("Shape of inputs_100:", inputs_100.shape)
print("Shape of targets_100:", targets_100.shape)
```

    Shape of inputs_100: torch.Size([100, 8])
    Shape of targets_100: torch.Size([100])


## 7. Basic Visualization

### 7.1 Importance of Visualization

Before building any machine learning model, it is essential to **visualize the data**.  
Visualization helps us:

🔹 **Spot Trends:**  
Identify obvious relationships between variables (e.g., higher cement content might lead to stronger concrete).

🔹 **Detect Outliers:**  
Find unusual or extreme values that could affect the model training.

🔹 **Understand Feature Distributions:**  
Know whether features are normally distributed, skewed, or contain strange patterns.

---

✅ Good visualizations build **intuition** and **guide feature engineering** later on!

---
**Reference:**  
- [Data Visualization for Machine Learning (Towards Data Science article)](https://towardsdatascience.com/why-data-visualization-matters-in-machine-learning-9a65ca92a9cd)


#### 🖥️ Code Cell 7.1: Plot Histogram of Target Variable


```python
import matplotlib.pyplot as plt

# Plot histogram for the target variable (Concrete Strength)
plt.hist(targets, bins=30, edgecolor='k')
plt.title('Distribution of Concrete Strength')
plt.xlabel('Strength (MPa)')
plt.ylabel('Count')
plt.show()
```


    
![png](output_64_0.png)
    


### 7.2 Observations from Histogram

After visualizing the distribution of **Concrete Strength**, ask yourself:

🔵 **Are values clustered or spread out?**
- Are most values concentrated in a narrow range (e.g., 20–50 MPa)?  
- Or are they widely spread?

🔵 **Are there any outliers?**
- Do we see extremely high or low strength values compared to the rest?

---

✅ These observations help you understand the complexity of the prediction task!

---


#### ✍️ Practice Cell 7.1 (Small Exercise)

#### 🛠️ Code Task 1.1.7.1

✅ Your Task:

- Plot a **histogram** for the **Water (component 4)** feature.

**Steps:**
1. Select the 'Water (component 4)(kg in a m^3 mixture)' column from the `inputs` and and store it in a variable named `water_feature`.
2. Use `plt.hist()` to plot the histogram.
3. Set at least 10 bins and add both `xlabel` and `title`.

🔔 *Hint:* Customize the number of bins and add labels/title!



```python
# Extract the column
water_feature = inputs["Water  (component 4)(kg in a m^3 mixture)"]

# Plot the histogram
plt.hist(water_feature, bins=30, edgecolor='k')
plt.title('Distribution of Concrete Strength')
plt.xlabel('Strength (MPa)')
plt.ylabel('Count')
plt.show()
```


    
![png](output_68_0.png)
    


### 7.3 Scatterplot — Inputs vs Target

Another important visualization is the **scatterplot**.

🔵 **What does a scatterplot show?**
- Scatterplots allow us to **visually explore the relationship** between an input feature and the target variable.

🔵 **Why use scatterplots?**
- They can reveal **linear** or **nonlinear** relationships.
- They can highlight **clusters**, **trends**, and **outliers**.

---

✅ In supervised learning, scatterplots often help to **guess how easily a model might predict the target**.


#### 🖥️ Code Cell 7.2: Scatterplot Example


```python
# Scatterplot: Cement vs Concrete Strength
# plt.scatter(inputs["Fly Ash (component 3)(kg in a m^3 mixture)"], targets, alpha=0.6)
# plt.title('Cement vs Concrete Strength')
# plt.xlabel('Cement (kg/m^3)')
# plt.ylabel('Strength (MPa)')
# plt.show()


# Cement (component 1)(kg in a m^3 mixture)                0
# Blast Furnace Slag (component 2)(kg in a m^3 mixture)    0
# Fly Ash (component 3)(kg in a m^3 mixture)               0
# Water  (component 4)(kg in a m^3 mixture)                0
# Superplasticizer (component 5)(kg in a m^3 mixture)      0
# Coarse Aggregate  (component 6)(kg in a m^3 mixture)     0
# Fine Aggregate (component 7)(kg in a m^3 mixture)        0
# Age (day)                                                0

import matplotlib.pyplot as plt

# All input features
features = [
    "Cement (component 1)(kg in a m^3 mixture)",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)",
    "Fly Ash (component 3)(kg in a m^3 mixture)",
    "Water  (component 4)(kg in a m^3 mixture)",
    "Superplasticizer (component 5)(kg in a m^3 mixture)",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)",
    "Age (day)"
]

# Create plots
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):
    plt.subplot(4, 2, i)
    plt.scatter(inputs[feature], targets, alpha=0.6)
    plt.title(f"{feature} vs Concrete Strength")
    plt.xlabel(feature)
    plt.ylabel("Strength (MPa)")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```


    
![png](output_71_0.png)
    


## 8 Mini Reflection

✅ Reflection Task:

Based on the visualizations you created and observed (histograms and scatterplots):

- **Which ingredients/features** do you think are the most important for predicting **Concrete Compressive Strength**?
- Were there any features that showed a **strong or weak relationship** with strength?

---

🔔 *Hint:* Think about:
- Features where strength increases clearly with the feature.
- Features that show scattered/noisy patterns.

✅ This type of reflection builds critical thinking about feature importance and helps you develop model intuition later!

---


## 🎉 Congratulations on Completing Notebook 1! 🎉

Amazing work! You have completed the first step of your Deep Learning journey. 🧠✨

**Today you have learned:**
- What Deep Learning is and its real-world impact.
- How tensors (scalars, vectors, matrices, and higher-dimensional arrays) work.
- How to perform essential tensor operations in PyTorch.
- How to load, inspect, and visualize real-world data.

**Coming Up Next:**
In the next notebook, we'll start building our **first machine learning model** — a simple **Linear Regression** model — and understand how models learn through **gradient descent**.

---

✅ Take a short break, reflect on what you've learned, and get ready to dive deeper!

---

