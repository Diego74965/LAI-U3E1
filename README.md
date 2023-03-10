# Unit 3, Exercise 1: Banknote Authentication
In this exercise, we are applying logistic regression to a banknote authentication dataset to distinguish between genuine and forged bank notes.

**The dataset consists of 1372 examples and 4 features for binary classification.** The features are:
1. Variance of wavelet-transformed image
2. Skewness of wavelet-transformed image
3. Kurtosis of a wavelet-transformed image
4. Entropy of the image

<sub>
    Details about this dataset <a href="https://archive.ics.uci.edu/ml/datasets/banknote+authentication">https://archive.ics.uci.edu/ml/datasets/banknote+authentication</a>
    </sub>

In essence, these four features represent features that were manually extracted from image data.

You are ecouraged to explore the dataset further.

## 1) Installing Libraries


```python
%load_ext watermark
%watermark -v -p numpy,pandas,matplotlib,torch
```

    Python implementation: CPython
    Python version       : 3.9.15
    IPython version      : 7.31.1
    
    numpy     : 1.23.5
    pandas    : 1.4.4
    matplotlib: 3.6.2
    torch     : 1.13.1
    
    

## 2) Loading the Dataset
We are using the familiar `read_csv` function from pandas to load the dataset:


```python
import pandas as pd
```


```python
df = pd.read_csv("data_banknote_authentication.txt", header = None)
df.head()
```




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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.62160</td>
      <td>8.6661</td>
      <td>-2.8073</td>
      <td>-0.44699</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.54590</td>
      <td>8.1674</td>
      <td>-2.4586</td>
      <td>-1.46210</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.86600</td>
      <td>-2.6383</td>
      <td>1.9242</td>
      <td>0.10645</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.45660</td>
      <td>9.5228</td>
      <td>-4.0112</td>
      <td>-3.59440</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.32924</td>
      <td>-4.4552</td>
      <td>4.5718</td>
      <td>-0.98880</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
x_features = df[[0, 1, 2, 3]].values
y_labels = df[4].values
```

Number of examples and features:


```python
x_features.shape
# 1372 examples, 4 features
```




    (1372, 4)



Looking at the label distribution:


```python
import numpy as np
```


```python
np.bincount(y_labels)
```




    array([762, 610], dtype=int64)



## 3) Descriptive analysis


```python
df.mean()
```




    0    0.433735
    1    1.922353
    2    1.397627
    3   -1.191657
    4    0.444606
    dtype: float64




```python
df.min()
```




    0    -7.0421
    1   -13.7731
    2    -5.2861
    3    -8.5482
    4     0.0000
    dtype: float64




```python
df.max()
```




    0     6.8248
    1    12.9516
    2    17.9274
    3     2.4495
    4     1.0000
    dtype: float64



## 4) Defining a DataLoader


```python
from torch.utils.data import Dataset, DataLoader
```


```python
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.features = torch.tensor(x, dtype = torch.float32)
        self.labels = torch.tensor(y, dtype = torch.float32)
        
    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return self.labels.shape[0]
```

We will be using 80% of the data for training, 20% of the data for validation. In real life, we would also have a separate dataset for the final test set.


```python
train_size = int(x_features.shape[0]*0.80)
train_size
```




    1097




```python
val_size = x_features.shape[0] - train_size
val_size
```




    275



Using `torch.utils.data.random_split`, we generate the training and validation sets along with the respective data loaders:


```python
import torch
```


```python
dataset = MyDataset(x_features, y_labels)

torch.manual_seed(74)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    dataset = train_set,
    batch_size = 10,
    shuffle = True
)

val_loader = DataLoader(
    dataset = val_set,
    batch_size = 10,
    shuffle = False
)
```

## 5) Implementing the model


```python
#Logistic regression model

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        
    def forward(self, x):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas
```

## 6) The training loop
In this section, we are using the training loopfrom Unit 3.6. We added the line `if not batch_idx % 20` to only print the loss for every 20th batch (to simplify output).


```python
import torch.nn.functional as F

torch.manual_seed(1)
model = LogisticRegression(num_features = 4)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.04)

num_epochs = 10

for epoch in range(num_epochs):
    
    model = model.train()
    for batch_idx, (features, class_labels) in enumerate(train_loader):
        probas = model(features)
        
        loss = F.binary_cross_entropy(probas, class_labels.view(probas.shape))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Logging
        if not batch_idx % 40:
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}'
                  f' | Batch {batch_idx:03d}/{len(train_loader):03d}'
                  f' | Loss: {loss:.2f}')
```

    Epoch: 001/010 | Batch 000/110 | Loss: 1.86
    Epoch: 001/010 | Batch 040/110 | Loss: 0.18
    Epoch: 001/010 | Batch 080/110 | Loss: 0.17
    Epoch: 002/010 | Batch 000/110 | Loss: 0.17
    Epoch: 002/010 | Batch 040/110 | Loss: 0.05
    Epoch: 002/010 | Batch 080/110 | Loss: 0.12
    Epoch: 003/010 | Batch 000/110 | Loss: 0.09
    Epoch: 003/010 | Batch 040/110 | Loss: 0.03
    Epoch: 003/010 | Batch 080/110 | Loss: 0.03
    Epoch: 004/010 | Batch 000/110 | Loss: 0.06
    Epoch: 004/010 | Batch 040/110 | Loss: 0.19
    Epoch: 004/010 | Batch 080/110 | Loss: 0.05
    Epoch: 005/010 | Batch 000/110 | Loss: 0.10
    Epoch: 005/010 | Batch 040/110 | Loss: 0.04
    Epoch: 005/010 | Batch 080/110 | Loss: 0.04
    Epoch: 006/010 | Batch 000/110 | Loss: 0.16
    Epoch: 006/010 | Batch 040/110 | Loss: 0.16
    Epoch: 006/010 | Batch 080/110 | Loss: 0.06
    Epoch: 007/010 | Batch 000/110 | Loss: 0.12
    Epoch: 007/010 | Batch 040/110 | Loss: 0.16
    Epoch: 007/010 | Batch 080/110 | Loss: 0.16
    Epoch: 008/010 | Batch 000/110 | Loss: 0.02
    Epoch: 008/010 | Batch 040/110 | Loss: 0.08
    Epoch: 008/010 | Batch 080/110 | Loss: 0.13
    Epoch: 009/010 | Batch 000/110 | Loss: 0.02
    Epoch: 009/010 | Batch 040/110 | Loss: 0.15
    Epoch: 009/010 | Batch 080/110 | Loss: 0.07
    Epoch: 010/010 | Batch 000/110 | Loss: 0.04
    Epoch: 010/010 | Batch 040/110 | Loss: 0.07
    Epoch: 010/010 | Batch 080/110 | Loss: 0.01
    

## 7) Evaluating the results
Reusing the code from Unit 3.6, we will calculate the training and validation set accuracy.


```python
def compute_accuracy(model, dataloader):
    model = model.eval()
    
    correct = 0.0
    total_examples = 0
    
    for idx, (features, class_labels) in enumerate(dataloader):
        
        with torch.no_grad():
            probas = model(features)
            
        pred = torch.where(probas > 0.5, 1, 0)
        lab = class_labels.view(pred.shape).to(pred.dtype)
        
        compare = lab == pred
        correct += torch.sum(compare)
        total_examples += len(compare)
        
    return correct / total_examples
```


```python
train_acc = compute_accuracy(model, train_loader)
print(f"Accuracy: {train_acc * 100: .2f}%")
```

    Accuracy:  98.81%
    


```python
val_acc = compute_accuracy(model, val_loader)
print(f"Accuracy: {val_acc * 100: .2f}%")
```

    Accuracy:  98.18%
    

We find that the best learning rate and number of epochs is 0.05 and 10 respectievly. These values gives us a training accuracy of >99% and a validation accuracy of >98%.

**Training Accuracy:** The accuracy of the model on the data it was trained on.

**Validation Accuracy:** The accuracy of the model on a separate set of data that was not used during training.
