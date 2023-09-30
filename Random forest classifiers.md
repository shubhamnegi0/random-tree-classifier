```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
```


```python
df = pd.read_csv("User_Data.csv")
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
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>Male</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>Male</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>Female</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>Female</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>Male</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>User ID</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.000000e+02</td>
      <td>400.000000</td>
      <td>400.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.569154e+07</td>
      <td>37.655000</td>
      <td>69742.500000</td>
      <td>0.357500</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.165832e+04</td>
      <td>10.482877</td>
      <td>34096.960282</td>
      <td>0.479864</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.556669e+07</td>
      <td>18.000000</td>
      <td>15000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.562676e+07</td>
      <td>29.750000</td>
      <td>43000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.569434e+07</td>
      <td>37.000000</td>
      <td>70000.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.575036e+07</td>
      <td>46.000000</td>
      <td>88000.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.581524e+07</td>
      <td>60.000000</td>
      <td>150000.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 400 entries, 0 to 399
    Data columns (total 5 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   User ID          400 non-null    int64 
     1   Gender           400 non-null    object
     2   Age              400 non-null    int64 
     3   EstimatedSalary  400 non-null    int64 
     4   Purchased        400 non-null    int64 
    dtypes: int64(4), object(1)
    memory usage: 15.8+ KB
    


```python
df.shape
```




    (400, 5)




```python
df.isnull().sum()
```




    User ID            0
    Gender             0
    Age                0
    EstimatedSalary    0
    Purchased          0
    dtype: int64




```python
df.columns
```




    Index(['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'], dtype='object')




```python
# Data visualization
```


```python
fishes_data = ['User ID', 'Gender', 'Age', 'EstimatedSalary',"Purchased",]
for fish in fishes_data:
    plt.figure(figsize=(4,2))
    sns.boxplot(data = df, x = "Purchased",y = fish,palette= "viridis")
    plt.title(" Box plot")
    plt.xlabel("weight of a fish ")
    plt.ylabel(fish)
    plt.show()
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



    
![png](output_8_4.png)
    



```python
# counterplot
```


```python
plt.figure(figsize=(4,6))
sns.countplot(data= df, x = "Purchased",palette="viridis")
plt.title("counter plot")
plt.xlabel("weight of a fish")
plt.ylabel("fish")
plt.show()
```


    
![png](output_10_0.png)
    



```python
cor_matix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(cor_matix,annot=True,cmap="coolwarm",linewidths=0.5)
plt.title("Heatmap")
plt.show()
```

    C:\Users\$hubh\AppData\Local\Temp\ipykernel_10972\3840178309.py:1: FutureWarning: The default value of numeric_only in DataFrame.corr is deprecated. In a future version, it will default to False. Select only valid columns or specify the value of numeric_only to silence this warning.
      cor_matix = df.corr()
    


    
![png](output_11_1.png)
    



```python
df.columns
```




    Index(['User ID', 'Gender', 'Age', 'EstimatedSalary', 'Purchased'], dtype='object')




```python
drop_columns =['User ID', 'Gender', 'Age', 'EstimatedSalary']
X = df.drop(drop_columns,axis= 1)
X
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
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>1</td>
    </tr>
    <tr>
      <th>398</th>
      <td>0</td>
    </tr>
    <tr>
      <th>399</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>400 rows × 1 columns</p>
</div>




```python
y = df["Purchased"]
y
```




    0      0
    1      0
    2      0
    3      0
    4      0
          ..
    395    1
    396    1
    397    1
    398    0
    399    1
    Name: Purchased, Length: 400, dtype: int64




```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)
```


```python
X_train.shape, X_test.shape,y_test.shape,y_train.shape
```




    ((280, 1), (120, 1), (120,), (280,))




```python
from sklearn.preprocessing import  StandardScaler
```


```python
scale = StandardScaler()
scale
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>StandardScaler()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div>




```python
X_train = scale.fit_transform(X_train)
X_train
```




    array([[-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852]])




```python
X_test = scale.transform(X_test)
X_test
```




    array([[ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [-0.72804852],
           [ 1.37353483],
           [-0.72804852],
           [ 1.37353483],
           [ 1.37353483],
           [-0.72804852]])




```python
rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)


```


```python

rf_classifier.fit(X_train, y_train)

```




<style>#sk-container-id-6 {color: black;background-color: white;}#sk-container-id-6 pre{padding: 0;}#sk-container-id-6 div.sk-toggleable {background-color: white;}#sk-container-id-6 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-6 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-6 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-6 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-6 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-6 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-6 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-6 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-6 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-6 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-6 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-6 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-6 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-6 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-6 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-6 div.sk-item {position: relative;z-index: 1;}#sk-container-id-6 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-6 div.sk-item::before, #sk-container-id-6 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-6 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-6 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-6 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-6 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-6 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-6 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-6 div.sk-label-container {text-align: center;}#sk-container-id-6 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-6 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-6" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(random_state=42)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" checked><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div>




```python
y_pred = rf_classifier.predict(X_test)

```


```python
from sklearn.metrics import accuracy_score
```


```python
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy:.2f}')

```

    Accuracy: 1.00
    


```python
from sklearn.metrics import classification_report,confusion_matrix
```


```python
print("Classification Report:")
print(classification_report(y_test, y_pred))


```

    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        74
               1       1.00      1.00      1.00        46
    
        accuracy                           1.00       120
       macro avg       1.00      1.00      1.00       120
    weighted avg       1.00      1.00      1.00       120
    
    


```python

```


```python

```


```python

```


```python

```
