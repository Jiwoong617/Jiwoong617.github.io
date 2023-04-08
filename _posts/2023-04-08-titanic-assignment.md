<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

<pre>
/kaggle/input/titanic/train.csv
/kaggle/input/titanic/test.csv
/kaggle/input/titanic/gender_submission.csv
</pre>

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings(action='ignore')
```

## 데이터 셋



```python
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(train.columns.values) # 특성
```

<pre>
['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch'
 'Ticket' 'Fare' 'Cabin' 'Embarked']
</pre>
## 특성에 따른 생존율



```python
#성별에 따른 생존율
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Sex</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>0.742038</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>0.188908</td>
    </tr>
  </tbody>
</table>
</div>



```python
#형제, 배우에따른 생존율
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.535885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
#부모자식에 따른 생존율
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Parch</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
#pclass에 따른 생존율
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
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
      <th>Pclass</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.629630</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.472826</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.242363</td>
    </tr>
  </tbody>
</table>
</div>


## 특성에 대한 분포율



```python
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#0은 사망, 1은 생존
```

<pre>
<seaborn.axisgrid.FacetGrid at 0x7a839dcc88d0>
</pre>
<img src="data:image/png;base64,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6urU0NAQNHev16uJEyd227nff//9Kiws1I033hjUHm9z3bJli8aNG6dvfvObGjhwoMaMGaMnnnjC2R9v873uuuv0u9/9Tu+++64k6a233tL27dv1ta99TVJsz5caE5ufS2dQZz4VT/ONVo2J2AMwI+mjjz7S6dOn5fP5gtp9Pp8aGhqiNKrwMzMtXLhQ1113nXJyciTJmd+55n7gwIEuH2Nnbdy4UW+++aZ27tzZbl+8zXX//v1as2aNFi5cqAcffFCvv/665s+fL6/Xq7vuuivu5rt48WI1NzdrxIgR6tWrl06fPq0VK1botttukxTbny81JjY/l46izsRnnYlWjemWwemMz/6CsZl16JfTY9W8efP09ttva/v27e32xcPc6+vrtWDBAm3dulWJiYnn7RcPc5WktrY2jRs3TiUlJZKkMWPGqLa2VmvWrNFdd93l9IuX+W7atElPPfWUysvL9aUvfUk1NTUqKipSRkaGZs+e7fSL5fnG8tjCId5rjESdiec6E60a0y0v1Q0YMEC9evVq9y+/xsbGdsmyu/re976nLVu26JVXXtGgQYOcdr/fL0lxMffq6mo1NjYqNzdXCQkJSkhIUGVlpX76058qISHBmU88zFWS0tPTdeWVVwa1ffGLX9QHH3wgKb4+W0l64IEHtGTJEs2cOVNXXXWV7rzzTn3/+99XaWmppNieLzUmNj+XjqDOxG+diVaN6ZbBqXfv3srNzVVFRUVQe0VFhfLy8qI0qvAwM82bN0+bN2/W73//e2VlZQXtz8rKkt/vD5p7a2urKisru93cb7jhBu3evVs1NTXONm7cON1+++2qqanR0KFD42aukvTlL3+53de+3333XefHr+Pps5WkY8eO6aKLgktMr169nK8Kx/J8qTGx+bl0BHUmfutM1GpMh28rj7IzXxVeu3at7dmzx4qKiqxPnz72/vvvR3tonfLd737X0tLSbNu2bXbw4EFnO3bsmNNn5cqVlpaWZps3b7bdu3fbbbfd1i2/SnouZ3/bxSy+5vr6669bQkKCrVixwvbt22dPP/20JScn21NPPeX0iaf5zp49277whS84XxXevHmzDRgwwBYtWuT0ieX5UmNi83MJB+pMfMw3WjWm2wYnM7N/+7d/syFDhljv3r1t7NixztdpuzNJ59zWrVvn9Glra7OHHnrI/H6/eb1eu/7662337t3RG3QYfbagxdtc//M//9NycnLM6/XaiBEj7PHHHw/aH0/zbWlpsQULFtjgwYMtMTHRhg4dasuWLbNAIOD0ifX5UmNi83PpLOpMfMw3WjXGY2bW8fUqAACAnqNb3uMEAAAQDQQnAAAAlwhOAAAALhGcAAAAXCI4AQAAuERwAgAAcIngBAAA4BLBCQAAwCWCEwAAgEsEJ0RFVVWVevXqpa9+9avRHgqAOESNQaTwkyuIiu985ztKSUnRz372M+3Zs0eDBw+O9pAAxBFqDCKFFSd0uaNHj+oXv/iFvvvd7+rrX/+61q9fH7R/y5Ytys7OVlJSkiZNmqQNGzbI4/Ho8OHDTp+qqipdf/31SkpKUmZmpubPn6+jR4927UQAxCRqDCKJ4IQut2nTJg0fPlzDhw/XHXfcoXXr1unMwuf777+vW265RdOmTVNNTY3mzJmjZcuWBR2/e/duTZkyRdOnT9fbb7+tTZs2afv27Zo3b140pgMgxlBjEFEGdLG8vDwrKyszM7OTJ0/agAEDrKKiwszMFi9ebDk5OUH9ly1bZpKsqanJzMzuvPNO+7u/+7ugPq+99ppddNFFdvz48chPAEBMo8YgklhxQpfau3evXn/9dc2cOVOSlJCQoFtvvVVPPvmks//qq68OOuaaa64Jel1dXa3169crJSXF2aZMmaK2tjbV1dV1zUQAxCRqDCItIdoDQM+ydu1anTp1Sl/4whecNjPTxRdfrKamJpmZPB5P0DH2me8vtLW1ac6cOZo/f3679+cGUKBno8Yg0ghO6DKnTp3Sz3/+cz388MMqKCgI2jdjxgw9/fTTGjFihF588cWgfW+88UbQ67Fjx6q2tlZXXHFFxMcMoPugxqBLRPM6IXqW5557znr37m2HDx9ut+/BBx+00aNH2/79++3iiy+2RYsW2d69e23Tpk02aNAgk+Qc99Zbb1lSUpLNnTvXdu3aZe+++679+te/tnnz5nX1lADEEGoMugL3OKHLrF27VjfeeKPS0tLa7ZsxY4ZqamrU1NSkX/3qV9q8ebNGjhypNWvWON948Xq9kqSRI0eqsrJS+/bt01e+8hWNGTNG//iP/6j09PQunQ+A2EKNQVfgAZiIeStWrNBjjz2m+vr6aA8FQByixiAU3OOEmPPoo4/q6quvVv/+/fXf//3f+ud//meenwIgbKgx6AyCE2LOvn37tHz5ch06dEiDBw/W3//932vp0qXRHhaAOEGNQWdwqQ4AAMAlbg4HAABwieAEAADgEsEJAADAJYITAACASwQnAAAAlwhOAAAALhGcAAAAXCI4AQAAuERwAgAAcOn/AJVwhixeMN4WAAAAAElFTkSuQmCC"/>


```python
#객실 등급과 생존여부에 따른 연령분포
# 열을 생존 여부, 행(row)과 색깔(hue)을 객실 등급으로 나눔, width = height * aspect
grid = sns.FacetGrid(train, col='Survived', row='Pclass', hue="Pclass", height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20) # 투명도(alpha): 0.5

# 범례 추가
grid.add_legend();
```

<img src="data:image/png;base64,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UlKiefPm6YYbblBJSYkeffRR3XnnnXK73WrWrJmeeeYZ7dmzR3/605/UoUMHFRQUeKZIvvTSS1qxYoW2bdum66+/XoWFhfrggw9q/Fx8CvXPP/98ldttNptSU1OVmppakz4BAAAAteq9997T1q1bNXToUK1evVohISHatm2bWrZsKUnq0qXLZfcdN26c1+Pnn39e7dq1U25urnr06KETJ04oJiZGAwcOlM1mU8eOHT1tT5w4IZfLpWHDhqlly5bq0KGD+vXrV+PnU+371AMAAABW8sorr6h169YKCAhQXFyc4uPjtWrVKrndbg0aNMgT6K/k008/1eTJkxUdHS2n06moqChJFwO7dPFLuW63W127dtWcOXO0b98+z77jx49XaWmpoqOjdd9992nXrl2X/UTAF4R6AAAANAlDhgyR2+1WXl6ezp49q507d6pdu3ZyOBw+HWfUqFH66quvtGHDBh0+fFiHDx+WdHEuvST16tVL+fn5euKJJ1RaWqoJEyborrvukiRFRkYqLy9Pq1evlsPhUFJSkuLj4/Xdd9/V6LkR6gEAANAkBAUFqXPnzurYsaPXqHxsbKwOHjx4VcH6q6++0vHjx/XrX/9aQ4cO1XXXXafTp09Xaud0OjVx4kRt2LBBL774onbs2KGvv/5a0sW5/HfccYeeeeYZ7d+/X++++66ys7Nr9NxqfEtLAAAAwMpmz56tVatWadKkSUpJSVFISIgOHTqkfv36qWvXrl5tr7nmGrVt21br169XeHi4Tpw4oQULFni1WbFihcLDw9WzZ081a9ZM27dvl8vlUps2bbRp0yadP39e/fv3V2BgoP7whz/I4XB4zbuvDkbqAQAA0KS1bdtWb731lr755hsNHjxYvXv31oYNGy45x75Zs2batm2bjh49qh49euihhx7SU0895dWmdevWWrp0qfr06aO+ffvqs88+06uvvqpmzZqpTZs22rBhg2655RbFxsbqzTff1J///Ge1bdu2Rs/BZowxNTqCnxUXFyskJERFRUVyOp313R0AsAzqJwA0XYzUAwAAABZHqAcAAAAsjlAPAAAAWByhHgAAALA4Qj0AAABgcYR6AAAAwOII9QAAAIDFEeoBAAAAiyPUAwAAABZHqAcAAAAsrkV9dwAAAADwtxUZH9Xp+R4a3sXnfQ4cOKCnnnpKR48e1cmTJ7Vr1y6NGTOm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RIRmzJjhaWe162O1/tYF6qc3aueVUT+r1ljrZ0Nk6ek3P/rRj9S8efNKI0unTp2q9H98TcWDDz6oPXv26O2331b79u09610ulyQ12Wt19OhRnTp1Sr1791aLFi3UokULZWZm6plnnlGLFi0816CpXp/w8HB1797da911112nEydOSOLfz//6X/9LCxYs0KRJk3TDDTdo2rRpeuihh5Seni7JeteH2nlp1M/KqJ1XRv2sWmOrnw2ZpUN9q1at1Lt3b2VkZHitz8jI0IABA+qpV/XDGKPZs2dr586deuuttxQVFeW1PSoqSi6Xy+talZeXKzMzs0lcq6FDhyo7O1tut9uz9OnTR1OmTJHb7VZ0dHSTvj633HJLpVv4ffTRR+rYsaMk/v18++23atbMu1w2b97cc0s2q10faqc36uflUTuvjPpZtcZWPxu0+vh2rj9V3Jbt+eefN7m5uSY5OdkEBQWZzz77rL67Vqd++ctfmpCQELN//35z8uRJz/Ltt9962jz55JMmJCTE7Ny502RnZ5uf//znTfqWUd+/g4MxTfv6vPfee6ZFixZmyZIl5uOPPzZbtmwxgYGB5oUXXvC0acrXZ8aMGebaa6/13JJt586d5kc/+pGZP3++p43Vrg+18z+on76hdnqjflatMdbPhsryod4YY1avXm06duxoWrVqZXr16uW5DVlTIumSy8aNGz1tLly4YBYtWmRcLpex2+0mPj7eZGdn11+n69kP35ia+vX585//bHr06GHsdrvp1q2bWb9+vdf2pnx9iouLzdy5c02HDh1MQECAiY6ONgsXLjRlZWWeNla8PtTOi6ifvqF2Vkb9vLzGWj8bIpsxxtTPZwQAAAAA/MHSc+oBAAAAEOoBAAAAyyPUAwAAABZHqAcAAAAsjlAPAAAAWByhHgAAALA4Qj0AAABgcYR6AAAAwOII9QAAAIDFEerR6GVlZal58+YaMWJEfXcFACyD2glYi80YY+q7E0Btuvfee9W6dWv97ne/U25urjp06FDfXQKABo/aCVgLI/Vo1EpKSvSnP/1Jv/zlL3X77bdr06ZNXtv37NmjmJgYORwODRkyRJs3b5bNZtP//M//eNpkZWUpPj5eDodDkZGRmjNnjkpKSur2iQBAHaJ2AtZDqEej9uKLL6pr167q2rWrpk6dqo0bN6riw6nPPvtMd911l8aMGSO32637779fCxcu9No/Oztbt956q8aOHasPP/xQL774ot555x3Nnj27Pp4OANQJaidgPUy/QaN2yy23aMKECZo7d67OnTun8PBw/fGPf9SwYcO0YMEC/eUvf1F2dran/a9//WstWbJEp0+fVps2bTR9+nQ5HA6tW7fO0+add97R4MGDVVJSooCAgPp4WgBQq6idgPUwUo9GKy8vT++9954mTZokSWrRooUmTpyo//N//o9ne9++fb326devn9fjo0ePatOmTWrdurVnufXWW3XhwgXl5+fXzRMBgDpE7QSsqUV9dwCoLc8//7zOnTuna6+91rPOGKOWLVvq9OnTMsbIZrN57fPDD64uXLig+++/X3PmzKl0fL40BqAxonYC1kSoR6N07tw5/f73v9dvf/tbJSYmem0bN26ctmzZom7duunVV1/12vb+++97Pe7Vq5dycnLUuXPnWu8zANQ3aidgXcypR6O0e/duTZw4UadOnVJISIjXtoULF+rVV1/Vzp071bVrVz300EP6xS9+Ibfbrf/+7//WP/7xD/3P//yPQkJC9OGHH+rmm2/W3Xffrfvuu09BQUE6fvy4MjIytGrVqnp6dgBQO6idgHUxpx6N0vPPP69hw4ZVelOSLo42ud1unT59Wi+99JJ27typ2NhYrV271nMHB7vdLkmKjY1VZmamPv74Yw0aNEg33XSTfvOb3yg8PLxOnw8A1AVqJ2BdjNQD37NkyRI999xzKigoqO+uAIBlUDuB+secejRpa9asUd++fdW2bVv99a9/1VNPPcV9lAHgCqidQMNDqEeT9vHHH2vx4sX6+uuv1aFDB/33f/+3UlJS6rtbANCgUTuBhofpNwAAAIDF8UVZAAAAwOII9QAAAIDFEeoBAAAAiyPUAwAAABZHqAcAAAAsjlAPAAAAWByhHgAAALA4Qj0AAABgcYR6AAAAwOL+P/9VbQKl8lHSAAAAAElFTkSuQmCC"/>


```python
#승선지와 객실에 따른 생존율
grid = sns.FacetGrid(train, row='Embarked', height=2.2, aspect=1.6)

# Pointplot으로 시각화, x: 객실 등급, y: 생존 여부, 색깔: 성별, x축 순서: [1, 2, 3], 색깔 순서: [남성, 여성]
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', order = [1, 2, 3], hue_order = ["male", "female"])

grid.add_legend()
```

<pre>
<seaborn.axisgrid.FacetGrid at 0x7a8393ccf750>
</pre>
<img src="data:image/png;base64,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"/>

## 데이터 전처리


### Name Feature



```python
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
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
      <th>Sex</th>
      <th>female</th>
      <th>male</th>
    </tr>
    <tr>
      <th>Title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Capt</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Col</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Countess</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Don</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Dr</th>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Jonkheer</th>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Lady</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Major</th>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>0</td>
      <td>40</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>182</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mlle</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mme</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0</td>
      <td>517</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>125</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Ms</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Rev</th>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Sir</th>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


흔하지 않은 Title을 Other로 대체



```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer',
                                                 'Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
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
      <th>Title</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Master</td>
      <td>0.575000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miss</td>
      <td>0.702703</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mr</td>
      <td>0.156673</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mrs</td>
      <td>0.793651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Other</td>
      <td>0.347826</td>
    </tr>
  </tbody>
</table>
</div>



```python
for dataset in combine:
    dataset['Title'] = dataset['Title'].astype(str)

train.head()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>


### Sex Feature



```python
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].astype(str)
```

### Embarked Feature



```python
train.Embarked.value_counts(dropna=False)
```

<pre>
S      644
C      168
Q       77
NaN      2
Name: Embarked, dtype: int64
</pre>

```python
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)
```

### Age Feature


age nan 값에 나이의 평균 넣기



```python
for dataset in combine:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
```

<pre>
         AgeBand  Survived
0  (-0.08, 16.0]  0.550000
1   (16.0, 32.0]  0.344762
2   (32.0, 48.0]  0.403226
3   (48.0, 64.0]  0.434783
4   (64.0, 80.0]  0.090909
</pre>

```python
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
```

### Fare Feature



```python
print (train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test["Fare"].isnull()]["Pclass"])
```

<pre>
   Pclass       Fare
0       1  84.154687
1       2  20.662183
2       3  13.675550

152    3
Name: Pclass, dtype: int64
</pre>

```python
for dataset in combine:
    dataset['Fare'] = dataset['Fare'].fillna(13.675) # The only one empty fare data's pclass is 3.
```


```python
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
```

### SibSp & Parch Feature

family로 통합



```python
for dataset in combine:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset['Family'] = dataset['Family'].astype(int)
```

## 특성 추출 및 나머지 전처리



```python
features_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'AgeBand'], axis=1)

print(train.head())
print(test.head())
```

<pre>
   Survived  Pclass     Sex     Age  Fare Embarked Title  Family
0         0       3    male   Young     0        S    Mr       1
1         1       1  female  Middle     4        C   Mrs       1
2         1       3  female   Young     1        S  Miss       0
3         1       1  female  Middle     4        S   Mrs       1
4         0       3    male  Middle     1        S    Mr       0
   PassengerId  Pclass     Sex     Age  Fare Embarked Title  Family
0          892       3    male  Middle     0        Q    Mr       0
1          893       3  female  Middle     0        S   Mrs       1
2          894       2    male   Prime     1        Q    Mr       0
3          895       3    male   Young     1        S    Mr       0
4          896       3  female   Young     2        S   Mrs       2
</pre>

```python
train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop("PassengerId", axis=1).copy()

from sklearn.utils import shuffle
train_data, train_label = shuffle(train_data, train_label, random_state = 5)
```


```python
def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction
```

## Logistic Regression



```python
log_pred = train_and_test(LogisticRegression())
```

<pre>
Accuracy :  82.72 %
</pre>
## SVC



```python
svm_pred = train_and_test(SVC())
```

<pre>
Accuracy :  83.5 %
</pre>
## KNN



```python
knn_pred_4 = train_and_test(KNeighborsClassifier(n_neighbors = 4))
```

<pre>
Accuracy :  84.51 %
</pre>
## Random Forest



```python
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100))
```

<pre>
Accuracy :  88.55 %
</pre>
## Navie Bayes



```python
nb_pred = train_and_test(GaussianNB())
```

<pre>
Accuracy :  79.8 %
</pre>
## Linear SVC



```python
l_svc = train_and_test(LinearSVC())
```

<pre>
Accuracy :  82.94 %
</pre>
## Stochastic Gradient Descent



```python
sgd = train_and_test(SGDClassifier())
```

<pre>
Accuracy :  82.83 %
</pre>
## Decision Tree



```python
dt = train_and_test(DecisionTreeClassifier())
```

<pre>
Accuracy :  88.55 %
</pre>
## Perceptron



```python
perceptron = train_and_test(Perceptron())
```

<pre>
Accuracy :  75.76 %
</pre>

```python
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": rf_pred
})

submission.to_csv('submission_rf.csv', index=False)
```


```python
```
