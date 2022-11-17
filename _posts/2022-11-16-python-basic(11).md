---
layout: single
title: "2202.11.16.파이썬 기초 데이터처리 기술(11)"
categories: python_basic
tag: [python]
author_profile: false
toc: true
toc_label: "Unique Title"
toc_icon: "heart" # corresponding Font Awesome icon name (without fa prefix)
---

<div class="notice--success">
<h4>Python_Basic 학습 순서</h4>
<ul>
    <li>1. 파이썬 기초 및 데이터 처리</li>
    <li>2. 파이썬 기초 및 데이터 수집</li>
    <li>3. 파이썬 크롤링</li>
    <li>4. 파이썬 api</li>
</ul>
</div>

## 소개

안녕하세요 인공지능 개발자를 지망하는 윤동민입니다. 지금까지 파이썬 데이터 처리와 크롤링 등은 많이 해봤지만, 머닝러신, 딥러닝 쪽은 이론 공부만 해봤지 실제 구현은 해본적이 없습니다. 때문에 이번기회에 지금까지 배운것을 쭉 정리하고, 인공지능을 공부해 볼려고 합니다. 지금은 부족하지만, 앞으로 성장해가는 모습을 보여드리겠습니다.

## 개요

여러 카테고리를 만들어 파이썬과 인공지능에 대해 공부해 나갈 겁니다. 이 중 python_basic는 지금까지 제가 배운 내용을 정리해서 복습하는 기회로 소개해드리겠습니다.

## 파이썬 기초 데이터처리 기술(1)

1. 머신러닝 분류모델 유방암 데이터셋
2. 머신러닝 선형회귀모델
3. 머신러닝 모델저장 복원
4. 머신러닝 영화추천엔진
5. LabelEncoder
6. MinMaxScaler

#### 1. 머닝러신 분류모델 유방암 데이터셋

```python
### ( 1. 데이터 준비 )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

df = pd.read_csv(url, header=None)
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
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

## ( 실습 )

1. 유방함 데이터셋의 컬럼 이름을 찾아서
2. 컬럼명 만들기

3. 컬럼명 만들기
   - 0.id_number, 1.clump_thickness 2.unif_cell_size 3.unif_cell_shape
   - 4.marg_adhesion 5.single_epith_cell_size 6.bare_nuclei 7.bland_chromatin
   - 8.normal_nucleoli 9.mitoses 10.class

```python
col_name = ["id_number","clump_thickness",
            "unif_cell_size","unif_cell_shape",
            "marg_adhesion","single_epith_cell_size",
            "bare_nuclei","bland_chromatin",
            "normal_nucleoli","mitoses","class"]
df.columns = col_name
df.head(3)
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
      <th>id_number</th>
      <th>clump_thickness</th>
      <th>unif_cell_size</th>
      <th>unif_cell_shape</th>
      <th>marg_adhesion</th>
      <th>single_epith_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chromatin</th>
      <th>normal_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

### (실습)

1. class 컬럼 확인
2. 2 -> 0 .. 4->1로 바꿈.
3. cancer_ind 컬럼으로 만들기

```python
df["class"].value_counts()
```

    2    458
    4    241
    Name: class, dtype: int64

```python
df["cancer_ind"] = 0
df.loc[df["class"]==4, "cancer_ind"] = 1
```

```python
df.loc[df["class"]==2, "cancer_ind1"] = 0
df.loc[df["class"]==4, "cancer_ind1"] = 1
```

```python
df["cancer_ind2"] = df["class"].map({2:0, 4:1})
```

```python
df.head(3)
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
      <th>id_number</th>
      <th>clump_thickness</th>
      <th>unif_cell_size</th>
      <th>unif_cell_shape</th>
      <th>marg_adhesion</th>
      <th>single_epith_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chromatin</th>
      <th>normal_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
      <th>cancer_ind</th>
      <th>cancer_ind1</th>
      <th>cancer_ind2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 699 entries, 0 to 698
    Data columns (total 14 columns):
     #   Column                  Non-Null Count  Dtype
    ---  ------                  --------------  -----
     0   id_number               699 non-null    int64
     1   clump_thickness         699 non-null    int64
     2   unif_cell_size          699 non-null    int64
     3   unif_cell_shape         699 non-null    int64
     4   marg_adhesion           699 non-null    int64
     5   single_epith_cell_size  699 non-null    int64
     6   bare_nuclei             699 non-null    object
     7   bland_chromatin         699 non-null    int64
     8   normal_nucleoli         699 non-null    int64
     9   mitoses                 699 non-null    int64
     10  class                   699 non-null    int64
     11  cancer_ind              699 non-null    int64
     12  cancer_ind1             699 non-null    float64
     13  cancer_ind2             699 non-null    int64
    dtypes: float64(1), int64(12), object(1)
    memory usage: 76.6+ KB

```python
df["bare_nuclei"].value_counts().sort_index()
```

    1     402
    10    132
    2      30
    3      28
    4      19
    5      30
    6       4
    7       8
    8      21
    9       9
    ?      16
    Name: bare_nuclei, dtype: int64

```python
df.loc[df["bare_nuclei"] == "?", "bare_nuclei"] = np.nan
df["bare_nuclei"] = df["bare_nuclei"].fillna(df["bare_nuclei"].value_counts().index[0])
```

```python
df["bare_nuclei"].value_counts().sort_index()
```

    1     418
    10    132
    2      30
    3      28
    4      19
    5      30
    6       4
    7       8
    8      21
    9       9
    Name: bare_nuclei, dtype: int64

```python
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
      <th>id_number</th>
      <th>clump_thickness</th>
      <th>unif_cell_size</th>
      <th>unif_cell_shape</th>
      <th>marg_adhesion</th>
      <th>single_epith_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chromatin</th>
      <th>normal_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
      <th>cancer_ind</th>
      <th>cancer_ind1</th>
      <th>cancer_ind2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
X = df.drop(["id_number", "class", "cancer_ind2"], axis=1)
y = df.cancer_ind2
X.shape, y.shape
```

    ((699, 11), (699,))

```python
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
      <th>id_number</th>
      <th>clump_thickness</th>
      <th>unif_cell_size</th>
      <th>unif_cell_shape</th>
      <th>marg_adhesion</th>
      <th>single_epith_cell_size</th>
      <th>bare_nuclei</th>
      <th>bland_chromatin</th>
      <th>normal_nucleoli</th>
      <th>mitoses</th>
      <th>class</th>
      <th>cancer_ind</th>
      <th>cancer_ind1</th>
      <th>cancer_ind2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000025</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1002945</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1015425</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1016277</td>
      <td>6</td>
      <td>8</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1017023</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
X = df.drop(["id_number", "class", "cancer_ind2"], axis=1)
y = df.cancer_ind2
X.shape, y.shape
```

    ((699, 11), (699,))

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

```python
X_train.values[:5]
```

    array([[8, 10, 10, 7, 10, '10', 7, 3, 8, 1, 1.0],
           [5, 10, 10, 10, 10, '2', 10, 10, 10, 1, 1.0],
           [1, 1, 1, 1, 2, '1', 3, 1, 1, 0, 0.0],
           [3, 2, 2, 3, 2, '3', 3, 1, 1, 0, 0.0],
           [5, 10, 10, 3, 8, '1', 5, 10, 3, 1, 1.0]], dtype=object)

```python
X_train_scaled[:5]
```

    array([[ 1.23203096,  2.25737003,  2.24548782,  1.50338883,  2.99901619,
             1.71995381,  1.3873881 ,  0.0493974 ,  3.93974808,  1.34549055,
             1.34549055],
           [ 0.16698121,  2.25737003,  2.24548782,  2.57661092,  2.99901619,
            -0.43660366,  2.57692542,  2.36566183,  5.16333449,  1.34549055,
             1.34549055],
           [-1.25308511, -0.71708671, -0.76927481, -0.64305537, -0.55618118,
            -0.70617334, -0.19866165, -0.61239244, -0.34280437, -0.74322335,
            -0.74322335],
           [-0.54305195, -0.38659151, -0.43430118,  0.07242603, -0.55618118,
            -0.16703398, -0.19866165, -0.61239244, -0.34280437, -0.74322335,
            -0.74322335],
           [ 0.16698121,  2.25737003,  2.24548782,  0.07242603,  2.11021684,
            -0.70617334,  0.59436323,  2.36566183,  0.88078204,  1.34549055,
             1.34549055]])

```python
new_col = ["id_number","clump_thickness",
                         "unif_cell_size","unif_cell_shape",
                         "marg_adhesion","single_epith_cell_size",
                         "bare_nuclei","bland_chromatin",
                         "normal_nucleoli","mitoses","class"]
```

```python
import pandas as pd
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", header=None)
df.columns = new_col

df["bare_nuclei"] = df["bare_nuclei"].replace("?",np.NAN)
df["bare_nuclei"] = df["bare_nuclei"].fillna(df["bare_nuclei"].value_counts().index[0])
df['cancer_ind'] = df['class'].map({2:0,4:1})

X = df.drop(["id_number", "class", "cancer_ind"], axis=1)
y = df.cancer_ind

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

#  모델 성능
from sklearn.metrics import accuracy_score, confusion_matrix,\
                              roc_auc_score, roc_curve
y_pred = knn.predict(X_test_scaled)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, {"n_neighbors": [1,2,3,4,5]},
                          n_jobs = -1, cv=7, scoring = "roc_auc")
grid_search.fit(X_train_scaled, y_train)

knn_best = grid_search.best_estimator_
y_pred = knn_best.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
```

    C:\Users\ehdal\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
    C:\Users\ehdal\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)





    0.9666666666666667

```python
#  모델 성능
from sklearn.metrics import accuracy_score, confusion_matrix,\
                              roc_auc_score, roc_curve
y_pred = knn.predict(X_test_scaled)
```

    C:\Users\ehdal\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)

```python
accuracy_score(y_test, y_pred)
```

    0.9761904761904762

```python
confusion_matrix(y_test, y_pred)
```

    array([[141,   2],
           [  3,  64]], dtype=int64)

```python
roc_auc_score(y_test, y_pred)
```

    0.9706189333055005

```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(knn, {"n_neighbors": [1,2,3,4,5]},
                          n_jobs = -1, cv=7, scoring = "roc_auc")
grid_search.fit(X_train_scaled, y_train)
```

    GridSearchCV(cv=7, estimator=KNeighborsClassifier(n_neighbors=3), n_jobs=-1,
                 param_grid={'n_neighbors': [1, 2, 3, 4, 5]}, scoring='roc_auc')

```python
grid_search.best_params_
```

    {'n_neighbors': 5}

```python
knn_best = grid_search.best_estimator_
y_pred = knn_best.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
```

    C:\Users\ehdal\anaconda3\lib\site-packages\sklearn\neighbors\_classification.py:228: FutureWarning: Unlike other reduction functions (e.g. `skew`, `kurtosis`), the default behavior of `mode` typically preserves the axis it acts along. In SciPy 1.11.0, this behavior will change: the default value of `keepdims` will become False, the `axis` over which the statistic is taken will be eliminated, and the value None will no longer be accepted. Set `keepdims` to True or False to avoid this warning.
      mode, _ = stats.mode(_y[neigh_ind, k], axis=1)





    0.9666666666666667

#### 2. 머신러닝 선형회귀모델

### ( 실습 )

1. 데이터셋 와인데이터셋 품질 모델 만들기
   - 레드와인, 화이트와인
2. 입력변수 11
3. 출력변수 1개 -> 1~10점 품질을 예측하는
4. LinearRegression
5. fit_intercept = True

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
redwine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                      sep=";", header=0)
redwine["type"] = "red"

whitewine = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                      sep=";", header=0)
whitewine["type"] = "white"
```

```python
import warnings
warnings.filterwarnings(action="ignore")

wine = redwine.append(whitewine)
wine.head(3)
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>

```python
whitewine.columns = whitewine.columns.str.replace(" ", "_")
redwine.columns = redwine.columns.str.replace(" ", "_")

wine.columns = wine.columns.str.replace(" ", "_")
```

```python
# 7개 메소드
wine.head(3)
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine.tail(3)
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4895</th>
      <td>6.5</td>
      <td>0.24</td>
      <td>0.19</td>
      <td>1.2</td>
      <td>0.041</td>
      <td>30.0</td>
      <td>111.0</td>
      <td>0.99254</td>
      <td>2.99</td>
      <td>0.46</td>
      <td>9.4</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4896</th>
      <td>5.5</td>
      <td>0.29</td>
      <td>0.30</td>
      <td>1.1</td>
      <td>0.022</td>
      <td>20.0</td>
      <td>110.0</td>
      <td>0.98869</td>
      <td>3.34</td>
      <td>0.38</td>
      <td>12.8</td>
      <td>7</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4897</th>
      <td>6.0</td>
      <td>0.21</td>
      <td>0.38</td>
      <td>0.8</td>
      <td>0.020</td>
      <td>22.0</td>
      <td>98.0</td>
      <td>0.98941</td>
      <td>3.26</td>
      <td>0.32</td>
      <td>11.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6497 entries, 0 to 4897
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype
    ---  ------                --------------  -----
     0   fixed_acidity         6497 non-null   float64
     1   volatile_acidity      6497 non-null   float64
     2   citric_acid           6497 non-null   float64
     3   residual_sugar        6497 non-null   float64
     4   chlorides             6497 non-null   float64
     5   free_sulfur_dioxide   6497 non-null   float64
     6   total_sulfur_dioxide  6497 non-null   float64
     7   density               6497 non-null   float64
     8   pH                    6497 non-null   float64
     9   sulphates             6497 non-null   float64
     10  alcohol               6497 non-null   float64
     11  quality               6497 non-null   int64
     12  type                  6497 non-null   object
    dtypes: float64(11), int64(1), object(1)
    memory usage: 710.6+ KB

```python
wine.shape, wine.shape[0], wine.shape[1]
```

    ((6497, 13), 6497, 13)

```python
# 결측치
wine.isnull().sum().sum()
```

    0

### ( 실습 )

1. quality별 건수는?

```python
wine["quality"].value_counts().sort_index()
```

    3      30
    4     216
    5    2138
    6    2836
    7    1079
    8     193
    9       5
    Name: quality, dtype: int64

2. wine df에서 red wine의 quality별 건수는?

```python
wine[wine["type"] == "red"]
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.880</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.760</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.280</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.700</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1594</th>
      <td>6.2</td>
      <td>0.600</td>
      <td>0.08</td>
      <td>2.0</td>
      <td>0.090</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99490</td>
      <td>3.45</td>
      <td>0.58</td>
      <td>10.5</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1595</th>
      <td>5.9</td>
      <td>0.550</td>
      <td>0.10</td>
      <td>2.2</td>
      <td>0.062</td>
      <td>39.0</td>
      <td>51.0</td>
      <td>0.99512</td>
      <td>3.52</td>
      <td>0.76</td>
      <td>11.2</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1596</th>
      <td>6.3</td>
      <td>0.510</td>
      <td>0.13</td>
      <td>2.3</td>
      <td>0.076</td>
      <td>29.0</td>
      <td>40.0</td>
      <td>0.99574</td>
      <td>3.42</td>
      <td>0.75</td>
      <td>11.0</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1597</th>
      <td>5.9</td>
      <td>0.645</td>
      <td>0.12</td>
      <td>2.0</td>
      <td>0.075</td>
      <td>32.0</td>
      <td>44.0</td>
      <td>0.99547</td>
      <td>3.57</td>
      <td>0.71</td>
      <td>10.2</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1598</th>
      <td>6.0</td>
      <td>0.310</td>
      <td>0.47</td>
      <td>3.6</td>
      <td>0.067</td>
      <td>18.0</td>
      <td>42.0</td>
      <td>0.99549</td>
      <td>3.39</td>
      <td>0.66</td>
      <td>11.0</td>
      <td>6</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
<p>1599 rows × 13 columns</p>
</div>

```python
wine[wine["type"] == "red"]["quality"].value_counts().sort_index()
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
aa = wine[wine["type"] == "red"]
aa["quality"].value_counts().sort_index()
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
wine.query("type == 'red'")["quality"].value_counts().sort_index()
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
str_expr = "type == 'red'"
wine.query(str_expr)["quality"].value_counts().sort_index()
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

### ( 3. 함수로 만들어 처리하기 )

1. 와인종류(red, white)를 입력받아
2. 해당 와인의 quality별 건수를 결과로 받는 함수로 만들어 처리하기
3. 함수 이름 def check_wine()

```python
def check_wine(red_white):
    res = wine[wine["type"] == red_white]["quality"].value_counts().sort_index()
    return res
```

```python
check_wine("white")
```

    3      20
    4     163
    5    1457
    6    2198
    7     880
    8     175
    9       5
    Name: quality, dtype: int64

```python
def check_wine1(red_white, feature):
    res = wine[wine["type"] == red_white][feature].value_counts().sort_index()
    return res
```

```python
check_wine1("red","quality")
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
wine.describe()
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.215307</td>
      <td>0.339666</td>
      <td>0.318633</td>
      <td>5.443235</td>
      <td>0.056034</td>
      <td>30.525319</td>
      <td>115.744574</td>
      <td>0.994697</td>
      <td>3.218501</td>
      <td>0.531268</td>
      <td>10.491801</td>
      <td>5.818378</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.296434</td>
      <td>0.164636</td>
      <td>0.145318</td>
      <td>4.757804</td>
      <td>0.035034</td>
      <td>17.749400</td>
      <td>56.521855</td>
      <td>0.002999</td>
      <td>0.160787</td>
      <td>0.148806</td>
      <td>1.192712</td>
      <td>0.873255</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.800000</td>
      <td>0.080000</td>
      <td>0.000000</td>
      <td>0.600000</td>
      <td>0.009000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.987110</td>
      <td>2.720000</td>
      <td>0.220000</td>
      <td>8.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.400000</td>
      <td>0.230000</td>
      <td>0.250000</td>
      <td>1.800000</td>
      <td>0.038000</td>
      <td>17.000000</td>
      <td>77.000000</td>
      <td>0.992340</td>
      <td>3.110000</td>
      <td>0.430000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>0.290000</td>
      <td>0.310000</td>
      <td>3.000000</td>
      <td>0.047000</td>
      <td>29.000000</td>
      <td>118.000000</td>
      <td>0.994890</td>
      <td>3.210000</td>
      <td>0.510000</td>
      <td>10.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.700000</td>
      <td>0.400000</td>
      <td>0.390000</td>
      <td>8.100000</td>
      <td>0.065000</td>
      <td>41.000000</td>
      <td>156.000000</td>
      <td>0.996990</td>
      <td>3.320000</td>
      <td>0.600000</td>
      <td>11.300000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.660000</td>
      <td>65.800000</td>
      <td>0.611000</td>
      <td>289.000000</td>
      <td>440.000000</td>
      <td>1.038980</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>9.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine.quality.describe()
```

    count    6497.000000
    mean        5.818378
    std         0.873255
    min         3.000000
    25%         5.000000
    50%         6.000000
    75%         6.000000
    max         9.000000
    Name: quality, dtype: float64

```python
wine["quality"].describe()
```

    count    6497.000000
    mean        5.818378
    std         0.873255
    min         3.000000
    25%         5.000000
    50%         6.000000
    75%         6.000000
    max         9.000000
    Name: quality, dtype: float64

```python
wine.quality.unique()  # value_counts()의 index와 동일
```

    array([5, 6, 7, 4, 8, 3, 9], dtype=int64)

```python
check_wine("red")
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
wine.groupby("type")["quality"].describe()
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>red</th>
      <td>1599.0</td>
      <td>5.636023</td>
      <td>0.807569</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>white</th>
      <td>4898.0</td>
      <td>5.877909</td>
      <td>0.885639</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine.groupby("type")["quality"].quantile([0, 0.25, 0.5, 0.75, 1])
```

    type
    red    0.00    3.0
           0.25    5.0
           0.50    6.0
           0.75    6.0
           1.00    8.0
    white  0.00    3.0
           0.25    5.0
           0.50    6.0
           0.75    6.0
           1.00    9.0
    Name: quality, dtype: float64

```python
wine.groupby("type")["quality"].quantile([0, 0.25, 0.5, 0.75, 1]).unstack("type")
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
      <th>type</th>
      <th>red</th>
      <th>white</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.00</th>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>0.25</th>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>0.50</th>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>0.75</th>
      <td>6.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>8.0</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>

### ( 와인 종류별 품질의 분포 )

```python
import seaborn as sns

red_q = wine.loc[wine["type"] == "red", "quality"]
white_q = wine.loc[wine["type"] == "white", "quality"]
```

```python
plt.figure(figsize=(5,3))
sns.set_style("darkgrid")
sns.distplot(red_q, norm_hist=False, kde=False, color="red", label="Red Wine")
sns.distplot(white_q, norm_hist=False, kde=False, color="blue", label="white Wine")
plt.title("Dist of Q of WIne Type")
plt.xlabel("Quality Score")
plt.ylabel("Density")
plt.legend()
```

    <matplotlib.legend.Legend at 0x230844cb4f0>

![png](output_33_1.png)

```python
import statsmodels.api as sm
t_stat, p_value, df = sm.stats.ttest_ind(red_q, white_q)
print("t_stat = ", t_stat, " p_value = ", p_value, " df", df)
```

    t_stat =  -9.685649554187691  p_value =  4.888069044201823e-22  df 6495.0

```python
"t_stat :{:.3f}, p_value: {:.4f}, df: {}".format(t_stat,p_value, df)
```

    't_stat :-9.686, p_value: 0.0000, df: 6495.0'

### ( 실습 )

1. wine의 컬럼간 상관관계를 출력
2. "fixed_acidity", "chlorides", "density", "pH" 4개 컬럼의 상관관계

```python
df = wine[["fixed_acidity","chlorides","density","pH"]]
df.corr()
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
      <th>fixed_acidity</th>
      <th>chlorides</th>
      <th>density</th>
      <th>pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed_acidity</th>
      <td>1.000000</td>
      <td>0.298195</td>
      <td>0.458910</td>
      <td>-0.252700</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.298195</td>
      <td>1.000000</td>
      <td>0.362615</td>
      <td>0.044708</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.458910</td>
      <td>0.362615</td>
      <td>1.000000</td>
      <td>0.011686</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.252700</td>
      <td>0.044708</td>
      <td>0.011686</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine[["fixed_acidity","chlorides","density","pH"]].corr()
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
      <th>fixed_acidity</th>
      <th>chlorides</th>
      <th>density</th>
      <th>pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed_acidity</th>
      <td>1.000000</td>
      <td>0.298195</td>
      <td>0.458910</td>
      <td>-0.252700</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.298195</td>
      <td>1.000000</td>
      <td>0.362615</td>
      <td>0.044708</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.458910</td>
      <td>0.362615</td>
      <td>1.000000</td>
      <td>0.011686</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.252700</td>
      <td>0.044708</td>
      <td>0.011686</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
col = ["fixed_acidity","chlorides","density","pH"]
wine[col].corr()
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
      <th>fixed_acidity</th>
      <th>chlorides</th>
      <th>density</th>
      <th>pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed_acidity</th>
      <td>1.000000</td>
      <td>0.298195</td>
      <td>0.458910</td>
      <td>-0.252700</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.298195</td>
      <td>1.000000</td>
      <td>0.362615</td>
      <td>0.044708</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.458910</td>
      <td>0.362615</td>
      <td>1.000000</td>
      <td>0.011686</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.252700</td>
      <td>0.044708</td>
      <td>0.011686</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

### ( 함수를 만들어서 처리 )

1. 몇개 컬럼이 들어올지 모름 (\*args)
2. 함수 fun_corr(): 로 만들어서 처리

```python
def fun_corr(*args):
    print("args = ", args)
    col = list(args)
    print("col = ", col)
    return wine[col].corr()
```

```python
fun_corr("fixed_acidity","chlorides","density","pH")
```

    args =  ('fixed_acidity', 'chlorides', 'density', 'pH')
    col =  ['fixed_acidity', 'chlorides', 'density', 'pH']

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
      <th>fixed_acidity</th>
      <th>chlorides</th>
      <th>density</th>
      <th>pH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed_acidity</th>
      <td>1.000000</td>
      <td>0.298195</td>
      <td>0.458910</td>
      <td>-0.252700</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.298195</td>
      <td>1.000000</td>
      <td>0.362615</td>
      <td>0.044708</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.458910</td>
      <td>0.362615</td>
      <td>1.000000</td>
      <td>0.011686</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.252700</td>
      <td>0.044708</td>
      <td>0.011686</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine_corr = wine.corr()
wine_corr
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>fixed_acidity</th>
      <td>1.000000</td>
      <td>0.219008</td>
      <td>0.324436</td>
      <td>-0.111981</td>
      <td>0.298195</td>
      <td>-0.282735</td>
      <td>-0.329054</td>
      <td>0.458910</td>
      <td>-0.252700</td>
      <td>0.299568</td>
      <td>-0.095452</td>
      <td>-0.076743</td>
    </tr>
    <tr>
      <th>volatile_acidity</th>
      <td>0.219008</td>
      <td>1.000000</td>
      <td>-0.377981</td>
      <td>-0.196011</td>
      <td>0.377124</td>
      <td>-0.352557</td>
      <td>-0.414476</td>
      <td>0.271296</td>
      <td>0.261454</td>
      <td>0.225984</td>
      <td>-0.037640</td>
      <td>-0.265699</td>
    </tr>
    <tr>
      <th>citric_acid</th>
      <td>0.324436</td>
      <td>-0.377981</td>
      <td>1.000000</td>
      <td>0.142451</td>
      <td>0.038998</td>
      <td>0.133126</td>
      <td>0.195242</td>
      <td>0.096154</td>
      <td>-0.329808</td>
      <td>0.056197</td>
      <td>-0.010493</td>
      <td>0.085532</td>
    </tr>
    <tr>
      <th>residual_sugar</th>
      <td>-0.111981</td>
      <td>-0.196011</td>
      <td>0.142451</td>
      <td>1.000000</td>
      <td>-0.128940</td>
      <td>0.402871</td>
      <td>0.495482</td>
      <td>0.552517</td>
      <td>-0.267320</td>
      <td>-0.185927</td>
      <td>-0.359415</td>
      <td>-0.036980</td>
    </tr>
    <tr>
      <th>chlorides</th>
      <td>0.298195</td>
      <td>0.377124</td>
      <td>0.038998</td>
      <td>-0.128940</td>
      <td>1.000000</td>
      <td>-0.195045</td>
      <td>-0.279630</td>
      <td>0.362615</td>
      <td>0.044708</td>
      <td>0.395593</td>
      <td>-0.256916</td>
      <td>-0.200666</td>
    </tr>
    <tr>
      <th>free_sulfur_dioxide</th>
      <td>-0.282735</td>
      <td>-0.352557</td>
      <td>0.133126</td>
      <td>0.402871</td>
      <td>-0.195045</td>
      <td>1.000000</td>
      <td>0.720934</td>
      <td>0.025717</td>
      <td>-0.145854</td>
      <td>-0.188457</td>
      <td>-0.179838</td>
      <td>0.055463</td>
    </tr>
    <tr>
      <th>total_sulfur_dioxide</th>
      <td>-0.329054</td>
      <td>-0.414476</td>
      <td>0.195242</td>
      <td>0.495482</td>
      <td>-0.279630</td>
      <td>0.720934</td>
      <td>1.000000</td>
      <td>0.032395</td>
      <td>-0.238413</td>
      <td>-0.275727</td>
      <td>-0.265740</td>
      <td>-0.041385</td>
    </tr>
    <tr>
      <th>density</th>
      <td>0.458910</td>
      <td>0.271296</td>
      <td>0.096154</td>
      <td>0.552517</td>
      <td>0.362615</td>
      <td>0.025717</td>
      <td>0.032395</td>
      <td>1.000000</td>
      <td>0.011686</td>
      <td>0.259478</td>
      <td>-0.686745</td>
      <td>-0.305858</td>
    </tr>
    <tr>
      <th>pH</th>
      <td>-0.252700</td>
      <td>0.261454</td>
      <td>-0.329808</td>
      <td>-0.267320</td>
      <td>0.044708</td>
      <td>-0.145854</td>
      <td>-0.238413</td>
      <td>0.011686</td>
      <td>1.000000</td>
      <td>0.192123</td>
      <td>0.121248</td>
      <td>0.019506</td>
    </tr>
    <tr>
      <th>sulphates</th>
      <td>0.299568</td>
      <td>0.225984</td>
      <td>0.056197</td>
      <td>-0.185927</td>
      <td>0.395593</td>
      <td>-0.188457</td>
      <td>-0.275727</td>
      <td>0.259478</td>
      <td>0.192123</td>
      <td>1.000000</td>
      <td>-0.003029</td>
      <td>0.038485</td>
    </tr>
    <tr>
      <th>alcohol</th>
      <td>-0.095452</td>
      <td>-0.037640</td>
      <td>-0.010493</td>
      <td>-0.359415</td>
      <td>-0.256916</td>
      <td>-0.179838</td>
      <td>-0.265740</td>
      <td>-0.686745</td>
      <td>0.121248</td>
      <td>-0.003029</td>
      <td>1.000000</td>
      <td>0.444319</td>
    </tr>
    <tr>
      <th>quality</th>
      <td>-0.076743</td>
      <td>-0.265699</td>
      <td>0.085532</td>
      <td>-0.036980</td>
      <td>-0.200666</td>
      <td>0.055463</td>
      <td>-0.041385</td>
      <td>-0.305858</td>
      <td>0.019506</td>
      <td>0.038485</td>
      <td>0.444319</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine_corr[wine_corr["quality"] > 0]["quality"]
```

    citric_acid            0.085532
    free_sulfur_dioxide    0.055463
    pH                     0.019506
    sulphates              0.038485
    alcohol                0.444319
    quality                1.000000
    Name: quality, dtype: float64

```python
wine_corr[wine_corr["quality"] > 0].quality
```

    citric_acid            0.085532
    free_sulfur_dioxide    0.055463
    pH                     0.019506
    sulphates              0.038485
    alcohol                0.444319
    quality                1.000000
    Name: quality, dtype: float64

```python
wine_corr.loc[wine_corr["quality"] > 0, "quality"]
```

    citric_acid            0.085532
    free_sulfur_dioxide    0.055463
    pH                     0.019506
    sulphates              0.038485
    alcohol                0.444319
    quality                1.000000
    Name: quality, dtype: float64

```python
red_sample = wine.loc[wine["type"] == "red", :]
white_sample = wine.loc[wine["type"] == "white", :]
```

```python
red_idx = np.random.choice(red_sample.index, replace=True, size=200)
white_idx = np.random.choice(white_sample.index, replace=True, size=200)
```

```python
red_idx[:5]
```

    array([ 571,  608, 1020, 1292,  987], dtype=int64)

```python
red_sample.loc[red_idx, ].head(3)
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>571</th>
      <td>6.2</td>
      <td>0.36</td>
      <td>0.24</td>
      <td>2.2</td>
      <td>0.095</td>
      <td>19.0</td>
      <td>42.0</td>
      <td>0.99460</td>
      <td>3.57</td>
      <td>0.57</td>
      <td>11.7</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>608</th>
      <td>10.1</td>
      <td>0.65</td>
      <td>0.37</td>
      <td>5.1</td>
      <td>0.110</td>
      <td>11.0</td>
      <td>65.0</td>
      <td>1.00260</td>
      <td>3.32</td>
      <td>0.64</td>
      <td>10.4</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>11.3</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>2.4</td>
      <td>0.123</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.99642</td>
      <td>3.20</td>
      <td>0.53</td>
      <td>11.9</td>
      <td>6</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine_sample = red_sample.loc[red_idx, ].append(white_sample.loc[white_idx,])
wine_sample.head()
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
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>571</th>
      <td>6.2</td>
      <td>0.360</td>
      <td>0.24</td>
      <td>2.2</td>
      <td>0.095</td>
      <td>19.0</td>
      <td>42.0</td>
      <td>0.99460</td>
      <td>3.57</td>
      <td>0.57</td>
      <td>11.7</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>608</th>
      <td>10.1</td>
      <td>0.650</td>
      <td>0.37</td>
      <td>5.1</td>
      <td>0.110</td>
      <td>11.0</td>
      <td>65.0</td>
      <td>1.00260</td>
      <td>3.32</td>
      <td>0.64</td>
      <td>10.4</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1020</th>
      <td>11.3</td>
      <td>0.360</td>
      <td>0.66</td>
      <td>2.4</td>
      <td>0.123</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.99642</td>
      <td>3.20</td>
      <td>0.53</td>
      <td>11.9</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>5.9</td>
      <td>0.395</td>
      <td>0.13</td>
      <td>2.4</td>
      <td>0.056</td>
      <td>14.0</td>
      <td>28.0</td>
      <td>0.99362</td>
      <td>3.62</td>
      <td>0.67</td>
      <td>12.4</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>987</th>
      <td>7.1</td>
      <td>0.360</td>
      <td>0.30</td>
      <td>1.6</td>
      <td>0.080</td>
      <td>35.0</td>
      <td>70.0</td>
      <td>0.99693</td>
      <td>3.44</td>
      <td>0.50</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine_sample.shape
```

    (400, 13)

```python
wine_sample.columns
```

    Index(['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
           'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality', 'type'],
          dtype='object')

```python
wine_sample.reset_index(inplace=True)
```

```python
wine_sample.head(3)
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
      <th>index</th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>571</td>
      <td>6.2</td>
      <td>0.36</td>
      <td>0.24</td>
      <td>2.2</td>
      <td>0.095</td>
      <td>19.0</td>
      <td>42.0</td>
      <td>0.99460</td>
      <td>3.57</td>
      <td>0.57</td>
      <td>11.7</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>608</td>
      <td>10.1</td>
      <td>0.65</td>
      <td>0.37</td>
      <td>5.1</td>
      <td>0.110</td>
      <td>11.0</td>
      <td>65.0</td>
      <td>1.00260</td>
      <td>3.32</td>
      <td>0.64</td>
      <td>10.4</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1020</td>
      <td>11.3</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>2.4</td>
      <td>0.123</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>0.99642</td>
      <td>3.20</td>
      <td>0.53</td>
      <td>11.9</td>
      <td>6</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>

```python
sns.set_style("dark")
sns.pairplot(wine_sample, vars=["quality","alcohol","residual_sugar"])
plt.show()
```

![png](output_56_0.png)

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model
```

    LinearRegression()

```python
wine.shape
```

    (6497, 13)

```python
X = wine.drop(["type","quality"], axis=1)
X.shape
```

    (6497, 11)

```python
y = wine.quality
y.shape
```

    (6497,)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

    ((4872, 11), (1625, 11), (4872,), (1625,))

```python
model.fit(X_train, y_train)
```

    LinearRegression()

```python
model.coef_
```

    array([ 5.55618691e-02, -1.29945358e+00, -1.08205046e-01,  4.52070539e-02,
           -3.95901596e-01,  5.76479819e-03, -2.47760359e-03, -5.30023471e+01,
            3.50283862e-01,  7.49149475e-01,  2.78530060e-01])

```python
model.intercept_
```

    54.058003854665195

```python
y_pred = model.predict(X_test)
```

```python
y_pred[:5]
```

    array([5.65651454, 5.48576725, 5.5250081 , 5.59799182, 5.0364669 ])

```python
y_test[:5]
```

    2173    5
    4410    5
    1865    6
    1518    5
    2162    6
    Name: quality, dtype: int64

### I 모델 저장 )

```python
import pickle
save_model = pickle.dumps(model)
```

```python
save_model
```

    b'\x80\x04\x95g\x03\x00\x00\x00\x00\x00\x00\x8c\x1asklearn.linear_model._base\x94\x8c\x10LinearRegression\x94\x93\x94)\x81\x94}\x94(\x8c\rfit_intercept\x94\x88\x8c\tnormalize\x94\x8c\ndeprecated\x94\x8c\x06copy_X\x94\x88\x8c\x06n_jobs\x94N\x8c\x08positive\x94\x89\x8c\x11feature_names_in_\x94\x8c\x15numpy.core.multiarray\x94\x8c\x0c_reconstruct\x94\x93\x94\x8c\x05numpy\x94\x8c\x07ndarray\x94\x93\x94K\x00\x85\x94C\x01b\x94\x87\x94R\x94(K\x01K\x0b\x85\x94h\x0f\x8c\x05dtype\x94\x93\x94\x8c\x02O8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01|\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?t\x94b\x89]\x94(\x8c\rfixed_acidity\x94\x8c\x10volatile_acidity\x94\x8c\x0bcitric_acid\x94\x8c\x0eresidual_sugar\x94\x8c\tchlorides\x94\x8c\x13free_sulfur_dioxide\x94\x8c\x14total_sulfur_dioxide\x94\x8c\x07density\x94\x8c\x02pH\x94\x8c\tsulphates\x94\x8c\x07alcohol\x94et\x94b\x8c\x0en_features_in_\x94K\x0b\x8c\x05coef_\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x0b\x85\x94h\x18\x8c\x02f8\x94\x89\x88\x87\x94R\x94(K\x03\x8c\x01<\x94NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00t\x94b\x89CX(\xc2\xa0\xf5\x9ar\xac?\xc8\x99|\xd6\x8f\xca\xf4\xbf\xf1\xf3\xd3mS\xb3\xbb\xbf\xf5\x16\xd9\x03a%\xa7?\xa2\x92x\xa5sV\xd9\xbf@\x1b\xdd:\xd4\x9cw?\x00\xda\nM\xe9Kd\xbf\xd7\xf6\r\xe9L\x80J\xc0Z\x8d\x93\x00\rk\xd6?\x051\tR\x08\xf9\xe7?\xa7\xdf\x8b\xbeo\xd3\xd1?\x94t\x94b\x8c\t_residues\x94h\x0c\x8c\x06scalar\x94\x93\x94h3C\x08>\x0c\xf7Z8\xbc\xa4@\x94\x86\x94R\x94\x8c\x05rank_\x94K\x0b\x8c\tsingular_\x94h\x0eh\x11K\x00\x85\x94h\x13\x87\x94R\x94(K\x01K\x0b\x85\x94h3\x89CX\x1b\x1f\xae\xc1\xf7\xbf\xaf@\xb1\x0ehI\x1bb\x8a@\xc2\xd3\xbc\x1c\xc0\xe7q@4\xa4d\x9dl)V@\x8f\xf2\xdb\x07\xdc\x0cR@\x15\x034\xf0\x04\xb6(@s\xa1\xe8\xd0\xc8/$@\xdb\xc1\x07\x89\n\xe3 @<\xc3w>0\xb7\x1c@\x08\x8d\x813?\xe6\xff?4\xd3jl\x92U\xab?\x94t\x94b\x8c\nintercept_\x94h:h3C\x08\x1bj\x99\xabl\x07K@\x94\x86\x94R\x94\x8c\x10_sklearn_version\x94\x8c\x051.0.2\x94ub.'

```python
def my_rmse(y_real, y_pred):
    return np.sqrt(np.mean((y_real - y_pred)**2))
```

```python
np.sqrt(np.mean((y_test - y_pred)**2))
```

    0.7254581077541975

```python
np.round(my_rmse(y_test, y_pred),2)
```

    0.73

```python
from sklearn.metrics import mean_squared_error

np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
```

    0.73

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

    LinearRegression()

```python
model
```

    LinearRegression()

#### 3. 머신러닝 모델저장 복원

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True) # 데이터 셋

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train) # fit
```

    C:\Users\ehdal\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(





    LogisticRegression(random_state=0)

### ( 모델 저장 )

```python
import pickle
saved_model = pickle.dumps(lr)
```

```python
new_data = [[1.5, 1.3, 1.7, 1.6]]
```

```python
lr_from_pickle = pickle.loads(saved_model)
```

```python
lr_from_pickle.predict(new_data)
```

    array([0])

### ( 모델을 외부 화일로 저장 )

```python
import joblib
joblib.dump(lr, "c:/test/iris_lr.pkl")
```

    ['c:/test/iris_lr.pkl']

### ( 외부 화일 모델 활용 )

```python
read_pkl = joblib.load("c:/test/iris_lr.pkl")
```

```python
read_pkl.predict(new_data)
```

    array([0])

```python
new_data = [[1.5, 1.3, 1.7, 1.6]]

# 메모리 저장
import pickle
saved_model = pickle.dumps(lr)
lr_from_pickle = pickle.loads(saved_model)
lr_from_pickle.predict(new_data)

# 외부 화일 저장
import joblib
joblib.dump(lr, "c:/test/iris_lr.pkl")
read_pkl = joblib.load("c:/test/iris_lr.pkl")
read_pkl.predict(new_data)
```

    array([0])

```python

```

#### 4. 머신러닝 영화추천엔진

```python
## 1. 데이터 준비
# http://files.grouplens.org/datasets/movielens/ml-100k.zip

import pandas as pd

df = pd.read_csv("c:/test/ml-100k/u.data", sep="\t", header=None)
df.columns = ["user_id","item_id","rating","timestamp"]
df.head(3)
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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100000 entries, 0 to 99999
    Data columns (total 4 columns):
     #   Column     Non-Null Count   Dtype
    ---  ------     --------------   -----
     0   user_id    100000 non-null  int64
     1   item_id    100000 non-null  int64
     2   rating     100000 non-null  int64
     3   timestamp  100000 non-null  int64
    dtypes: int64(4)
    memory usage: 3.1 MB

```python
df["user_id"].value_counts()
```

    405    737
    655    685
    13     636
    450    540
    276    518
          ...
    441     20
    36      20
    812     20
    895     20
    93      20
    Name: user_id, Length: 943, dtype: int64

```python
rp = df.pivot_table(columns=["item_id"], index=["user_id"], values="rating")
rp = rp.fillna(0)
A = rp.values
A.shape
```

    (943, 1682)

```python
df["item_id"].value_counts()
```

    50      583
    258     509
    100     508
    181     507
    294     485
           ...
    852       1
    1505      1
    1653      1
    1452      1
    1641      1
    Name: item_id, Length: 1682, dtype: int64

```python
rp.head(6)
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
      <th>item_id</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>1673</th>
      <th>1674</th>
      <th>1675</th>
      <th>1676</th>
      <th>1677</th>
      <th>1678</th>
      <th>1679</th>
      <th>1680</th>
      <th>1681</th>
      <th>1682</th>
    </tr>
    <tr>
      <th>user_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 1682 columns</p>
</div>

```python
df.groupby(["rating"])
```

    <pandas.core.groupby.generic.DataFrameGroupBy object at 0x000001C011489A60>

```python
df.groupby(["rating"])[["user_id"]].count()
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
      <th>user_id</th>
    </tr>
    <tr>
      <th>rating</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6110</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27145</td>
    </tr>
    <tr>
      <th>4</th>
      <td>34174</td>
    </tr>
    <tr>
      <th>5</th>
      <td>21201</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.groupby(["item_id"])[["user_id"]].count()
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
      <th>user_id</th>
    </tr>
    <tr>
      <th>item_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>452</td>
    </tr>
    <tr>
      <th>2</th>
      <td>131</td>
    </tr>
    <tr>
      <th>3</th>
      <td>90</td>
    </tr>
    <tr>
      <th>4</th>
      <td>209</td>
    </tr>
    <tr>
      <th>5</th>
      <td>86</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1678</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1679</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1680</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1681</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1682</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1682 rows × 1 columns</p>
</div>

```python
n_users = df.user_id.unique().shape[0]; n_users
```

    943

```python
n_items = df.item_id.unique().shape[0];n_items
```

    1682

```python
import numpy as np
ratings = np.zeros((n_users, n_items));ratings
```

    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])

```python
for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

ratings
```

    array([[5., 3., 4., ..., 0., 0., 0.],
           [4., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [5., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 5., 0., ..., 0., 0., 0.]])

```python
df.head(3)
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
      <th>user_id</th>
      <th>item_id</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>196</td>
      <td>242</td>
      <td>3</td>
      <td>881250949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>186</td>
      <td>302</td>
      <td>3</td>
      <td>891717742</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>377</td>
      <td>1</td>
      <td>878887116</td>
    </tr>
  </tbody>
</table>
</div>

```python
rp.values
```

    array([[5., 3., 4., ..., 0., 0., 0.],
           [4., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [5., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 5., 0., ..., 0., 0., 0.]])

```python
from sklearn.model_selection import train_test_split

ratings_train, ratings_test = train_test_split(ratings, test_size=0.33)
ratings_train.shape, ratings_test.shape
```

    ((631, 1682), (312, 1682))

```python
# 유사도 계산
from sklearn.metrics.pairwise import cosine_distances
cosine_distances(ratings_train)
```

    array([[0.        , 0.63557438, 0.89853146, ..., 0.57631529, 1.        ,
            0.96101588],
           [0.63557438, 0.        , 0.91704994, ..., 0.64915587, 1.        ,
            0.99153709],
           [0.89853146, 0.91704994, 0.        , ..., 0.90900558, 0.80104903,
            0.92739854],
           ...,
           [0.57631529, 0.64915587, 0.90900558, ..., 0.        , 1.        ,
            0.99396961],
           [1.        , 1.        , 0.80104903, ..., 1.        , 0.        ,
            0.90870891],
           [0.96101588, 0.99153709, 0.92739854, ..., 0.99396961, 0.90870891,
            0.        ]])

```python
distances = 1 - cosine_distances(ratings_train)
distances
```

    array([[1.        , 0.36442562, 0.10146854, ..., 0.42368471, 0.        ,
            0.03898412],
           [0.36442562, 1.        , 0.08295006, ..., 0.35084413, 0.        ,
            0.00846291],
           [0.10146854, 0.08295006, 1.        , ..., 0.09099442, 0.19895097,
            0.07260146],
           ...,
           [0.42368471, 0.35084413, 0.09099442, ..., 1.        , 0.        ,
            0.00603039],
           [0.        , 0.        , 0.19895097, ..., 0.        , 1.        ,
            0.09129109],
           [0.03898412, 0.00846291, 0.07260146, ..., 0.00603039, 0.09129109,
            1.        ]])

```python
user_pred = distances.dot(ratings_train) / np.array([np.abs(distances).sum(axis=1)]).T
```

```python
from sklearn.metrics import mean_squared_error
def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
```

```python
np.sqrt(get_mse(user_pred, ratings_train))
```

    2.792910731149253

```python
user_pred[ratings_train.nonzero()].flatten()
```

    array([0.43765419, 2.68617948, 0.45292083, ..., 0.06294556, 0.13734603,
           0.10936382])

```python

```

#### 5. LabelEncoder

```python

```

```python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
```

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```

```python
le.fit([1,2,2,6])
```

    LabelEncoder()

```python
le.classes_
```

    array([1, 2, 6])

```python
le.transform([1,1,2,6])
```

    array([0, 0, 1, 2], dtype=int64)

```python
le.inverse_transform([0,0,1,2])
```

    array([1, 1, 2, 6])

```python
le.fit(["paris","paris","tokyo", "amsterdam" ])
```

    LabelEncoder()

```python
list(le.classes_)
```

    ['amsterdam', 'paris', 'tokyo']

```python
le.transform(["tokyo","tokyo","paris"])
```

    array([2, 2, 1])

```python
le.inverse_transform([2, 2, 1])
```

    array(['tokyo', 'tokyo', 'paris'], dtype='<U9')

```python

```

#### 6. MinMaxScaler

```python
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
```

    MinMaxScaler()

```python
print(scaler.data_max_)
```

    [ 1. 18.]

```python
print(scaler.transform(data))
```

    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]

```python
 print(scaler.transform([[2, 2]]))
```

    [[1.5 0. ]]

```python

```

```python

```
