---
layout: single
title: "2202.11.14.파이썬 기초 데이터처리 기술(9)"
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

1. 데이터처리
2. matplotlib\_시각화
3. numpy_quickstart

### ( 데이터 처리 실습 )

1. 머신러닝\_기초
2. 머신러닝\_scikit_learn_quickstart

#### 1. 머신러닝\_기초

```python
import numpy as np
rs = np.random.RandomState(10)

x =10 * rs.rand(5)
y = 2 * x - 1 * rs.rand(5)

x.shape, y.shape
```

    ((5,), (5,))

```python
x
```

    array([7.71320643, 0.20751949, 6.33648235, 7.48803883, 4.98507012])

```python
y
```

    array([15.20161622,  0.21697612, 11.91243399, 14.80696681,  9.88180043])

```python
x.shape
```

    (5,)

```python
x.reshape(-1,1)
```

    array([[7.71320643],
           [0.20751949],
           [6.33648235],
           [7.48803883],
           [4.98507012]])

```python
x.reshape(-1,1).shape
```

    (5, 1)

```python
import seaborn as sns
iris_df = sns.load_dataset("iris")
iris_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype
    ---  ------        --------------  -----
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB

```python
iris_df.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>

```python
X = iris_df.drop("species", axis=1);X.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>

```python
X.shape
```

    (150, 4)

```python
y = iris_df["species"] ; y.shape
```

    (150,)

```python
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
```

    sklearn.utils.Bunch

```python
iris.keys()
```

    dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

```python
iris.feature_names
```

    ['sepal length (cm)',
     'sepal width (cm)',
     'petal length (cm)',
     'petal width (cm)']

```python
X = iris.data
y = iris.target
```

```python
y
```

    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

### ( 실습 )

1. df iris에서 species를 변환 하기
2. setosa(0), versicolor(1), virginica(2)로 변환하여
3. new_species 변수를 만들시오

```python
import seaborn as sns
iris_df = sns.load_dataset("iris")
iris_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype
    ---  ------        --------------  -----
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB

```python
iris_df.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>

```python
iris_df["new_01"] = ""
for i in range(0,len(iris_df)):
#     print(iris_df.loc[i, "species"])
#     print(iris_df.iloc[i, 4])
    if iris_df.loc[i, "species"] == "setosa":
        iris_df.loc[i, "new_01"] = 0
    elif iris_df.loc[i, "species"] == "versicolor":
        iris_df.loc[i, "new_01"] = 1
    else:
        iris_df.loc[i, "new_01"] = 2
```

```python
iris_df["new_01"] = ""
for i in range(0,len(iris_df)):
    if iris_df.loc[i, "species"] == "setosa":
        iris_df.loc[i, "new_01"] = 0
    elif iris_df.loc[i, "species"] == "versicolor":
        iris_df.loc[i, "new_01"] = 1
    else:
        iris_df.loc[i, "new_01"] = 2
```

```python
iris_df.loc[iris_df["species"] == "setosa","new_02"] = 0
iris_df.loc[iris_df["species"] == "versicolor","new_02"] = 1
iris_df.loc[iris_df["species"] == "virginica","new_02"] = 2

#iris_df.tail(3)
```

```python
species_mapping = {"setosa":0, "versicolor":1, "virginica":2}
iris_df["new_03"] = iris_df["species"].map(species_mapping)
iris_df.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
      <th>new_01</th>
      <th>new_02</th>
      <th>new_03</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
y1 = iris_df["new_03"]
y1.values
```

    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=int64)

```python
y = iris.target
#y
```

```python
X1 = iris_df.iloc[:,:4].values
#X1[:5]
```

```python
X = iris.data;X[:5]
```

    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2]])

```python

```

#### 2. 머신러닝\_scikit_learn_quickstart

# https://scikit-learn.org/stable/getting_started.html

```python
# 모델 선택 및 사용 준비
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
```

```python
# 데이터 셋
X = [[1, 2, 3],
     [11, 12, 13]]
y = [0, 1]
```

```python
# 모델 훈련
clf.fit(X, y)
```

    RandomForestClassifier(random_state=0)

```python
# 모델로 값 예측
clf.predict(X)
```

    array([0, 1])

```python
X1 = [[4,5,6], [14,15,16]]
clf.predict(X1)
```

    array([0, 1])

```python
X2 = [[4,5,6], [6,7,8]]
clf.predict(X2)
```

    array([0, 0])

### ( Transformers and pre-processors )

```python
from sklearn.preprocessing import StandardScaler
```

```python
X = [[0, 15],
     [1, -10]]
```

```python
StandardScaler().fit(X).transform(X)
```

    array([[-1.,  1.],
           [ 1., -1.]])

```python
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
```

```python
from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
```

```python
print(scaler.fit(data))
```

    StandardScaler()

```python
scaler.mean_
```

    array([0.5, 0.5])

```python
scaler.transform(data)
```

    array([[-1., -1.],
           [-1., -1.],
           [ 1.,  1.],
           [ 1.,  1.]])

```python
scaler.transform([[2,2]])
```

    array([[3., 3.]])

### ( Pipelines: chaining pre-processors and estimators )

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

```python
#  pipeline 객체 생성
pipe = make_pipeline(
       StandardScaler(),
       LogisticRegression())
```

```python
# 데이터 셋
X, y = load_iris(return_X_y=True)
```

```python
X[:5]
```

    array([[5.1, 3.5, 1.4, 0.2],
           [4.9, 3. , 1.4, 0.2],
           [4.7, 3.2, 1.3, 0.2],
           [4.6, 3.1, 1.5, 0.2],
           [5. , 3.6, 1.4, 0.2]])

```python
y[:5]
```

    array([0, 0, 0, 0, 0])

```python
import seaborn as sns
iris = sns.load_dataset("iris")
```

```python
# train and test set 만들기
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size= 0.85,
                                                    random_state=123)

print(X.shape, y.shape)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (150, 4) (150,)
    (127, 4) (23, 4) (127,) (23,)

```python
# fit the pipeline
pipe.fit(X_train, y_train)
```

    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression())])

```python
pipe.predict(X_test)
```

    array([1, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 0, 1,
           0])

```python
y_test
```

    array([1, 2, 2, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 2, 0, 0, 1, 0, 0, 2,
           0])

```python
# test 데이터 예측 결과 채점
accuracy_score(pipe.predict(X_test),y_test )
```

    0.9565217391304348

```python
iris.head(3)
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>

```python
iris.iloc[120]
```

    sepal_length          6.9
    sepal_width           3.2
    petal_length          5.7
    petal_width           2.3
    species         virginica
    Name: 120, dtype: object

```python
new_data = [[1.5, 1.3, 1.7, 1.6]]

pipe.predict(new_data)
```

    array([0])

```python
MY_RANDOM = 1234
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#  pipeline 객체 생성
pipe = make_pipeline(
       StandardScaler(),
       LogisticRegression())

X, y = load_iris(return_X_y=True) # 데이터 셋

# train and test set 만들기
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size= 0.85,
                                                    random_state=123)

pipe.fit(X_train, y_train) # fit the pipeline
accuracy_score(pipe.predict(X_test),y_test ) # test 데이터 예측 결과 채점

pipe.predict(new_data) # 모델 확정후 실 데이터 적용
```

    array([0])

### ( Model evaluation )

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
```

```python
MY_RANDOM = 123
X, y = make_regression(n_samples=1000, random_state=MY_RANDOM)
```

```python
lr = LinearRegression()
```

```python
result = cross_validate(lr, X, y)
```

```python
result
```

    {'fit_time': array([0.06217909, 0.00362563, 0.00201988, 0.00303578, 0.0025084 ]),
     'score_time': array([0.0010004 , 0.        , 0.00107765, 0.        , 0.00100374]),
     'test_score': array([1., 1., 1., 1., 1.])}

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
```

```python
X, y = fetch_california_housing(return_X_y=True)
```

```python
y
```

    array([4.526, 3.585, 3.521, ..., 0.923, 0.847, 0.894])

```python
X[0]
```

    array([   8.3252    ,   41.        ,    6.98412698,    1.02380952,
            322.        ,    2.55555556,   37.88      , -122.23      ])

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

```python
param_distributions = {'n_estimators': randint(1, 5),
                       'max_depth': randint(5, 10)}
```

```python
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
                            n_iter=5,
                            param_distributions=param_distributions,
                            random_state=0)
```

```python
search.fit(X_train, y_train)
```

    RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                       param_distributions={'max_depth': <scipy.stats._distn_infrastructure.rv_discrete_frozen object at 0x00000212537691F0>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_discrete_frozen object at 0x0000021252FC14C0>},
                       random_state=0)

```python
search.best_params_
```

    {'max_depth': 9, 'n_estimators': 4}

```python
search.score(X_test, y_test)
```

    0.735363411343253

```python
# base line ->  0.55 ... 65%??
```
