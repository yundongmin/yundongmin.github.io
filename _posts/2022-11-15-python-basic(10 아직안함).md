---
layout: single
title: "2202.11.15.파이썬 기초 데이터처리 기술(10)"
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

#### 1. 데이터처리

### ( 데이터 처리 실습 )

1. 데이터셋-winequality 데이터셋
2. matplotlib 시각화
3. numpy_quickstart

```python
import pandas as pd

red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_wine = pd.read_csv(red_url, sep=";")
white_wine = pd.read_csv(white_url, sep=";")
```

```python
red_wine.head(3)
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
    </tr>
  </tbody>
</table>
</div>

```python
white_wine.head(3)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>

```python
red_wine["type"] = "red"
white_wine["type"] = "white"

wine_df = pd.concat([red_wine, white_wine])

print(red_wine.shape, " ** ", white_wine.shape, " ** ", wine_df.shape)
```

    (1599, 13)  **  (4898, 13)  **  (6497, 13)

```python
# row 1598 ~ 1603 데이터 확인
wine_df.iloc[1596:1603,:]
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
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.270</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.00100</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.300</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.99400</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.280</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.99510</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>0.230</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.99560</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>

```python
wine_df = pd.concat([red_wine, white_wine], ignore_index=True)
wine_df.iloc[1596:1603,:]
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
    <tr>
      <th>1599</th>
      <td>7.0</td>
      <td>0.270</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.00100</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1600</th>
      <td>6.3</td>
      <td>0.300</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.99400</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1601</th>
      <td>8.1</td>
      <td>0.280</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.99510</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1602</th>
      <td>7.2</td>
      <td>0.230</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.99560</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>

```python
new_df = red_wine.append(white_wine)
new_df.shape
```

    C:\Users\ehdal\AppData\Local\Temp\ipykernel_136404\3978243007.py:1: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      new_df = red_wine.append(white_wine)





    (6497, 13)

### ( 실습 2)

2. wine_df 필수 탐색
   - 처음 5건의 데이터는 어떻게 생겼나?
   - 마지막 3건의 데이터는 어떻게 생겼나?
   - 컬럼 이름은?
   - 몇행 몇열로 구성되어 있나
   - 전체 컬럼 구성은?
   - 결측치는?
   - 품질별로 몇건씩 구성되어 있나?
     - 전체는?
     - red는?
     - wihte는?
   - 요약 통계량은?

```python
wine_df.head(5)
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
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
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
  </tbody>
</table>
</div>

```python
wine_df.tail(3)
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
      <th>6494</th>
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
      <th>6495</th>
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
      <th>6496</th>
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
wine_df.columns
```

    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality', 'type'],
          dtype='object')

```python
wine_df.shape
```

    (6497, 13)

```python
wine_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype
    ---  ------                --------------  -----
     0   fixed acidity         6497 non-null   float64
     1   volatile acidity      6497 non-null   float64
     2   citric acid           6497 non-null   float64
     3   residual sugar        6497 non-null   float64
     4   chlorides             6497 non-null   float64
     5   free sulfur dioxide   6497 non-null   float64
     6   total sulfur dioxide  6497 non-null   float64
     7   density               6497 non-null   float64
     8   pH                    6497 non-null   float64
     9   sulphates             6497 non-null   float64
     10  alcohol               6497 non-null   float64
     11  quality               6497 non-null   int64
     12  type                  6497 non-null   object
    dtypes: float64(11), int64(1), object(1)
    memory usage: 660.0+ KB

```python
wine_df.isnull().sum().sum()
```

    0

```python
wine_df["quality"].value_counts().sort_index()
```

    3      30
    4     216
    5    2138
    6    2836
    7    1079
    8     193
    9       5
    Name: quality, dtype: int64

```python
wine_df[wine_df["type"] == "red"]["quality"].value_counts().sort_index()
```

    3     10
    4     53
    5    681
    6    638
    7    199
    8     18
    Name: quality, dtype: int64

```python
wine_df[wine_df["type"] == "white"]["quality"].value_counts().sort_index()
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
wine_df.query("type == 'white'")["quality"].value_counts().sort_index()
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
str_expr = "type == 'white'"
wine_df.query(str_expr)["quality"].value_counts().sort_index()
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
wine_df.describe()
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
wine_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype
    ---  ------                --------------  -----
     0   fixed acidity         6497 non-null   float64
     1   volatile acidity      6497 non-null   float64
     2   citric acid           6497 non-null   float64
     3   residual sugar        6497 non-null   float64
     4   chlorides             6497 non-null   float64
     5   free sulfur dioxide   6497 non-null   float64
     6   total sulfur dioxide  6497 non-null   float64
     7   density               6497 non-null   float64
     8   pH                    6497 non-null   float64
     9   sulphates             6497 non-null   float64
     10  alcohol               6497 non-null   float64
     11  quality               6497 non-null   int64
     12  type                  6497 non-null   object
    dtypes: float64(11), int64(1), object(1)
    memory usage: 660.0+ KB

### ( 실습 히스토그램 histogram 그리기 )

1. 도수분포표를 그래프로
   - 가로축은 계급
   - 세로축은 도수
2. 컬럼 -> "fixed acidity"

```python
wine_df["fixed acidity"].describe()
```

    count    6497.000000
    mean        7.215307
    std         1.296434
    min         3.800000
    25%         6.400000
    50%         7.000000
    75%         7.700000
    max        15.900000
    Name: fixed acidity, dtype: float64

```python
import matplotlib.pyplot as plt

plt.hist(wine_df["fixed acidity"])
plt.show()
```

![png](output_24_0.png)

```python
plt.hist(wine_df["fixed acidity"],alpha=0.4, bins=10, rwidth=0.3, color="red")
plt.show()
```

![png](output_25_0.png)

```python
def hist_chart(feature):
    plt.hist(wine_df[feature],alpha=0.4,
             bins=20, rwidth=0.1, color="red")
    plt.show()
```

```python
hist_chart("fixed acidity")
```

![png](output_27_0.png)

### ( 산점도, Scatter plot )

1. x="density", y="chlorides"

```python
plt.scatter(x="density",y="chlorides", data=wine_df)
```

    <matplotlib.collections.PathCollection at 0x287507067c0>

![png](output_29_1.png)

```python
wine_df.plot.scatter(x="density",y="chlorides", grid=True, color="green")
```

    <AxesSubplot:xlabel='density', ylabel='chlorides'>

![png](output_30_1.png)

#### 그림 깨지는데 해결 방법을 모르겠네요....

#### 2. matplotlib\_시각화

# https://matplotlib.org/stable/gallery/

```python
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:

# 외부 화일에서 데이터를 읽어서 df
# 데이터로 조작해서

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
```

![png](output_1_0.png)

```python
import numpy as np
import matplotlib.pyplot as plt


# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute areas and colors
N = 150
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
```

![png](output_2_0.png)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
x = [1,2,3,4]
y = [1,4,2,3]

plt.plot(x,y)
plt.show()
```

![png](output_4_0.png)

```python
year = [1950, 1960, 1970, 1980, 1990, 2000]
pop =[32, 38, 42, 47, 49, 51]
plt.plot(year, pop)
```

    [<matplotlib.lines.Line2D at 0x24643c6f3a0>]

![png](output_5_1.png)

```python
import matplotlib.pyplot as plt

t = [2005,2006,2007,2008,2009,
      2010,2011,2012,2013,2014]

temperature = [11.7, 11.5, 12.1, 12.5, 12.6,
               10.8, 11, 12.2, 11.6, 13.1]
dewpoint = [3, 4, 4, 3, 2, 4, 3, 3, 2, 3]

plt.plot(t, temperature, "red")
plt.plot(t, dewpoint, "blue")
plt.xlabel("Date")
plt.title("Temperatuer & Dew Point")
```

    Text(0.5, 1.0, 'Temperatuer & Dew Point')

![png](output_6_1.png)

```python
import matplotlib.pyplot as plt
t = [2005,2006,2007,2008,2009,
      2010,2011,2012,2013,2014]
temperature = [11.7, 11.5, 12.1, 12.5, 12.6,
                      10.8, 11, 12.2, 11.6, 13.1]
dewpoint = [3, 4, 4, 3, 2, 4, 3, 3, 2, 3]

plt.axes([0.05, 0.05, 0.425, 0.9])
plt.plot(t, temperature, "red")
plt.xlabel("Date")
plt.title("Temperature")

plt.axes([0.525, 0.05, 0.425, 0.9])
plt.plot(t, dewpoint, "blue")
plt.xlabel("Data")
plt.title("Dew Point")
```

    Text(0.5, 1.0, 'Dew Point')

![png](output_7_1.png)

```python
import matplotlib.pyplot as plt
t = [2005,2006,2007,2008,2009,
      2010,2011,2012,2013,2014]
temperature = [11.7, 11.5, 12.1, 12.5, 12.6,
                      10.8, 11, 12.2, 11.6, 13.1]
dewpoint = [3, 4, 4, 3, 2, 4, 3, 3, 2, 3]

i =1
plt.subplot(2,1,i)
plt.plot(t, temperature, "red")
plt.xlabel("Date")
plt.title("Temp")

j = 2
plt.subplot(2,1,j)
plt.plot(t, dewpoint, "blue")
plt.xlabel("Date")
plt.title("Dew Point")

plt.tight_layout()
```

![png](output_8_0.png)

```python
import matplotlib.pyplot as plt
t = [2005,2006,2007,2008,2009,
      2010,2011,2012,2013,2014]
temperature = [11.7, 11.5, 12.1, 12.5, 12.6,
                      10.8, 11, 12.2, 11.6, 13.1]
dewpoint = [3, 4, 4, 3, 2, 4, 3, 3, 2, 3]

i =1
plt.subplot(2,2,i)
plt.plot(t, temperature, "red")
plt.xlabel("Date")
plt.title("Temp")

j = 2
plt.subplot(2,2,j)
plt.plot(t, dewpoint, "blue")
plt.xlabel("Date")
plt.title("Dew Point")

plt.tight_layout()
```

![png](output_9_0.png)

```python
x = list(range(0,11)); x
plt.plot(x)
plt.ylabel("Y축")
```

    Text(0, 0.5, 'Y축')



    C:\Users\ehdal\anaconda3\lib\site-packages\IPython\core\pylabtools.py:151: UserWarning: Glyph 52629 (\N{HANGUL SYLLABLE CUG}) missing from current font.
      fig.canvas.print_figure(bytes_io, **kw)

![png](output_10_2.png)

```python
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame({"a":[2005,2006,2007,2008,2009,2010,2011,2012,2013,2014],
                   "b":[11.7, 11.5, 12.1, 12.5, 12.6,10.8, 11, 12.2, 11.6, 13.1],
                   "c":[3, 4, 4, 3, 2, 4, 3, 3, 2, 3]})
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>11.7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>11.5</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007</td>
      <td>12.1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>

```python
plt.subplot(2,1,1)
plt.plot(df["a"],df["b"], "red")
plt.xlabel("Date")
plt.title("Temp")

plt.subplot(2,1,2)
plt.plot(df["a"],df["c"], "blue")
plt.xlabel("Date")
plt.title("Dew Point")

plt.tight_layout()
```

![png](output_12_0.png)

```python
df.to_csv("c:/test/temp.csv")
```

```python
df = pd.read_csv("c:/test/temp.csv")

plt.subplot(2,1,1)
plt.plot(df["a"],df["b"], "red")
plt.xlabel("Date")
plt.title("Temp")

plt.subplot(2,1,2)
plt.plot(df["a"],df["c"], "blue")
plt.xlabel("Date")
plt.title("Dew Point")

plt.tight_layout()
```

![png](output_14_0.png)

```python
np.random.seed(13)  # seed the random number generator.

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b');
```

![png](output_15_0.png)

```python
np.random.seed(13)  # seed the random number generator.

data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b');
```

![png](output_16_0.png)

```python
mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# the histogram of the data
n, bins, patches = ax.hist(x, 50, density=True, facecolor='C0', alpha=0.2)

ax.set_xlabel('Length [cm]')
ax.set_ylabel('Probability')
ax.set_title('Aardvark lengths\n (not really)')
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
ax.axis([55, 175, 0, 0.03])
ax.grid(True);
```

![png](output_17_0.png)

```python
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
dates = np.arange(np.datetime64('2021-11-15'), np.datetime64('2021-12-25'),
                  np.timedelta64(1, 'h'))
data = np.cumsum(np.random.randn(len(dates)))
ax.plot(dates, data)

cdf = mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator())
ax.xaxis.set_major_formatter(cdf);
```

![png](output_18_0.png)

```python
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
dates = np.arange(np.datetime64('2021-11-15'), np.datetime64('2021-12-25'),
                  np.timedelta64(1, 'h'))
data = np.cumsum(np.random.randn(len(dates)))
ax.plot(dates, data)
```

    [<matplotlib.lines.Line2D at 0x2464568eeb0>]

![png](output_19_1.png)

```python
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0, 0.3, 0.5)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie(sizes, # 값
        labels=labels, # 분류, 구분, class
        autopct='%1.1f%%', # 값 표시,
        explode=(0.3, 0, 0, 0),
        shadow=True,
        startangle= 90 )
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
```

![png](output_20_0.png)

### (실습. Piechart 그리기 )

1. iris 데이터 셋
   - import seaborn as sns
   - iris = sns.load_dataset("iris")
2. species의분포 파이차트 그리기
   - 분류별 카운트

```python
import seaborn as sns
iris = sns.load_dataset("iris")

iris_index = iris["species"].value_counts().index
iris_values = iris["species"].value_counts().values

labels = iris_index # 품종구분
sizes = iris_values # 품종별 데이터

plt.pie(sizes, # 값
        labels=labels,
        autopct='%1.1f%%', # 값 표시,
        shadow=True,
        explode = (0, 0.1, 0),
        startangle= 90 )
```

    ([<matplotlib.patches.Wedge at 0x246457594f0>,
      <matplotlib.patches.Wedge at 0x24645759e80>,
      <matplotlib.patches.Wedge at 0x24645767850>],
     [Text(-0.9526279613277876, 0.5499999702695114, 'setosa'),
      Text(1.1235210819632121e-07, -1.199999999999995, 'versicolor'),
      Text(0.9526278583383436, 0.5500001486524351, 'virginica')],
     [Text(-0.5196152516333387, 0.29999998378336984, '33.3%'),
      Text(6.553872978118737e-08, -0.699999999999997, '33.3%'),
      Text(0.5196151954572783, 0.3000000810831464, '33.3%')])

![png](output_22_1.png)

labels ... 품종구분 ( setosa, versicolor, virginica )
sizes .. 품종별 데이터 [50, 50, 50]
... value_counts()

```python
# 데이터 셋

import seaborn as sns
iris = sns.load_dataset("iris")
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
iris["species"].value_counts()
```

    setosa        50
    versicolor    50
    virginica     50
    Name: species, dtype: int64

```python
iris_index = iris["species"].value_counts().index
iris_values = iris["species"].value_counts().values
print(iris_index, iris_values)
```

    Index(['setosa', 'versicolor', 'virginica'], dtype='object') [50 50 50]

```python
import seaborn as sns
iris = sns.load_dataset("iris")

iris_index = iris["species"].value_counts().index
iris_values = iris["species"].value_counts().values

plt.pie(iris_values, # 값
        labels=iris_index,
        autopct='%1.1f%%', # 값 표시,
        shadow=True,
        explode = (0, 0.2, 0.1),
        startangle= 90 )
plt.show()
```

![png](output_27_0.png)

### ( 실습. pie chart 그리기 )

1. tips 데이터셋
2. 데이터중 분류형 변수 각각의 파이차트를 그리기

```python
tips = sns.load_dataset("tips")
tips.head(3)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

```python
def chart_pie(feature):

    labels = tips[feature].value_counts().index
    values = tips[feature].value_counts().values

    plt.pie(values, # 값
            labels=labels,
            autopct='%1.1f%%', # 값 표시,
            shadow=True,
            startangle= 90 )
    plt.show()
```

```python
chart_pie("smoker")
```

![png](output_31_0.png)

```python
chart_pie("sex")
```

![png](output_32_0.png)

```python
chart_pie("day")
```

![png](output_33_0.png)

```python
chart_pie("time")
```

![png](output_34_0.png)

```python
chart_pie("size")
```

![png](output_35_0.png)

# 8개 판다스 명령..

```python
tips.head(3)
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

```python
tips.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 244 entries, 0 to 243
    Data columns (total 7 columns):
     #   Column      Non-Null Count  Dtype
    ---  ------      --------------  -----
     0   total_bill  244 non-null    float64
     1   tip         244 non-null    float64
     2   sex         244 non-null    category
     3   smoker      244 non-null    category
     4   day         244 non-null    category
     5   time        244 non-null    category
     6   size        244 non-null    int64
    dtypes: category(4), float64(2), int64(1)
    memory usage: 7.4 KB

```python
# 1. 하나의 변수로 파이 그래프 그리기
index = tips["smoker"].value_counts().index
values = tips["smoker"].value_counts().values

plt.pie(values, # 값
        labels=index,
        autopct='%1.1f%%', # 값 표시,
        shadow=True,
        startangle= 90 )
plt.show()
```

![png](output_39_0.png)

```python
tips.columns
```

    Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size'], dtype='object')

```python
tips = sns.load_dataset("tips")

def pie_chart(feature):
    index = tips[feature].value_counts().index
    values = tips[feature].value_counts().values

    plt.pie(values, # 값
            labels=index,
            autopct='%1.1f%%', # 값 표시,
            shadow=True,
            startangle= 90 )
    plt.show()
```

```python
pie_chart("day") # sex, smoker, day, time, size
```

![png](output_42_0.png)

```python
pie_chart("time") # "size"
```

![png](output_43_0.png)

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

def pie_chart1(ax, feature):
    index = tips[feature].value_counts().index
    values = tips[feature].value_counts().values

    ax.pie(values, # 값
            labels=index,
            autopct='%1.1f%%', # 값 표시,
            shadow=True,
            startangle= 90 )
    plt.show()
```

```python
fig, ax = plt.subplots()
pie_chart1(ax,"day")
```

![png](output_45_0.png)

```python
fig, ax = plt.subplots(2,3)
pie_chart1(ax[1][0],"day")
```

![png](output_46_0.png)

#### 시각화쪽 사진 마크업 언어 안깨지게 하는법 조만간 수정

#### 3. numpy_quickstart

```python
# https://numpy.org/doc/stable/user/quickstart.html
```

```python
import numpy as np
import pandas as pd
```

```python
a = np.arange(15).reshape(3,5)
a
```

    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])

```python
a.shape
```

    (3, 5)

```python
a.ndim
```

    2

```python
a.dtype
```

    dtype('int32')

```python
a.dtype.name
```

    'int32'

```python
a.itemsize
```

    4

```python
a.size
```

    15

```python
from numpy import pi
```

```python
pi
```

    3.141592653589793

```python
x = np.linspace(0, 2 * pi, 100)
```

```python
np.sin(x)
```

    array([ 0.00000000e+00,  6.34239197e-02,  1.26592454e-01,  1.89251244e-01,
            2.51147987e-01,  3.12033446e-01,  3.71662456e-01,  4.29794912e-01,
            4.86196736e-01,  5.40640817e-01,  5.92907929e-01,  6.42787610e-01,
            6.90079011e-01,  7.34591709e-01,  7.76146464e-01,  8.14575952e-01,
            8.49725430e-01,  8.81453363e-01,  9.09631995e-01,  9.34147860e-01,
            9.54902241e-01,  9.71811568e-01,  9.84807753e-01,  9.93838464e-01,
            9.98867339e-01,  9.99874128e-01,  9.96854776e-01,  9.89821442e-01,
            9.78802446e-01,  9.63842159e-01,  9.45000819e-01,  9.22354294e-01,
            8.95993774e-01,  8.66025404e-01,  8.32569855e-01,  7.95761841e-01,
            7.55749574e-01,  7.12694171e-01,  6.66769001e-01,  6.18158986e-01,
            5.67059864e-01,  5.13677392e-01,  4.58226522e-01,  4.00930535e-01,
            3.42020143e-01,  2.81732557e-01,  2.20310533e-01,  1.58001396e-01,
            9.50560433e-02,  3.17279335e-02, -3.17279335e-02, -9.50560433e-02,
           -1.58001396e-01, -2.20310533e-01, -2.81732557e-01, -3.42020143e-01,
           -4.00930535e-01, -4.58226522e-01, -5.13677392e-01, -5.67059864e-01,
           -6.18158986e-01, -6.66769001e-01, -7.12694171e-01, -7.55749574e-01,
           -7.95761841e-01, -8.32569855e-01, -8.66025404e-01, -8.95993774e-01,
           -9.22354294e-01, -9.45000819e-01, -9.63842159e-01, -9.78802446e-01,
           -9.89821442e-01, -9.96854776e-01, -9.99874128e-01, -9.98867339e-01,
           -9.93838464e-01, -9.84807753e-01, -9.71811568e-01, -9.54902241e-01,
           -9.34147860e-01, -9.09631995e-01, -8.81453363e-01, -8.49725430e-01,
           -8.14575952e-01, -7.76146464e-01, -7.34591709e-01, -6.90079011e-01,
           -6.42787610e-01, -5.92907929e-01, -5.40640817e-01, -4.86196736e-01,
           -4.29794912e-01, -3.71662456e-01, -3.12033446e-01, -2.51147987e-01,
           -1.89251244e-01, -1.26592454e-01, -6.34239197e-02, -2.44929360e-16])

```python
b = np.arange(12).reshape(3, 4);b
```

    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])

```python
b.sum(axis=0)
```

    array([12, 15, 18, 21])

```python
b.sum(axis=1)
```

    array([ 6, 22, 38])

```python
b.cumsum(axis=1)
```

    array([[ 0,  1,  3,  6],
           [ 4,  9, 15, 22],
           [ 8, 17, 27, 38]], dtype=int32)

```python
rg = np.random.default_rng(1)
```

```python
rg = np.random.default_rng(3)

a = np.floor(10 * rg.random((3, 4)))
a
```

    array([[0., 2., 8., 5.],
           [0., 4., 4., 1.],
           [7., 1., 3., 5.]])

```python
a.resize((2, 6));a
```

    array([[0., 2., 8., 5., 0., 4.],
           [4., 1., 7., 1., 3., 5.]])

```python
a.reshape(3,4)
```

    array([[0., 2., 8., 5.],
           [0., 4., 4., 1.],
           [7., 1., 3., 5.]])

```python
a.reshape(3,-1)
```

    array([[0., 2., 8., 5.],
           [0., 4., 4., 1.],
           [7., 1., 3., 5.]])

```python
a.reshape(2,3,-1)
```

    array([[[0., 2.],
            [8., 5.],
            [0., 4.]],

           [[4., 1.],
            [7., 1.],
            [3., 5.]]])

```python
a1 = a.reshape(2,-1,2); a1.shape
```

    (2, 3, 2)

```python
a = np.array([4., 2.])
b = np.array([3., 8.])
```

```python
a
```

    array([4., 2.])

```python
b
```

    array([3., 8.])

```python
np.column_stack((a, b))
```

    array([[4., 3.],
           [2., 8.]])

```python
np.hstack((a, b))
```

    array([4., 2., 3., 8.])

```python
from numpy import newaxis
```

```python
a
```

    array([4., 2.])

```python
a[:, newaxis]
```

    array([[4.],
           [2.]])

```python
np.column_stack((a[:, newaxis], b[:, newaxis]))
```

    array([[4., 3.],
           [2., 8.]])

```python
a = np.floor(10 * rg.random((2, 12)))
```

```python
a
```

    array([[4., 5., 7., 9., 2., 6., 6., 2., 0., 9., 2., 3.],
           [8., 5., 4., 7., 0., 7., 3., 0., 6., 9., 2., 6.]])

```python
np.hsplit(a, 4)
```

    [array([[4., 5., 7.],
            [8., 5., 4.]]),
     array([[9., 2., 6.],
            [7., 0., 7.]]),
     array([[6., 2., 0.],
            [3., 0., 6.]]),
     array([[9., 2., 3.],
            [9., 2., 6.]])]

```python
np.hsplit(a, (3, 4))
```

    [array([[4., 5., 7.],
            [8., 5., 4.]]),
     array([[9.],
            [7.]]),
     array([[2., 6., 6., 2., 0., 9., 2., 3.],
            [0., 7., 3., 0., 6., 9., 2., 6.]])]

```python
a = np.arange(12)**2
a
```

    array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121],
          dtype=int32)

```python
i = np.array([1, 1, 3, 8, 5]);i
```

    array([1, 1, 3, 8, 5])

```python
a[i]
```

    array([ 1,  1,  9, 64, 25], dtype=int32)

```python
a
```

    array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121],
          dtype=int32)

```python
j = np.array([[3, 4], [9, 7]]) ;j
```

    array([[3, 4],
           [9, 7]])

```python
a[j]
```

    array([[ 9, 16],
           [81, 49]], dtype=int32)

```python
palette = np.array([[0, 0, 0],         # black
                    [255, 0, 0],       # red
                    [0, 255, 0],       # green
                    [0, 0, 255],       # blue
                    [255, 255, 255]])  # white
palette
```

    array([[  0,   0,   0],
           [255,   0,   0],
           [  0, 255,   0],
           [  0,   0, 255],
           [255, 255, 255]])

```python
data = np.sin(np.arange(20)).reshape(5, 4);data
```

    array([[ 0.        ,  0.84147098,  0.90929743,  0.14112001],
           [-0.7568025 , -0.95892427, -0.2794155 ,  0.6569866 ],
           [ 0.98935825,  0.41211849, -0.54402111, -0.99999021],
           [-0.53657292,  0.42016704,  0.99060736,  0.65028784],
           [-0.28790332, -0.96139749, -0.75098725,  0.14987721]])

```python
data.argmax(axis=0)
```

    array([2, 0, 3, 1], dtype=int64)
