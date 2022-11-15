---
layout: single
title: "2202.11.07.파이썬 기초 데이터처리 기술(4)"
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

1. 실습\_알고리즘
2. 실습\_판다스
3. numpy\_기초
4. numpy\_통계사례
5. pandas_10min
6. pandas\_기술통계량

#### 1. 실습 알고리즘

### 문) 1차원의 점들이 주어졌을 때,

### 그 중 가장 거리가 짧은 것의 쌍을 출력하는 함수를 작성하시요.

### ( 단 점들의 배열은 모두 정렬되어 있다고 가정한다. )

#### s = [1,8,4,17,20,13,3] -> 결과는 (3,4 )

```python
 s = [1,8,4,17,20,13,3]
```

```python
 # 1. 정렬
s1 = [1,8,4,17,20,13,3]
```

```python
s1 = sorted(s);s1
```

    [1, 3, 4, 8, 13, 17, 20]

# 2. 이웃간 거리 계산하기

```python
for i in range(0, len(s1)-1):
    print(i, " ", s1[i], " ** ", s1[i+1])
```

    0   1  **  3
    1   3  **  4
    2   4  **  8
    3   8  **  13
    4   13  **  17
    5   17  **  20

```python
for i in range(0, len(s1)-1):
    print(i, " ", s1[i], " ** ", s1[i+1])
    diff = s1[i+1] - s1[i]
    print("diff = ", diff)
```

    0   1  **  3
    diff =  2
    1   3  **  4
    diff =  1
    2   4  **  8
    diff =  4
    3   8  **  13
    diff =  5
    4   13  **  17
    diff =  4
    5   17  **  20
    diff =  3

```python
pair = {}
for i in range(0, len(s1)-1):
    print(i, " ", s1[i], " ** ", s1[i+1])
    diff = s1[i+1] - s1[i]
    print("diff = ", diff)
    pair[diff] = [s1[i],s1[i+1]]
pair
```

    0   1  **  3
    diff =  2
    1   3  **  4
    diff =  1
    2   4  **  8
    diff =  4
    3   8  **  13
    diff =  5
    4   13  **  17
    diff =  4
    5   17  **  20
    diff =  3





    {2: [1, 3], 1: [3, 4], 4: [13, 17], 5: [8, 13], 3: [17, 20]}

```python
#3. 가장 짧은 거리 찾기
new_key = sorted(pair.keys())
new_key
```

    [1, 2, 3, 4, 5]

```python
pair[new_key[0]]
```

    [3, 4]

```python
s = [1,8,4,17,20,13,3]
s1 = sorted(s)
pair = {}
for i in range(0, len(s1)-1):
    diff = s1[i+1] - s1[i]
    pair[diff] = [s1[i],s1[i+1]]

new_key = sorted(pair.keys())
pair[new_key[0]]
```

    [3, 4]

```python
s1[0:]
```

    [1, 3, 4, 8, 13, 17, 20]

```python
s1[1:]
```

    [3, 4, 8, 13, 17, 20]

```python
pair1 = list(zip(s1[0:],s1[1:])); pair1
```

    [(1, 3), (3, 4), (4, 8), (8, 13), (13, 17), (17, 20)]

```python
s = [1,8,4,17,20,13,3]
s1 = sorted(s)
pair = list(zip(s1[0:],s1[1:]))
pair.sort(key=lambda x: x[1] - x[0])
pair[0]
```

    (3, 4)

## 4. 함수로 만들어 처리 하기

```python
def new_pair(s):
    s1 = sorted(s)
    pair = list(zip(s1[0:], s1[1:]))
    pair.sort(key=lambda x: x[1]-x[0])
    return pair[0]
```

```python
s = [8, 23, 31, 17, 33, 8, 34, 12, 20, 13, 3]
new_pair(s)
```

    (8, 8)

```python
s = [8, 23, 31, 17, 33, 10, 34, 12, 20, 13, 3]
new_pair(s)
```

    (12, 13)

#### 2. 실습\_판다스

```python
import pandas as pd
```

```python
data = {"나라명":["벨기에","인도","브라질"],
        "도시명":["브뤼셀","뉴델리","브라질리아"],
        "인구":["1천만명","13억명","2억명"]}
```

```python
df = pd.DataFrame(data)
df
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
      <th>나라명</th>
      <th>도시명</th>
      <th>인구</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>벨기에</td>
      <td>브뤼셀</td>
      <td>1천만명</td>
    </tr>
    <tr>
      <th>1</th>
      <td>인도</td>
      <td>뉴델리</td>
      <td>13억명</td>
    </tr>
    <tr>
      <th>2</th>
      <td>브라질</td>
      <td>브라질리아</td>
      <td>2억명</td>
    </tr>
  </tbody>
</table>
</div>

```python
score = {"국어":[76, 90, 64, 70, 98],
         "영어":[65, 75, 70, 82, 93],
         "수학":[80, 98, 62, 84, 85]}
score_df = pd.DataFrame(score)
score_df
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
      <th>국어</th>
      <th>영어</th>
      <th>수학</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76</td>
      <td>65</td>
      <td>80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>75</td>
      <td>98</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>70</td>
      <td>62</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>82</td>
      <td>84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98</td>
      <td>93</td>
      <td>85</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 총점
score_df["총점"] = score_df["국어"] + score_df["영어"] + score_df["수학"]
score_df
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
      <th>국어</th>
      <th>영어</th>
      <th>수학</th>
      <th>총점</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76</td>
      <td>65</td>
      <td>80</td>
      <td>221</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>75</td>
      <td>98</td>
      <td>263</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>70</td>
      <td>62</td>
      <td>196</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>82</td>
      <td>84</td>
      <td>236</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98</td>
      <td>93</td>
      <td>85</td>
      <td>276</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 평균
score_df["평균"] = (score_df["국어"] + score_df["영어"] + score_df["수학"])/3
score_df
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
      <th>국어</th>
      <th>영어</th>
      <th>수학</th>
      <th>총점</th>
      <th>평균</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76</td>
      <td>65</td>
      <td>80</td>
      <td>221</td>
      <td>73.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>75</td>
      <td>98</td>
      <td>263</td>
      <td>87.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>70</td>
      <td>62</td>
      <td>196</td>
      <td>65.333333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>82</td>
      <td>84</td>
      <td>236</td>
      <td>78.666667</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98</td>
      <td>93</td>
      <td>85</td>
      <td>276</td>
      <td>92.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 평균
score_df["평균"] = round((score_df["국어"] + score_df["영어"] + score_df["수학"])/3,2)
score_df
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
      <th>국어</th>
      <th>영어</th>
      <th>수학</th>
      <th>총점</th>
      <th>평균</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>76</td>
      <td>65</td>
      <td>80</td>
      <td>221</td>
      <td>73.67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>75</td>
      <td>98</td>
      <td>263</td>
      <td>87.67</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64</td>
      <td>70</td>
      <td>62</td>
      <td>196</td>
      <td>65.33</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>82</td>
      <td>84</td>
      <td>236</td>
      <td>78.67</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98</td>
      <td>93</td>
      <td>85</td>
      <td>276</td>
      <td>92.00</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

#### numpy\_기초

```python
# ndarry 생성
import pandas as pd
import numpy as np
```

```python
arr = np.array([1,2,3,4])
arr
```

    array([1, 2, 3, 4])

```python
np.zeros((3,3))
```

    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])

```python
np.ones((3,5))
```

    array([[1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1.]])

```python
np.empty((4,4))
```

    array([[4.67296746e-307, 1.69121096e-306, 8.90100164e-307,
            8.34441742e-308],
           [1.78022342e-306, 6.23058028e-307, 9.79107872e-307,
            6.89807188e-307],
           [7.56594375e-307, 6.23060065e-307, 1.78021527e-306,
            8.34454050e-308],
           [1.11261027e-306, 1.15706896e-306, 1.33512173e-306,
            1.33504432e-306]])

```python
np.arange(10)
```

    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

```python
np.arange(1,10,2)
```

    array([1, 3, 5, 7, 9])

```python
arr = np.array([[1,2,3], [4,5,6]]);arr
```

    array([[1, 2, 3],
           [4, 5, 6]])

```python
arr.shape
```

    (2, 3)

```python
arr.ndim
```

    2

```python
arr.dtype
```

    dtype('int32')

```python
arr_int = np.array(range(1,5))
arr_int
```

    array([1, 2, 3, 4])

```python
arr_int.dtype
```

    dtype('int32')

```python
arr_float = arr_int.astype(np.float64)
arr_float.dtype
```

    dtype('float64')

```python
arr_str = np.array(["1","2","3"])
arr_str.dtype
```

    dtype('<U1')

```python
arr_str
```

    array(['1', '2', '3'], dtype='<U1')

```python
arr_int = arr_str.astype(np.int64)
arr_int.dtype
```

    dtype('int64')

```python
arr_int
```

    array([1, 2, 3], dtype=int64)

```python
# 배열의 연산
arr1 = np.array([[1,2],[3,4]]); arr1
```

    array([[1, 2],
           [3, 4]])

```python
arr2 = np.array([[5,6], [7,8]]); arr2
```

    array([[5, 6],
           [7, 8]])

```python
arr1 + arr2
```

    array([[ 6,  8],
           [10, 12]])

```python
np.add(arr1, arr2)
```

    array([[ 6,  8],
           [10, 12]])

```python
arr1 * arr2
```

    array([[ 5, 12],
           [21, 32]])

```python
np.multiply(arr1, arr2)
```

    array([[ 5, 12],
           [21, 32]])

```python
arr1
```

    array([[1, 2],
           [3, 4]])

```python
arr2
```

    array([[5, 6],
           [7, 8]])

```python
1*5 + 2*7
```

    19

```python
1*6 + 2*8
```

    22

```python
# 행렬곱
arr1.dot(arr2)
```

    array([[19, 22],
           [43, 50]])

```python
np.dot(arr1, arr2)
```

    array([[19, 22],
           [43, 50]])

```python
arr = np.array([[1,2,3], [4,5,6], [7,8,9]])
arr
```

    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])

```python
arr[:2, 1:3]
```

    array([[2, 3],
           [5, 6]])

```python
arr[0,2]
```

    3

```python
arr[[0,1,2],[2,0,1]]
```

    array([3, 4, 8])

```python
arr > 4
```

    array([[False, False, False],
           [False,  True,  True],
           [ True,  True,  True]])

```python
arr[arr>4]
```

    array([5, 6, 7, 8, 9])

```python
idx = arr>4
arr[idx]
```

    array([5, 6, 7, 8, 9])

```python
print(arr[idx])
```

    [5 6 7 8 9]

```python

```

#### 4. numpy\_통계사례

```python
import pandas as pd
import numpy as np
```

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
url = "c:/test/winequality-red.csv"

redwine = np.loadtxt(url, delimiter=";", skiprows=1)
redwine
```

    array([[ 7.4  ,  0.7  ,  0.   , ...,  0.56 ,  9.4  ,  5.   ],
           [ 7.8  ,  0.88 ,  0.   , ...,  0.68 ,  9.8  ,  5.   ],
           [ 7.8  ,  0.76 ,  0.04 , ...,  0.65 ,  9.8  ,  5.   ],
           ...,
           [ 6.3  ,  0.51 ,  0.13 , ...,  0.75 , 11.   ,  6.   ],
           [ 5.9  ,  0.645,  0.12 , ...,  0.71 , 10.2  ,  5.   ],
           [ 6.   ,  0.31 ,  0.47 , ...,  0.66 , 11.   ,  6.   ]])

```python
print(redwine)
```

    [[ 7.4    0.7    0.    ...  0.56   9.4    5.   ]
     [ 7.8    0.88   0.    ...  0.68   9.8    5.   ]
     [ 7.8    0.76   0.04  ...  0.65   9.8    5.   ]
     ...
     [ 6.3    0.51   0.13  ...  0.75  11.     6.   ]
     [ 5.9    0.645  0.12  ...  0.71  10.2    5.   ]
     [ 6.     0.31   0.47  ...  0.66  11.     6.   ]]

```python
redwine.shape
```

    (1599, 12)

```python
redwine.ndim
```

    2

```python
redwine.sum()
```

    152084.78194

```python
redwine.mean()
```

    7.926036165311652

```python
redwine.mean(axis=0)
```

    array([ 8.31963727,  0.52782051,  0.27097561,  2.5388055 ,  0.08746654,
           15.87492183, 46.46779237,  0.99674668,  3.3111132 ,  0.65814884,
           10.42298311,  5.63602251])

```python
redwine.mean(axis=1)
```

    array([ 6.21198333, 10.25456667,  8.30825   , ...,  8.37347833,
            8.76795583,  7.7077075 ])

```python
redwine[:,0]
```

    array([7.4, 7.8, 7.8, ..., 6.3, 5.9, 6. ])

```python
import sys
np.set_printoptions(threshold=20)
```

```python
redwine[0,0:-1]
```

    array([ 7.4   ,  0.7   ,  0.    ,  1.9   ,  0.076 , 11.    , 34.    ,
            0.9978,  3.51  ,  0.56  ,  9.4   ])

#### 5. pandas_10min

```python
import numpy as np
import pandas as pd
```

```python
# 버젼 출력
print("pandas ver = ", pd.__version__)
print("numpy ver = {}".format(np.__version__))
```

    pandas ver =  1.4.4
    numpy ver = 1.21.5

1. Object Creation(객체 생성)

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8]);s
```

    0    1.0
    1    3.0
    2    5.0
    3    NaN
    4    6.0
    5    8.0
    dtype: float64

```python
dates = pd.date_range("20130101", periods=6);dates
```

    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')

```python
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD")); df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-0.572740</td>
      <td>-1.935663</td>
      <td>0.086809</td>
      <td>-1.419936</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.422815</td>
      <td>-1.674857</td>
      <td>-1.038498</td>
      <td>-1.115653</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1.315095</td>
      <td>-0.268049</td>
      <td>-0.748059</td>
      <td>1.778579</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1.586233</td>
      <td>0.869545</td>
      <td>-0.642550</td>
      <td>0.131243</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.989636</td>
      <td>-1.391699</td>
      <td>-1.416713</td>
      <td>-0.478856</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.058432</td>
      <td>-0.820128</td>
      <td>-1.424753</td>
      <td>-0.233420</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=["A","B","C","D"]); df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
); df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>test</td>
      <td>foo</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>2013-01-02</td>
      <td>1.0</td>
      <td>3</td>
      <td>train</td>
      <td>foo</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2.dtypes
```

    A           float64
    B    datetime64[ns]
    C           float32
    D             int32
    E          category
    F            object
    dtype: object

```python
dir(df2)
```

    ['A',
     'B',
     'C',
     'D',
     'E',
     'F',
     'T',
     '_AXIS_LEN',
     '_AXIS_ORDERS',
     '_AXIS_TO_AXIS_NUMBER',
     '_HANDLED_TYPES',
     '__abs__',
     '__add__',
     '__and__',
     '__annotations__',
     '__array__',
     '__array_priority__',
     '__array_ufunc__',
     '__array_wrap__',
     '__bool__',
     '__class__',
     '__contains__',
     '__copy__',
     '__deepcopy__',
     '__delattr__',
     '__delitem__',
     '__dict__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__finalize__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattr__',
     '__getattribute__',
     '__getitem__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__iadd__',
     '__iand__',
     '__ifloordiv__',
     '__imod__',
     '__imul__',
     '__init__',
     '__init_subclass__',
     '__invert__',
     '__ior__',
     '__ipow__',
     '__isub__',
     '__iter__',
     '__itruediv__',
     '__ixor__',
     '__le__',
     '__len__',
     '__lt__',
     '__matmul__',
     '__mod__',
     '__module__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
     '__nonzero__',
     '__or__',
     '__pos__',
     '__pow__',
     '__radd__',
     '__rand__',
     '__rdivmod__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rfloordiv__',
     '__rmatmul__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__round__',
     '__rpow__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__setitem__',
     '__setstate__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__weakref__',
     '__xor__',
     '_accessors',
     '_accum_func',
     '_add_numeric_operations',
     '_agg_by_level',
     '_agg_examples_doc',
     '_agg_summary_and_see_also_doc',
     '_align_frame',
     '_align_series',
     '_append',
     '_arith_method',
     '_as_manager',
     '_attrs',
     '_box_col_values',
     '_can_fast_transpose',
     '_check_inplace_and_allows_duplicate_labels',
     '_check_inplace_setting',
     '_check_is_chained_assignment_possible',
     '_check_label_or_level_ambiguity',
     '_check_setitem_copy',
     '_clear_item_cache',
     '_clip_with_one_bound',
     '_clip_with_scalar',
     '_cmp_method',
     '_combine_frame',
     '_consolidate',
     '_consolidate_inplace',
     '_construct_axes_dict',
     '_construct_axes_from_arguments',
     '_construct_result',
     '_constructor',
     '_constructor_sliced',
     '_convert',
     '_count_level',
     '_data',
     '_dir_additions',
     '_dir_deletions',
     '_dispatch_frame_op',
     '_drop_axis',
     '_drop_labels_or_levels',
     '_ensure_valid_index',
     '_find_valid_index',
     '_flags',
     '_from_arrays',
     '_from_mgr',
     '_get_agg_axis',
     '_get_axis',
     '_get_axis_name',
     '_get_axis_number',
     '_get_axis_resolvers',
     '_get_block_manager_axis',
     '_get_bool_data',
     '_get_cleaned_column_resolvers',
     '_get_column_array',
     '_get_index_resolvers',
     '_get_item_cache',
     '_get_label_or_level_values',
     '_get_numeric_data',
     '_get_value',
     '_getitem_bool_array',
     '_getitem_multilevel',
     '_gotitem',
     '_hidden_attrs',
     '_indexed_same',
     '_info_axis',
     '_info_axis_name',
     '_info_axis_number',
     '_info_repr',
     '_init_mgr',
     '_inplace_method',
     '_internal_names',
     '_internal_names_set',
     '_is_copy',
     '_is_homogeneous_type',
     '_is_label_or_level_reference',
     '_is_label_reference',
     '_is_level_reference',
     '_is_mixed_type',
     '_is_view',
     '_iset_item',
     '_iset_item_mgr',
     '_iset_not_inplace',
     '_item_cache',
     '_iter_column_arrays',
     '_ixs',
     '_join_compat',
     '_logical_func',
     '_logical_method',
     '_maybe_cache_changed',
     '_maybe_update_cacher',
     '_metadata',
     '_mgr',
     '_min_count_stat_function',
     '_needs_reindex_multi',
     '_protect_consolidate',
     '_reduce',
     '_reduce_axis1',
     '_reindex_axes',
     '_reindex_columns',
     '_reindex_index',
     '_reindex_multi',
     '_reindex_with_indexers',
     '_rename',
     '_replace_columnwise',
     '_repr_data_resource_',
     '_repr_fits_horizontal_',
     '_repr_fits_vertical_',
     '_repr_html_',
     '_repr_latex_',
     '_reset_cache',
     '_reset_cacher',
     '_sanitize_column',
     '_series',
     '_set_axis',
     '_set_axis_name',
     '_set_axis_nocheck',
     '_set_is_copy',
     '_set_item',
     '_set_item_frame_value',
     '_set_item_mgr',
     '_set_value',
     '_setitem_array',
     '_setitem_frame',
     '_setitem_slice',
     '_slice',
     '_stat_axis',
     '_stat_axis_name',
     '_stat_axis_number',
     '_stat_function',
     '_stat_function_ddof',
     '_take_with_is_copy',
     '_to_dict_of_blocks',
     '_typ',
     '_update_inplace',
     '_validate_dtype',
     '_values',
     '_where',
     'abs',
     'add',
     'add_prefix',
     'add_suffix',
     'agg',
     'aggregate',
     'align',
     'all',
     'any',
     'append',
     'apply',
     'applymap',
     'asfreq',
     'asof',
     'assign',
     'astype',
     'at',
     'at_time',
     'attrs',
     'axes',
     'backfill',
     'between_time',
     'bfill',
     'bool',
     'boxplot',
     'clip',
     'columns',
     'combine',
     'combine_first',
     'compare',
     'convert_dtypes',
     'copy',
     'corr',
     'corrwith',
     'count',
     'cov',
     'cummax',
     'cummin',
     'cumprod',
     'cumsum',
     'describe',
     'diff',
     'div',
     'divide',
     'dot',
     'drop',
     'drop_duplicates',
     'droplevel',
     'dropna',
     'dtypes',
     'duplicated',
     'empty',
     'eq',
     'equals',
     'eval',
     'ewm',
     'expanding',
     'explode',
     'ffill',
     'fillna',
     'filter',
     'first',
     'first_valid_index',
     'flags',
     'floordiv',
     'from_dict',
     'from_records',
     'ge',
     'get',
     'groupby',
     'gt',
     'head',
     'hist',
     'iat',
     'idxmax',
     'idxmin',
     'iloc',
     'index',
     'infer_objects',
     'info',
     'insert',
     'interpolate',
     'isin',
     'isna',
     'isnull',
     'items',
     'iteritems',
     'iterrows',
     'itertuples',
     'join',
     'keys',
     'kurt',
     'kurtosis',
     'last',
     'last_valid_index',
     'le',
     'loc',
     'lookup',
     'lt',
     'mad',
     'mask',
     'max',
     'mean',
     'median',
     'melt',
     'memory_usage',
     'merge',
     'min',
     'mod',
     'mode',
     'mul',
     'multiply',
     'ndim',
     'ne',
     'nlargest',
     'notna',
     'notnull',
     'nsmallest',
     'nunique',
     'pad',
     'pct_change',
     'pipe',
     'pivot',
     'pivot_table',
     'plot',
     'pop',
     'pow',
     'prod',
     'product',
     'quantile',
     'query',
     'radd',
     'rank',
     'rdiv',
     'reindex',
     'reindex_like',
     'rename',
     'rename_axis',
     'reorder_levels',
     'replace',
     'resample',
     'reset_index',
     'rfloordiv',
     'rmod',
     'rmul',
     'rolling',
     'round',
     'rpow',
     'rsub',
     'rtruediv',
     'sample',
     'select_dtypes',
     'sem',
     'set_axis',
     'set_flags',
     'set_index',
     'shape',
     'shift',
     'size',
     'skew',
     'slice_shift',
     'sort_index',
     'sort_values',
     'squeeze',
     'stack',
     'std',
     'style',
     'sub',
     'subtract',
     'sum',
     'swapaxes',
     'swaplevel',
     'tail',
     'take',
     'to_clipboard',
     'to_csv',
     'to_dict',
     'to_excel',
     'to_feather',
     'to_gbq',
     'to_hdf',
     'to_html',
     'to_json',
     'to_latex',
     'to_markdown',
     'to_numpy',
     'to_parquet',
     'to_period',
     'to_pickle',
     'to_records',
     'to_sql',
     'to_stata',
     'to_string',
     'to_timestamp',
     'to_xarray',
     'to_xml',
     'transform',
     'transpose',
     'truediv',
     'truncate',
     'tz_convert',
     'tz_localize',
     'unstack',
     'update',
     'value_counts',
     'values',
     'var',
     'where',
     'xs']

2. Viewing Data(데이터 확인)

```python
# 자주사용
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 자주사용
df.tail(3)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.index
```

    DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
                   '2013-01-05', '2013-01-06'],
                  dtype='datetime64[ns]', freq='D')

```python
# 자주사용
df.columns
```

    Index(['A', 'B', 'C', 'D'], dtype='object')

```python
df.columns = ['A', 'B', 'C', 'D']; df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.to_numpy()
```

    array([[-1.68586299, -0.08285189,  0.11007761, -0.08906774],
           [ 0.89522807,  0.02259359, -0.5056847 ,  1.47684125],
           [ 0.2900124 , -0.49747906, -0.38395653,  0.51158157],
           [ 0.01194919,  0.38828465,  1.1333263 , -0.80127585],
           [ 0.44764817,  0.11225644, -0.98910918, -0.33009466],
           [-0.39960961,  1.04324988, -2.28945837, -0.66377446]])

```python
df.values
```

    array([[-1.68586299, -0.08285189,  0.11007761, -0.08906774],
           [ 0.89522807,  0.02259359, -0.5056847 ,  1.47684125],
           [ 0.2900124 , -0.49747906, -0.38395653,  0.51158157],
           [ 0.01194919,  0.38828465,  1.1333263 , -0.80127585],
           [ 0.44764817,  0.11225644, -0.98910918, -0.33009466],
           [-0.39960961,  1.04324988, -2.28945837, -0.66377446]])

```python
# 자주사용
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-0.073439</td>
      <td>0.164342</td>
      <td>-0.487467</td>
      <td>0.017368</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.900615</td>
      <td>0.518565</td>
      <td>1.139640</td>
      <td>0.853288</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.685863</td>
      <td>-0.497479</td>
      <td>-2.289458</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-0.296720</td>
      <td>-0.056491</td>
      <td>-0.868253</td>
      <td>-0.580355</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.150981</td>
      <td>0.067425</td>
      <td>-0.444821</td>
      <td>-0.209581</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.408239</td>
      <td>0.319278</td>
      <td>-0.013431</td>
      <td>0.361419</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.895228</td>
      <td>1.043250</td>
      <td>1.133326</td>
      <td>1.476841</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.T
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
      <th>2013-01-01</th>
      <th>2013-01-02</th>
      <th>2013-01-03</th>
      <th>2013-01-04</th>
      <th>2013-01-05</th>
      <th>2013-01-06</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>-1.685863</td>
      <td>0.895228</td>
      <td>0.290012</td>
      <td>0.011949</td>
      <td>0.447648</td>
      <td>-0.399610</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.082852</td>
      <td>0.022594</td>
      <td>-0.497479</td>
      <td>0.388285</td>
      <td>0.112256</td>
      <td>1.043250</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.110078</td>
      <td>-0.505685</td>
      <td>-0.383957</td>
      <td>1.133326</td>
      <td>-0.989109</td>
      <td>-2.289458</td>
    </tr>
    <tr>
      <th>D</th>
      <td>-0.089068</td>
      <td>1.476841</td>
      <td>0.511582</td>
      <td>-0.801276</td>
      <td>-0.330095</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.sort_index(axis=1, ascending=True) # axis = 0 or 1,  True / False
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.sort_values(by="B")
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

## ( 실습 )

1. 데이터셋은 winequality 데이터 셋
2. winequality-red.csv로 red_df 만들고
3. winequality-white.csv로 white_df 만들고
4. red_df에서 pH가 가장큰 숫자 5개를 추출

```python
red_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

red_df = pd.read_csv(red_url,delimiter=";")
white_df = pd.read_csv(white_url, delimiter=";")
```

```python
red_df.shape
```

    (1599, 12)

```python
red_df.head(3)
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
red_df.sort_values(by="pH", ascending=False).head(5)["pH"]
```

    1316    4.01
    1321    4.01
    695     3.90
    45      3.90
    95      3.85
    Name: pH, dtype: float64

```python
red_df["pH"].sort_values(ascending=False).head(5)
```

    1316    4.01
    1321    4.01
    695     3.90
    45      3.90
    95      3.85
    Name: pH, dtype: float64

```python
red_df.sort_values(by=["pH","alcohol"], ascending=[False, True]).\
                                         head(5)[["pH","alcohol"]]
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
      <th>pH</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1316</th>
      <td>4.01</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>1321</th>
      <td>4.01</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>695</th>
      <td>3.90</td>
      <td>12.8</td>
    </tr>
    <tr>
      <th>45</th>
      <td>3.90</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>95</th>
      <td>3.85</td>
      <td>12.9</td>
    </tr>
  </tbody>
</table>
</div>

3. Selection (데이터 추출)

```python
df["A"]
```

    2013-01-01   -1.685863
    2013-01-02    0.895228
    2013-01-03    0.290012
    2013-01-04    0.011949
    2013-01-05    0.447648
    2013-01-06   -0.399610
    Freq: D, Name: A, dtype: float64

```python
df.A
```

    2013-01-01   -1.685863
    2013-01-02    0.895228
    2013-01-03    0.290012
    2013-01-04    0.011949
    2013-01-05    0.447648
    2013-01-06   -0.399610
    Freq: D, Name: A, dtype: float64

```python
df[["A","B"]]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[0:3]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
  </tbody>
</table>
</div>

```python
df["20130102":"20130104"]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.loc[dates[0]]
```

    A   -1.685863
    B   -0.082852
    C    0.110078
    D   -0.089068
    Name: 2013-01-01 00:00:00, dtype: float64

```python
df.loc[:, ["A", "B"]]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.loc["20130102":"20130104", ["A", "B"]]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.loc["20130102", ["A", "B"]]
```

    A    0.895228
    B    0.022594
    Name: 2013-01-02 00:00:00, dtype: float64

```python
df.loc[dates[0], "A"]
```

    -1.6858629905070324

```python
df_tmp = df.copy()
```

```python
df_tmp.loc[dates[0], "A"] = 9999
```

```python
df_tmp
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>9999.000000</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.at[dates[0], "A"]
```

    -1.6858629905070324

### Selection by position

```python
df.iloc[3]
```

    A    0.011949
    B    0.388285
    C    1.133326
    D   -0.801276
    Name: 2013-01-04 00:00:00, dtype: float64

```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[3:5, 0:2]
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
      <th>A</th>
      <th>B</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[[1, 2, 4], [0, 2]]
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
      <th>A</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>-0.505685</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.383957</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>-0.989109</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[1:3, :]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[:, 1:3]
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
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-0.082852</td>
      <td>0.110078</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.022594</td>
      <td>-0.505685</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>-0.497479</td>
      <td>-0.383957</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.388285</td>
      <td>1.133326</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.112256</td>
      <td>-0.989109</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>1.043250</td>
      <td>-2.289458</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.iloc[1, 1]
```

    0.022593592290612436

```python
df.iat[1, 1]
```

    0.022593592290612436

#### Boolean indexing

```python
df[df["A"] > 0]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[df.A > 0]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[df > 0]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.110078</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>NaN</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>NaN</td>
      <td>1.043250</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
df[(df.A>0) & (df.A < 1)]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2 = df.copy()
df2["E"] = ["one", "one", "two", "three", "four", "three"]
df2
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>three</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
      <td>four</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
      <td>three</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2[df2["E"].isin(["two", "four"])]
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
      <td>four</td>
    </tr>
  </tbody>
</table>
</div>

```python
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range("20130102", periods=6))
s1
```

    2013-01-02    1
    2013-01-03    2
    2013-01-04    3
    2013-01-05    4
    2013-01-06    5
    2013-01-07    6
    Freq: D, dtype: int64

```python
df["F"] = s1
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.isnull().sum().sum()
```

    1

```python
df.isnull().sum()
```

    A    0
    B    0
    C    0
    D    0
    F    1
    dtype: int64

### ( 실습 )

1. red_df에서 pH가 3.53 보다 크고, pH가 3.55보다 작은 데이터셋 aa를 만드세요

```python
aa = red_df[(red_df["pH"] > 3.53) & (red_df["pH"] < 3.55)]
```

```python
aa.shape
```

    (16, 12)

```python
aa.head()
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
      <th>75</th>
      <td>8.8</td>
      <td>0.41</td>
      <td>0.64</td>
      <td>2.2</td>
      <td>0.093</td>
      <td>9.0</td>
      <td>42.0</td>
      <td>0.9986</td>
      <td>3.54</td>
      <td>0.66</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>76</th>
      <td>8.8</td>
      <td>0.41</td>
      <td>0.64</td>
      <td>2.2</td>
      <td>0.093</td>
      <td>9.0</td>
      <td>42.0</td>
      <td>0.9986</td>
      <td>3.54</td>
      <td>0.66</td>
      <td>10.5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>127</th>
      <td>8.1</td>
      <td>1.33</td>
      <td>0.00</td>
      <td>1.8</td>
      <td>0.082</td>
      <td>3.0</td>
      <td>12.0</td>
      <td>0.9964</td>
      <td>3.54</td>
      <td>0.48</td>
      <td>10.9</td>
      <td>5</td>
    </tr>
    <tr>
      <th>146</th>
      <td>5.8</td>
      <td>0.68</td>
      <td>0.02</td>
      <td>1.8</td>
      <td>0.087</td>
      <td>21.0</td>
      <td>94.0</td>
      <td>0.9944</td>
      <td>3.54</td>
      <td>0.52</td>
      <td>10.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>183</th>
      <td>6.8</td>
      <td>0.61</td>
      <td>0.20</td>
      <td>1.8</td>
      <td>0.077</td>
      <td>11.0</td>
      <td>65.0</td>
      <td>0.9971</td>
      <td>3.54</td>
      <td>0.58</td>
      <td>9.3</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

### Missing data

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ["E"])
df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.loc[dates[0] : dates[1], "E"] = 1;df1
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.dropna(how="any")  # 한개라도 na가 있으면 삭제
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.dropna(how="all") #전체가 na면 삭제
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.fillna(value=5)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>5.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

### Operations

```python
df.mean()
```

    A   -0.073439
    B    0.164342
    C   -0.487467
    D    0.017368
    F    3.000000
    dtype: float64

```python
df.mean(1)
```

    2013-01-01   -0.436926
    2013-01-02    0.577796
    2013-01-03    0.384032
    2013-01-04    0.746457
    2013-01-05    0.648140
    2013-01-06    0.538081
    Freq: D, dtype: float64

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates)
s
```

    2013-01-01    1.0
    2013-01-02    3.0
    2013-01-03    5.0
    2013-01-04    NaN
    2013-01-05    6.0
    2013-01-06    8.0
    Freq: D, dtype: float64

```python
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
s
```

    2013-01-01    NaN
    2013-01-02    NaN
    2013-01-03    1.0
    2013-01-04    3.0
    2013-01-05    5.0
    2013-01-06    NaN
    Freq: D, dtype: float64

```python
df.sub(s, axis="index")
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>-0.709988</td>
      <td>-1.497479</td>
      <td>-1.383957</td>
      <td>-0.488418</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-2.988051</td>
      <td>-2.611715</td>
      <td>-1.866674</td>
      <td>-3.801276</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>-4.552352</td>
      <td>-4.887744</td>
      <td>-5.989109</td>
      <td>-5.330095</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

```python
df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>0.895228</td>
      <td>0.022594</td>
      <td>-0.505685</td>
      <td>1.476841</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>0.290012</td>
      <td>-0.497479</td>
      <td>-0.383957</td>
      <td>0.511582</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>0.011949</td>
      <td>0.388285</td>
      <td>1.133326</td>
      <td>-0.801276</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>0.447648</td>
      <td>0.112256</td>
      <td>-0.989109</td>
      <td>-0.330095</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.399610</td>
      <td>1.043250</td>
      <td>-2.289458</td>
      <td>-0.663774</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

### Apply

```python
df.apply(np.cumsum)
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>F</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>-1.685863</td>
      <td>-0.082852</td>
      <td>0.110078</td>
      <td>-0.089068</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>-0.790635</td>
      <td>-0.060258</td>
      <td>-0.395607</td>
      <td>1.387774</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>-0.500623</td>
      <td>-0.557737</td>
      <td>-0.779564</td>
      <td>1.899355</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>-0.488673</td>
      <td>-0.169453</td>
      <td>0.353763</td>
      <td>1.098079</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>-0.041025</td>
      <td>-0.057196</td>
      <td>-0.635347</td>
      <td>0.767985</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>-0.440635</td>
      <td>0.986054</td>
      <td>-2.924805</td>
      <td>0.104210</td>
      <td>15.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.apply(lambda x: x.max() - x.min())
```

    A    2.581091
    B    1.540729
    C    3.422785
    D    2.278117
    F    4.000000
    dtype: float64

### String Methods

```python
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"]);s
```

    0       A
    1       B
    2       C
    3    Aaba
    4    Baca
    5     NaN
    6    CABA
    7     dog
    8     cat
    dtype: object

```python
s.str.lower()
```

    0       a
    1       b
    2       c
    3    aaba
    4    baca
    5     NaN
    6    caba
    7     dog
    8     cat
    dtype: object

```python
df = pd.DataFrame(np.random.randn(10, 4));df
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.534426</td>
      <td>0.443640</td>
      <td>0.763320</td>
      <td>0.709349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.379915</td>
      <td>-1.266352</td>
      <td>-0.865880</td>
      <td>0.246908</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.429715</td>
      <td>-0.626917</td>
      <td>0.377427</td>
      <td>2.459311</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.245463</td>
      <td>0.367838</td>
      <td>1.046948</td>
      <td>-0.157438</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.359391</td>
      <td>0.283233</td>
      <td>0.961465</td>
      <td>-0.810725</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.384726</td>
      <td>-0.391406</td>
      <td>-2.766937</td>
      <td>0.257281</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.594053</td>
      <td>-1.384256</td>
      <td>0.581234</td>
      <td>0.955387</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.714662</td>
      <td>-1.028273</td>
      <td>0.268160</td>
      <td>-1.640451</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.516751</td>
      <td>-0.572942</td>
      <td>-0.332845</td>
      <td>0.335118</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-1.380051</td>
      <td>0.679796</td>
      <td>0.673701</td>
      <td>-1.511586</td>
    </tr>
  </tbody>
</table>
</div>

```python
a1 = df[:3]
a2 = df[3:7]
a3 = df[7:]
```

```python
pieces = [a1, a2, a3]
```

```python
pd.concat(pieces).head(3)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.534426</td>
      <td>0.443640</td>
      <td>0.763320</td>
      <td>0.709349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.379915</td>
      <td>-1.266352</td>
      <td>-0.865880</td>
      <td>0.246908</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.429715</td>
      <td>-0.626917</td>
      <td>0.377427</td>
      <td>2.459311</td>
    </tr>
  </tbody>
</table>
</div>

```python
pd.concat([a1, a2, a3]).head(3)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.534426</td>
      <td>0.443640</td>
      <td>0.763320</td>
      <td>0.709349</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.379915</td>
      <td>-1.266352</td>
      <td>-0.865880</td>
      <td>0.246908</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.429715</td>
      <td>-0.626917</td>
      <td>0.377427</td>
      <td>2.459311</td>
    </tr>
  </tbody>
</table>
</div>

```python
tuples = list(
    zip(
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    )
)
tuples
```

    [('bar', 'one'),
     ('bar', 'two'),
     ('baz', 'one'),
     ('baz', 'two'),
     ('foo', 'one'),
     ('foo', 'two'),
     ('qux', 'one'),
     ('qux', 'two')]

```python
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index
```

    MultiIndex([('bar', 'one'),
                ('bar', 'two'),
                ('baz', 'one'),
                ('baz', 'two'),
                ('foo', 'one'),
                ('foo', 'two'),
                ('qux', 'one'),
                ('qux', 'two')],
               names=['first', 'second'])

```python
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=["A", "B"])
df
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>-1.494596</td>
      <td>-0.634684</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.096347</td>
      <td>-0.567407</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>-2.060786</td>
      <td>-1.481477</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.592279</td>
      <td>0.224237</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">foo</th>
      <th>one</th>
      <td>-0.359884</td>
      <td>0.306792</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.026033</td>
      <td>-1.650093</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">qux</th>
      <th>one</th>
      <td>-0.583517</td>
      <td>0.256274</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.715174</td>
      <td>2.299707</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2 = df[:4]; df2
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>-1.494596</td>
      <td>-0.634684</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.096347</td>
      <td>-0.567407</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>-2.060786</td>
      <td>-1.481477</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.592279</td>
      <td>0.224237</td>
    </tr>
  </tbody>
</table>
</div>

```python
stacked = df2.stack(); stacked
```

    first  second
    bar    one     A   -1.494596
                   B   -0.634684
           two     A   -0.096347
                   B   -0.567407
    baz    one     A   -2.060786
                   B   -1.481477
           two     A    0.592279
                   B    0.224237
    dtype: float64

```python
stacked.unstack()
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
      <th></th>
      <th>A</th>
      <th>B</th>
    </tr>
    <tr>
      <th>first</th>
      <th>second</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">bar</th>
      <th>one</th>
      <td>-1.494596</td>
      <td>-0.634684</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.096347</td>
      <td>-0.567407</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">baz</th>
      <th>one</th>
      <td>-2.060786</td>
      <td>-1.481477</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.592279</td>
      <td>0.224237</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = pd.DataFrame(
    {
        "A": ["one", "one", "two", "three"] * 3,
        "B": ["A", "B", "C"] * 4,
        "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
        "D": np.random.randn(12),
        "E": np.random.randn(12),
    }
); df
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
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one</td>
      <td>A</td>
      <td>foo</td>
      <td>0.567068</td>
      <td>0.359968</td>
    </tr>
    <tr>
      <th>1</th>
      <td>one</td>
      <td>B</td>
      <td>foo</td>
      <td>-0.340880</td>
      <td>-0.846081</td>
    </tr>
    <tr>
      <th>2</th>
      <td>two</td>
      <td>C</td>
      <td>foo</td>
      <td>-0.301356</td>
      <td>0.256247</td>
    </tr>
    <tr>
      <th>3</th>
      <td>three</td>
      <td>A</td>
      <td>bar</td>
      <td>0.709220</td>
      <td>0.665442</td>
    </tr>
    <tr>
      <th>4</th>
      <td>one</td>
      <td>B</td>
      <td>bar</td>
      <td>-0.226716</td>
      <td>0.128314</td>
    </tr>
    <tr>
      <th>5</th>
      <td>one</td>
      <td>C</td>
      <td>bar</td>
      <td>0.022434</td>
      <td>-0.517643</td>
    </tr>
    <tr>
      <th>6</th>
      <td>two</td>
      <td>A</td>
      <td>foo</td>
      <td>2.105826</td>
      <td>0.317199</td>
    </tr>
    <tr>
      <th>7</th>
      <td>three</td>
      <td>B</td>
      <td>foo</td>
      <td>-1.506627</td>
      <td>1.128718</td>
    </tr>
    <tr>
      <th>8</th>
      <td>one</td>
      <td>C</td>
      <td>foo</td>
      <td>0.779921</td>
      <td>2.076121</td>
    </tr>
    <tr>
      <th>9</th>
      <td>one</td>
      <td>A</td>
      <td>bar</td>
      <td>-0.662814</td>
      <td>-0.655627</td>
    </tr>
    <tr>
      <th>10</th>
      <td>two</td>
      <td>B</td>
      <td>bar</td>
      <td>0.326756</td>
      <td>-0.451785</td>
    </tr>
    <tr>
      <th>11</th>
      <td>three</td>
      <td>C</td>
      <td>bar</td>
      <td>-1.213507</td>
      <td>-1.232443</td>
    </tr>
  </tbody>
</table>
</div>

```python
pd.pivot_table(df, values="D", index=["A", "B"], columns=["C"])
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
      <th>C</th>
      <th>bar</th>
      <th>foo</th>
    </tr>
    <tr>
      <th>A</th>
      <th>B</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">one</th>
      <th>A</th>
      <td>-0.662814</td>
      <td>0.567068</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.226716</td>
      <td>-0.340880</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.022434</td>
      <td>0.779921</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">three</th>
      <th>A</th>
      <td>0.709220</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>B</th>
      <td>NaN</td>
      <td>-1.506627</td>
    </tr>
    <tr>
      <th>C</th>
      <td>-1.213507</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">two</th>
      <th>A</th>
      <td>NaN</td>
      <td>2.105826</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.326756</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>C</th>
      <td>NaN</td>
      <td>-0.301356</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

#### 6. pandas\_기술통계량

```python
import pandas as pd
```

```python
# 웹에서 데이터를 읽어오는 방법
url = "http://freakonometrics.free.fr/german_credit.csv"
german = pd.read_csv(url)
german.head(3)
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
      <th>Creditability</th>
      <th>Account Balance</th>
      <th>Duration of Credit (month)</th>
      <th>Payment Status of Previous Credit</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
      <th>Value Savings/Stocks</th>
      <th>Length of current employment</th>
      <th>Instalment per cent</th>
      <th>Sex &amp; Marital Status</th>
      <th>...</th>
      <th>Duration in Current address</th>
      <th>Most valuable available asset</th>
      <th>Age (years)</th>
      <th>Concurrent Credits</th>
      <th>Type of apartment</th>
      <th>No of Credits at this Bank</th>
      <th>Occupation</th>
      <th>No of dependents</th>
      <th>Telephone</th>
      <th>Foreign Worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>2</td>
      <td>1049</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2799</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>36</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>841</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>23</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>

```python
# PC에서 화일 읽기
url = "c:/test/german_credit.csv"
german = pd.read_csv(url)
german.head(3)
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
      <th>Creditability</th>
      <th>Account Balance</th>
      <th>Duration of Credit (month)</th>
      <th>Payment Status of Previous Credit</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
      <th>Value Savings/Stocks</th>
      <th>Length of current employment</th>
      <th>Instalment per cent</th>
      <th>Sex &amp; Marital Status</th>
      <th>...</th>
      <th>Duration in Current address</th>
      <th>Most valuable available asset</th>
      <th>Age (years)</th>
      <th>Concurrent Credits</th>
      <th>Type of apartment</th>
      <th>No of Credits at this Bank</th>
      <th>Occupation</th>
      <th>No of dependents</th>
      <th>Telephone</th>
      <th>Foreign Worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>2</td>
      <td>1049</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2799</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>36</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>841</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>23</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>

```python
# 1. head()
german.head(3)
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
      <th>Creditability</th>
      <th>Account Balance</th>
      <th>Duration of Credit (month)</th>
      <th>Payment Status of Previous Credit</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
      <th>Value Savings/Stocks</th>
      <th>Length of current employment</th>
      <th>Instalment per cent</th>
      <th>Sex &amp; Marital Status</th>
      <th>...</th>
      <th>Duration in Current address</th>
      <th>Most valuable available asset</th>
      <th>Age (years)</th>
      <th>Concurrent Credits</th>
      <th>Type of apartment</th>
      <th>No of Credits at this Bank</th>
      <th>Occupation</th>
      <th>No of dependents</th>
      <th>Telephone</th>
      <th>Foreign Worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>2</td>
      <td>1049</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2799</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>36</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>841</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>23</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>

```python
# 2. tail()
german.tail(3)
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
      <th>Creditability</th>
      <th>Account Balance</th>
      <th>Duration of Credit (month)</th>
      <th>Payment Status of Previous Credit</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
      <th>Value Savings/Stocks</th>
      <th>Length of current employment</th>
      <th>Instalment per cent</th>
      <th>Sex &amp; Marital Status</th>
      <th>...</th>
      <th>Duration in Current address</th>
      <th>Most valuable available asset</th>
      <th>Age (years)</th>
      <th>Concurrent Credits</th>
      <th>Type of apartment</th>
      <th>No of Credits at this Bank</th>
      <th>Occupation</th>
      <th>No of dependents</th>
      <th>Telephone</th>
      <th>Foreign Worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>997</th>
      <td>0</td>
      <td>4</td>
      <td>21</td>
      <td>4</td>
      <td>0</td>
      <td>12680</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>4</td>
      <td>4</td>
      <td>30</td>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>998</th>
      <td>0</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>3</td>
      <td>6468</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>1</td>
      <td>4</td>
      <td>52</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>999</th>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>2</td>
      <td>2</td>
      <td>6350</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>31</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>

```python
# 3. columns 이름
german.columns
```

    Index(['Creditability', 'Account Balance', 'Duration of Credit (month)',
           'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
           'Value Savings/Stocks', 'Length of current employment',
           'Instalment per cent', 'Sex & Marital Status', 'Guarantors',
           'Duration in Current address', 'Most valuable available asset',
           'Age (years)', 'Concurrent Credits', 'Type of apartment',
           'No of Credits at this Bank', 'Occupation', 'No of dependents',
           'Telephone', 'Foreign Worker'],
          dtype='object')

```python
# 4. df 건수 확인
german.shape
```

    (1000, 21)

```python
german.shape[0]
```

    1000

```python
len(german)
```

    1000

```python
len(german.columns)
```

    21

```python
# 5. df 전체 정보 확인
german.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 21 columns):
     #   Column                             Non-Null Count  Dtype
    ---  ------                             --------------  -----
     0   Creditability                      1000 non-null   int64
     1   Account Balance                    1000 non-null   int64
     2   Duration of Credit (month)         1000 non-null   int64
     3   Payment Status of Previous Credit  1000 non-null   int64
     4   Purpose                            1000 non-null   int64
     5   Credit Amount                      1000 non-null   int64
     6   Value Savings/Stocks               1000 non-null   int64
     7   Length of current employment       1000 non-null   int64
     8   Instalment per cent                1000 non-null   int64
     9   Sex & Marital Status               1000 non-null   int64
     10  Guarantors                         1000 non-null   int64
     11  Duration in Current address        1000 non-null   int64
     12  Most valuable available asset      1000 non-null   int64
     13  Age (years)                        1000 non-null   int64
     14  Concurrent Credits                 1000 non-null   int64
     15  Type of apartment                  1000 non-null   int64
     16  No of Credits at this Bank         1000 non-null   int64
     17  Occupation                         1000 non-null   int64
     18  No of dependents                   1000 non-null   int64
     19  Telephone                          1000 non-null   int64
     20  Foreign Worker                     1000 non-null   int64
    dtypes: int64(21)
    memory usage: 164.2 KB

```python
# 6. 전체 결측치 확인
german.isnull().sum().sum()
```

    0

```python
# 컬럼별 결측치 확인
german.isnull().sum()
```

    Creditability                        0
    Account Balance                      0
    Duration of Credit (month)           0
    Payment Status of Previous Credit    0
    Purpose                              0
    Credit Amount                        0
    Value Savings/Stocks                 0
    Length of current employment         0
    Instalment per cent                  0
    Sex & Marital Status                 0
    Guarantors                           0
    Duration in Current address          0
    Most valuable available asset        0
    Age (years)                          0
    Concurrent Credits                   0
    Type of apartment                    0
    No of Credits at this Bank           0
    Occupation                           0
    No of dependents                     0
    Telephone                            0
    Foreign Worker                       0
    dtype: int64

```python
# 7. 컬럼별 조회
german["Purpose"].value_counts()
```

    3     280
    0     234
    2     181
    1     103
    9      97
    6      50
    5      22
    10     12
    4      12
    8       9
    Name: Purpose, dtype: int64

```python
german["Purpose"].value_counts().sort_index()
```

    0     234
    1     103
    2     181
    3     280
    4      12
    5      22
    6      50
    8       9
    9      97
    10     12
    Name: Purpose, dtype: int64

```python
german["Purpose"].value_counts().sort_values()
```

    8       9
    10     12
    4      12
    5      22
    6      50
    9      97
    1     103
    2     181
    0     234
    3     280
    Name: Purpose, dtype: int64

# 실습화일 컬럼 내용 확인?

```python
german_sample.min()  # ??? 왜 안되지
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_135528\2301447214.py in <module>
    ----> 1 german_sample.min()  # ??? 왜 안되지


    NameError: name 'german_sample' is not defined

1. german 데이터프레임에서 4개 컬럼을 뽑아서 새로운 데이터 프레임을 만든다
   - Creditability, Duration of Credit (month), Purpose, Credit Amount
   - german_sample 만들기

```python
german.columns
```

    Index(['Creditability', 'Account Balance', 'Duration of Credit (month)',
           'Payment Status of Previous Credit', 'Purpose', 'Credit Amount',
           'Value Savings/Stocks', 'Length of current employment',
           'Instalment per cent', 'Sex & Marital Status', 'Guarantors',
           'Duration in Current address', 'Most valuable available asset',
           'Age (years)', 'Concurrent Credits', 'Type of apartment',
           'No of Credits at this Bank', 'Occupation', 'No of dependents',
           'Telephone', 'Foreign Worker'],
          dtype='object')

```python
german_sample =  german[["Creditability",
                         "Duration of Credit (month)",
                         "Purpose",
                         "Credit Amount"]]
german_sample.head(3)
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
      <th>Creditability</th>
      <th>Duration of Credit (month)</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18</td>
      <td>2</td>
      <td>1049</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>2799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>12</td>
      <td>9</td>
      <td>841</td>
    </tr>
  </tbody>
</table>
</div>

```python
new_col = ["Creditability","Duration of Credit (month)",
           "Purpose","Credit Amount"]
german_sample = german[new_col]
german_sample.head(3)
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
      <th>Creditability</th>
      <th>Duration of Credit (month)</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>18</td>
      <td>2</td>
      <td>1049</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>2799</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>12</td>
      <td>9</td>
      <td>841</td>
    </tr>
  </tbody>
</table>
</div>

```python
german_sample.min()
```

    Creditability                   0
    Duration of Credit (month)      4
    Purpose                         0
    Credit Amount                 250
    dtype: int64

```python
german_sample.Creditability.value_counts()
```

    1    700
    0    300
    Name: Creditability, dtype: int64

```python
german_sample.max()
```

    Creditability                     1
    Duration of Credit (month)       72
    Purpose                          10
    Credit Amount                 18424
    dtype: int64

```python
german_sample.mean()
```

    Creditability                    0.700
    Duration of Credit (month)      20.903
    Purpose                          2.828
    Credit Amount                 3271.248
    dtype: float64

```python
# 8. 요약 통계량
german_sample.describe()
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
      <th>Creditability</th>
      <th>Duration of Credit (month)</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
      <td>1000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.700000</td>
      <td>20.903000</td>
      <td>2.828000</td>
      <td>3271.24800</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.458487</td>
      <td>12.058814</td>
      <td>2.744439</td>
      <td>2822.75176</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>250.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>1.000000</td>
      <td>1365.50000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>2.000000</td>
      <td>2319.50000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>24.000000</td>
      <td>3.000000</td>
      <td>3972.25000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>72.000000</td>
      <td>10.000000</td>
      <td>18424.00000</td>
    </tr>
  </tbody>
</table>
</div>

```python
german = pd.read_csv("c:/test/german_credit.csv")
german.head(3)
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
      <th>Creditability</th>
      <th>Account Balance</th>
      <th>Duration of Credit (month)</th>
      <th>Payment Status of Previous Credit</th>
      <th>Purpose</th>
      <th>Credit Amount</th>
      <th>Value Savings/Stocks</th>
      <th>Length of current employment</th>
      <th>Instalment per cent</th>
      <th>Sex &amp; Marital Status</th>
      <th>...</th>
      <th>Duration in Current address</th>
      <th>Most valuable available asset</th>
      <th>Age (years)</th>
      <th>Concurrent Credits</th>
      <th>Type of apartment</th>
      <th>No of Credits at this Bank</th>
      <th>Occupation</th>
      <th>No of dependents</th>
      <th>Telephone</th>
      <th>Foreign Worker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>18</td>
      <td>4</td>
      <td>2</td>
      <td>1049</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>21</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>4</td>
      <td>0</td>
      <td>2799</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>1</td>
      <td>36</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>12</td>
      <td>2</td>
      <td>9</td>
      <td>841</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>4</td>
      <td>1</td>
      <td>23</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>

```python
german = pd.read_csv("c:/test/german_credit.csv")
german_sample =  german[["Duration of Credit (month)",
                         "Credit Amount",
                         "Age (years)"]]
german_sample.head(3)
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
      <th>Duration of Credit (month)</th>
      <th>Credit Amount</th>
      <th>Age (years)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18</td>
      <td>1049</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>2799</td>
      <td>36</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12</td>
      <td>841</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>

```python
german_sample.corr()
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
      <th>Duration of Credit (month)</th>
      <th>Credit Amount</th>
      <th>Age (years)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Duration of Credit (month)</th>
      <td>1.000000</td>
      <td>0.624988</td>
      <td>-0.037550</td>
    </tr>
    <tr>
      <th>Credit Amount</th>
      <td>0.624988</td>
      <td>1.000000</td>
      <td>0.032273</td>
    </tr>
    <tr>
      <th>Age (years)</th>
      <td>-0.037550</td>
      <td>0.032273</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
german_sample.cov()
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
      <th>Duration of Credit (month)</th>
      <th>Credit Amount</th>
      <th>Age (years)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Duration of Credit (month)</th>
      <td>145.415006</td>
      <td>2.127401e+04</td>
      <td>-5.140567</td>
    </tr>
    <tr>
      <th>Credit Amount</th>
      <td>21274.007063</td>
      <td>7.967927e+06</td>
      <td>1034.202787</td>
    </tr>
    <tr>
      <th>Age (years)</th>
      <td>-5.140567</td>
      <td>1.034203e+03</td>
      <td>128.883119</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = german[["Credit Amount",
             "Purpose",
             "Type of apartment"]]
```

```python
df_grouped = df["Credit Amount"].groupby(df["Type of apartment"])
df_grouped
```

    <pandas.core.groupby.generic.SeriesGroupBy object at 0x000002FB3AFD9F10>

```python
df_grouped.mean()
```

    Type of apartment
    1    3122.553073
    2    3067.257703
    3    4881.205607
    Name: Credit Amount, dtype: float64

```python
list(df_grouped)
```

    [(1,
      0      1049
      1      2799
      2       841
      3      2122
      5      2241
             ...
      983    1882
      989    2718
      993    3966
      994    6199
      995    1987
      Name: Credit Amount, Length: 179, dtype: int64),
     (2,
      4      2171
      6      3398
      7      1361
      8      1098
      12     1957
             ...
      988     976
      990     750
      996    2303
      998    6468
      999    6350
      Name: Credit Amount, Length: 714, dtype: int64),
     (3,
      29      4796
      44      1239
      69      2032
      125     5103
      146     2964
             ...
      971     8318
      979     3386
      991    12579
      992     7511
      997    12680
      Name: Credit Amount, Length: 107, dtype: int64)]

```python
df_grouped1 = df["Credit Amount"].groupby([df["Purpose"],df["Type of apartment"]])
df_grouped1
```

    <pandas.core.groupby.generic.SeriesGroupBy object at 0x000002FB3AFD99A0>

```python
df_grouped1.mean()
```

    Purpose  Type of apartment
    0        1                    2597.225000
             2                    2811.024242
             3                    5138.689655
    1        1                    5037.086957
             2                    4915.222222
             3                    6609.923077
    2        1                    2727.354167
             2                    3107.450820
             3                    4100.181818
    3        1                    2199.763158
             2                    2540.533040
             3                    2417.333333
    4        1                    1255.500000
             2                    1546.500000
    5        1                    1522.000000
             2                    2866.000000
             3                    2750.666667
    6        1                    3156.444444
             2                    2492.423077
             3                    4387.266667
    8        1                     902.000000
             2                    1243.875000
    9        1                    5614.125000
             2                    3800.592105
             3                    4931.800000
    10       2                    8576.111111
             3                    7109.000000
    Name: Credit Amount, dtype: float64

```python
list(df_grouped1)
```

    [((0, 1),
      1      2799
      3      2122
      5      2241
      10     3905
      20     3676
      45     1216
      89     1965
      103    7432
      115    6260
      305    1469
      319    1204
      320    1597
      326    1418
      333    1283
      343    3577
      368    1403
      383     276
      464    7472
      514    3186
      550    2002
      593    1193
      714    3518
      733    2511
      755     950
      790    1778
      801    1285
      802    1371
      805    1207
      827    4843
      866    1295
      883    1264
      884    2631
      897    2570
      922    1442
      932    1223
      934    2625
      935    2235
      960    6761
      989    2718
      993    3966
      Name: Credit Amount, dtype: int64),
     ((0, 2),
      4       2171
      6       3398
      7       1361
      16      3939
      19      7228
             ...
      969     1647
      977    14896
      986     8978
      988      976
      996     2303
      Name: Credit Amount, Length: 165, dtype: int64),
     ((0, 3),
      69      2032
      208    12169
      288     2507
      347      781
      382    13756
      391     1480
      422     1175
      423     2133
      452     7308
      454     3973
      467     5324
      586     5302
      634     3249
      710     3757
      731     6527
      750     2862
      771     1422
      782    14318
      799      947
      816     1240
      855     3931
      882     7763
      902     2225
      908     1333
      910     4870
      940    10127
      943    12389
      948     1442
      997    12680
      Name: Credit Amount, dtype: int64),
     ((1, 1),
      11      6187
      39      3868
      117     2603
      201     7824
      216     2901
      270     1236
      282     4675
      341     3632
      372     7758
      389     2812
      398     1503
      402     1352
      459     2445
      558     5433
      574     6948
      682     7057
      725     2779
      732     3368
      744     4811
      753     4113
      835    11560
      867     9398
      953    11590
      Name: Credit Amount, dtype: int64),
     ((1, 2),
      38      3378
      54      6313
      61      2679
      76      3275
      83      2360
      86      2346
      121     5842
      145     2476
      177     1804
      195     7253
      222     4679
      223     8613
      232     8588
      233     3857
      239     2569
      245     3488
      249     5248
      254     5711
      274     2670
      295     8133
      309     9436
      311     2751
      338     2848
      339     1413
      342     3229
      356     2924
      377     5965
      390     3029
      406     4657
      417     5804
      460     6468
      462     3812
      466     2028
      494     1409
      497     4042
      504     7814
      507     6842
      524     1860
      531     2197
      561     3850
      572     7966
      600     8229
      608     4788
      639     2957
      693     3594
      698     4576
      701     5800
      722     2993
      723     8947
      728    11054
      739     6148
      743     7596
      891     9629
      947     7485
      Name: Credit Amount, dtype: int64),
     ((1, 3),
      29      4796
      146     2964
      152     8858
      315     1300
      355    10477
      448     4686
      585     2910
      614     6579
      616    10623
      620     9277
      622     1526
      624     6615
      642     6419
      643     6143
      700     8335
      730     9283
      734     5493
      746     2760
      814     5129
      895    12976
      918     4605
      919     6331
      928    10297
      979     3386
      991    12579
      992     7511
      Name: Credit Amount, dtype: int64),
     ((2, 1),
      0      1049
      26      652
      116    1919
      119    3062
      159    3617
      162    3972
      238     601
      257    2146
      268    1258
      291    2221
      296    2301
      298     983
      323    2132
      361    1858
      363    1388
      369    3021
      380    3650
      381    3599
      473    2186
      481    3001
      483    5801
      519    7127
      549    1402
      589    3017
      592    6229
      613    1845
      630    1433
      694    1768
      703    3622
      706    3617
      718    1924
      745    1766
      777    3384
      808    3345
      819    1980
      842    1123
      868     951
      889    9034
      904    3114
      905    2124
      907    2406
      914    4110
      916    1282
      917    2969
      925    2039
      946    2210
      957    3441
      963    3234
      Name: Credit Amount, dtype: int64),
     ((2, 2),
      70     2745
      73      929
      75     2030
      84     1520
      94     6361
             ...
      936     959
      965    9960
      978    2762
      982    5096
      999    6350
      Name: Credit Amount, Length: 122, dtype: int64),
     ((2, 3),
      163    3343
      470    1984
      584    7865
      609    3069
      625    1872
      629    2578
      631    7882
      635    3149
      749    2892
      856    3349
      909    7119
      Name: Credit Amount, dtype: int64),
     ((3, 1),
      9      3758
      14     1936
      15     2647
      17     3213
      122    2063
      150     409
      172    1297
      174    1963
      190    1901
      206    1107
      237    2384
      276    3568
      308    2606
      332    1554
      409    7166
      430    4530
      443    1386
      468    2323
      496    2108
      505    1740
      523    1126
      532    1881
      538     730
      565    1553
      580    1546
      662     585
      677    1795
      699    1231
      824    1659
      850    3031
      857    2302
      862    1534
      872     433
      878    1366
      959    3092
      983    1882
      994    6199
      995    1987
      Name: Credit Amount, dtype: int64),
     ((3, 2),
      8      1098
      12     1957
      18     2337
      25     4771
      27     1154
             ...
      972    2100
      980    2039
      984    6999
      987     674
      998    6468
      Name: Credit Amount, Length: 227, dtype: int64),
     ((3, 3),
      125    5103
      220    1940
      354    1346
      364    2279
      393    1471
      435     846
      446     700
      528    1505
      563    1377
      729    9157
      761    2600
      784    2149
      848    2671
      865    1271
      968    1845
      Name: Credit Amount, dtype: int64),
     ((4, 1),
      180    1236
      769    1275
      Name: Credit Amount, dtype: int64),
     ((4, 2),
      23     1424
      126     874
      373     343
      388    1225
      583     741
      687    3990
      705    1262
      735    1338
      772    1217
      870    3051
      Name: Credit Amount, dtype: int64),
     ((5, 1),
      22     2384
      272     660
      Name: Credit Amount, dtype: int64),
     ((5, 2),
      40       996
      248     2631
      251     6204
      336     1288
      376     1108
      378     1514
      479     3872
      480     5190
      503     2058
      692      454
      711     3394
      715     2613
      738     1308
      787     1512
      812     1943
      828      639
      975    11998
      Name: Credit Amount, dtype: int64),
     ((5, 3),
      431    1555
      747    5507
      955    1190
      Name: Credit Amount, dtype: int64),
     ((6, 1),
      164      392
      426    11760
      428     1200
      577      684
      702     8471
      713     1244
      721     1905
      783      433
      797     2319
      Name: Credit Amount, dtype: int64),
     ((6, 2),
      47     1864
      109    2012
      118     936
      141    3832
      188    2273
      191    3711
      290    3565
      330    1927
      349     701
      392    1047
      427    1501
      436    1532
      469    1393
      511    1538
      551    2096
      748    1199
      779    4623
      793     795
      821    6887
      840    1977
      864    5998
      887    8065
      924    3414
      942     719
      974     448
      990     750
      Name: Credit Amount, dtype: int64),
     ((6, 3),
      44      1239
      346      727
      453     5743
      478     2748
      517     1819
      571     9055
      644     1597
      709     6110
      716     7476
      720     6224
      752     1136
      763     6288
      809     1198
      849    12612
      874     1837
      Name: Credit Amount, dtype: int64),
     ((8, 1),
      927    902
      Name: Credit Amount, dtype: int64),
     ((8, 2),
      138     932
      275    3447
      300     339
      367    1410
      475     937
      553     894
      628    1238
      652     754
      Name: Credit Amount, dtype: int64),
     ((9, 1),
      2        841
      158     6416
      242     1568
      255     2687
      434     1264
      516     6967
      708     7685
      811     7297
      832     7980
      861     4308
      926     3161
      956     2767
      958     4280
      962    15945
      970     4844
      973    11816
      Name: Credit Amount, dtype: int64),
     ((9, 2),
      43     2825
      46     1258
      49     1382
      92     4221
      97     4455
             ...
      952    1188
      954     609
      966    8648
      981    2169
      985    2292
      Name: Credit Amount, Length: 76, dtype: int64),
     ((9, 3),
      482    3863
      654    6681
      795    1953
      826    3844
      971    8318
      Name: Credit Amount, dtype: int64),
     ((10, 2),
      13      7582
      41      1755
      456     2629
      621     6314
      645    15857
      843    11328
      844    11938
      879     1358
      976    18424
      Name: Credit Amount, dtype: int64),
     ((10, 3),
      405     5381
      591     1164
      847    14782
      Name: Credit Amount, dtype: int64)]

```python
for (purpose, type1), item in df_grouped1:
    print(purpose, " ** ", type1)
    print(item)
```

    0  **  1
    1      2799
    3      2122
    5      2241
    10     3905
    20     3676
    45     1216
    89     1965
    103    7432
    115    6260
    305    1469
    319    1204
    320    1597
    326    1418
    333    1283
    343    3577
    368    1403
    383     276
    464    7472
    514    3186
    550    2002
    593    1193
    714    3518
    733    2511
    755     950
    790    1778
    801    1285
    802    1371
    805    1207
    827    4843
    866    1295
    883    1264
    884    2631
    897    2570
    922    1442
    932    1223
    934    2625
    935    2235
    960    6761
    989    2718
    993    3966
    Name: Credit Amount, dtype: int64
    0  **  2
    4       2171
    6       3398
    7       1361
    16      3939
    19      7228
           ...
    969     1647
    977    14896
    986     8978
    988      976
    996     2303
    Name: Credit Amount, Length: 165, dtype: int64
    0  **  3
    69      2032
    208    12169
    288     2507
    347      781
    382    13756
    391     1480
    422     1175
    423     2133
    452     7308
    454     3973
    467     5324
    586     5302
    634     3249
    710     3757
    731     6527
    750     2862
    771     1422
    782    14318
    799      947
    816     1240
    855     3931
    882     7763
    902     2225
    908     1333
    910     4870
    940    10127
    943    12389
    948     1442
    997    12680
    Name: Credit Amount, dtype: int64
    1  **  1
    11      6187
    39      3868
    117     2603
    201     7824
    216     2901
    270     1236
    282     4675
    341     3632
    372     7758
    389     2812
    398     1503
    402     1352
    459     2445
    558     5433
    574     6948
    682     7057
    725     2779
    732     3368
    744     4811
    753     4113
    835    11560
    867     9398
    953    11590
    Name: Credit Amount, dtype: int64
    1  **  2
    38      3378
    54      6313
    61      2679
    76      3275
    83      2360
    86      2346
    121     5842
    145     2476
    177     1804
    195     7253
    222     4679
    223     8613
    232     8588
    233     3857
    239     2569
    245     3488
    249     5248
    254     5711
    274     2670
    295     8133
    309     9436
    311     2751
    338     2848
    339     1413
    342     3229
    356     2924
    377     5965
    390     3029
    406     4657
    417     5804
    460     6468
    462     3812
    466     2028
    494     1409
    497     4042
    504     7814
    507     6842
    524     1860
    531     2197
    561     3850
    572     7966
    600     8229
    608     4788
    639     2957
    693     3594
    698     4576
    701     5800
    722     2993
    723     8947
    728    11054
    739     6148
    743     7596
    891     9629
    947     7485
    Name: Credit Amount, dtype: int64
    1  **  3
    29      4796
    146     2964
    152     8858
    315     1300
    355    10477
    448     4686
    585     2910
    614     6579
    616    10623
    620     9277
    622     1526
    624     6615
    642     6419
    643     6143
    700     8335
    730     9283
    734     5493
    746     2760
    814     5129
    895    12976
    918     4605
    919     6331
    928    10297
    979     3386
    991    12579
    992     7511
    Name: Credit Amount, dtype: int64
    2  **  1
    0      1049
    26      652
    116    1919
    119    3062
    159    3617
    162    3972
    238     601
    257    2146
    268    1258
    291    2221
    296    2301
    298     983
    323    2132
    361    1858
    363    1388
    369    3021
    380    3650
    381    3599
    473    2186
    481    3001
    483    5801
    519    7127
    549    1402
    589    3017
    592    6229
    613    1845
    630    1433
    694    1768
    703    3622
    706    3617
    718    1924
    745    1766
    777    3384
    808    3345
    819    1980
    842    1123
    868     951
    889    9034
    904    3114
    905    2124
    907    2406
    914    4110
    916    1282
    917    2969
    925    2039
    946    2210
    957    3441
    963    3234
    Name: Credit Amount, dtype: int64
    2  **  2
    70     2745
    73      929
    75     2030
    84     1520
    94     6361
           ...
    936     959
    965    9960
    978    2762
    982    5096
    999    6350
    Name: Credit Amount, Length: 122, dtype: int64
    2  **  3
    163    3343
    470    1984
    584    7865
    609    3069
    625    1872
    629    2578
    631    7882
    635    3149
    749    2892
    856    3349
    909    7119
    Name: Credit Amount, dtype: int64
    3  **  1
    9      3758
    14     1936
    15     2647
    17     3213
    122    2063
    150     409
    172    1297
    174    1963
    190    1901
    206    1107
    237    2384
    276    3568
    308    2606
    332    1554
    409    7166
    430    4530
    443    1386
    468    2323
    496    2108
    505    1740
    523    1126
    532    1881
    538     730
    565    1553
    580    1546
    662     585
    677    1795
    699    1231
    824    1659
    850    3031
    857    2302
    862    1534
    872     433
    878    1366
    959    3092
    983    1882
    994    6199
    995    1987
    Name: Credit Amount, dtype: int64
    3  **  2
    8      1098
    12     1957
    18     2337
    25     4771
    27     1154
           ...
    972    2100
    980    2039
    984    6999
    987     674
    998    6468
    Name: Credit Amount, Length: 227, dtype: int64
    3  **  3
    125    5103
    220    1940
    354    1346
    364    2279
    393    1471
    435     846
    446     700
    528    1505
    563    1377
    729    9157
    761    2600
    784    2149
    848    2671
    865    1271
    968    1845
    Name: Credit Amount, dtype: int64
    4  **  1
    180    1236
    769    1275
    Name: Credit Amount, dtype: int64
    4  **  2
    23     1424
    126     874
    373     343
    388    1225
    583     741
    687    3990
    705    1262
    735    1338
    772    1217
    870    3051
    Name: Credit Amount, dtype: int64
    5  **  1
    22     2384
    272     660
    Name: Credit Amount, dtype: int64
    5  **  2
    40       996
    248     2631
    251     6204
    336     1288
    376     1108
    378     1514
    479     3872
    480     5190
    503     2058
    692      454
    711     3394
    715     2613
    738     1308
    787     1512
    812     1943
    828      639
    975    11998
    Name: Credit Amount, dtype: int64
    5  **  3
    431    1555
    747    5507
    955    1190
    Name: Credit Amount, dtype: int64
    6  **  1
    164      392
    426    11760
    428     1200
    577      684
    702     8471
    713     1244
    721     1905
    783      433
    797     2319
    Name: Credit Amount, dtype: int64
    6  **  2
    47     1864
    109    2012
    118     936
    141    3832
    188    2273
    191    3711
    290    3565
    330    1927
    349     701
    392    1047
    427    1501
    436    1532
    469    1393
    511    1538
    551    2096
    748    1199
    779    4623
    793     795
    821    6887
    840    1977
    864    5998
    887    8065
    924    3414
    942     719
    974     448
    990     750
    Name: Credit Amount, dtype: int64
    6  **  3
    44      1239
    346      727
    453     5743
    478     2748
    517     1819
    571     9055
    644     1597
    709     6110
    716     7476
    720     6224
    752     1136
    763     6288
    809     1198
    849    12612
    874     1837
    Name: Credit Amount, dtype: int64
    8  **  1
    927    902
    Name: Credit Amount, dtype: int64
    8  **  2
    138     932
    275    3447
    300     339
    367    1410
    475     937
    553     894
    628    1238
    652     754
    Name: Credit Amount, dtype: int64
    9  **  1
    2        841
    158     6416
    242     1568
    255     2687
    434     1264
    516     6967
    708     7685
    811     7297
    832     7980
    861     4308
    926     3161
    956     2767
    958     4280
    962    15945
    970     4844
    973    11816
    Name: Credit Amount, dtype: int64
    9  **  2
    43     2825
    46     1258
    49     1382
    92     4221
    97     4455
           ...
    952    1188
    954     609
    966    8648
    981    2169
    985    2292
    Name: Credit Amount, Length: 76, dtype: int64
    9  **  3
    482    3863
    654    6681
    795    1953
    826    3844
    971    8318
    Name: Credit Amount, dtype: int64
    10  **  2
    13      7582
    41      1755
    456     2629
    621     6314
    645    15857
    843    11328
    844    11938
    879     1358
    976    18424
    Name: Credit Amount, dtype: int64
    10  **  3
    405     5381
    591     1164
    847    14782
    Name: Credit Amount, dtype: int64
