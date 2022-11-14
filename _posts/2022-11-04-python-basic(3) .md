---
layout: single
title: "2202.11.04.파이썬 기초 데이터처리 기술(3)"
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
    <li>2. 파이썬 크롤링</li>
    <li>3. 파이썬 api</li>
</ul>
</div>

## 소개

안녕하세요 인공지능 개발자를 지망하는 윤동민입니다. 지금까지 파이썬 데이터 처리와 크롤링 등은 많이 해봤지만, 머닝러신, 딥러닝 쪽은 이론 공부만 해봤지 실제 구현은 해본적이 없습니다. 때문에 이번기회에 지금까지 배운것을 쭉 정리하고, 인공지능을 공부해 볼려고 합니다. 지금은 부족하지만, 앞으로 성장해가는 모습을 보여드리겠습니다.

## 개요

여러 카테고리를 만들어 파이썬과 인공지능에 대해 공부해 나갈 겁니다. 이 중 python_basic는 지금까지 제가 배운 내용을 정리해서 복습하는 기회로 소개해드리겠습니다.

## 파이썬 기초 데이터처리 기술(1)

1. 내장함수
2. 내장함수\_datetime
3. 함수만들기\_실습
4. 클래스\_모듈
5. pandas\_기초공부

#### 1. 내장함수

```python
from datetime import date
```

```python
today = date.today();today
```

    datetime.date(2022, 11, 14)

```python
import time
```

```python
today == date.fromtimestamp(time.time())
```

    True

```python
my_birthday = date(today.year, 6, 24)
my_birthday
```

    datetime.date(2022, 6, 24)

```python
if my_birthday < today:
    my_birtyday = my_birthday.replace(year=today.year + 1)
```

```python
my_birtyday
```

    datetime.date(2023, 6, 24)

```python
abs(my_birthday - today )
```

    datetime.timedelta(days=143)

```python
type(today)
```

    datetime.date

```python
today.strftime("%Y%m%d")
```

    '20221114'

```python
today.strftime("%Y-%m-%d")
```

    '2022-11-14'

```python
today.strftime("%y-%m-%d")
```

    '22-11-14'

```python

```

#### 2. 내장함수\_datetime

```python
from datetime import date
```

```python
today = date.today();today
```

    datetime.date(2022, 11, 14)

```python
import time
```

```python
today == date.fromtimestamp(time.time())
```

    True

```python
my_birthday = date(today.year, 6, 24)
my_birthday
```

    datetime.date(2022, 6, 24)

```python
if my_birthday < today:
    my_birtyday = my_birthday.replace(year=today.year + 1)
```

```python
my_birtyday
```

    datetime.date(2023, 6, 24)

```python
abs(my_birthday - today )
```

    datetime.timedelta(days=143)

```python
type(today)
```

    datetime.date

```python
today.strftime("%Y%m%d")
```

    '20221114'

```python
today.strftime("%Y-%m-%d")
```

    '2022-11-14'

```python
today.strftime("%y-%m-%d")
```

    '22-11-14'

```python

```

#### 3. 함수만들기\_실습

### (실습. 섭씨 온도를 화씨로 계산하는 함수(conv_fahr) 만들기 )

1. 화씨 = ( ( 9/5) \* 섭씨 ) + 32
   - 섭씨 25도는 화씨 77 도
2. 함수로 화씨를 변환 하여 결과를 받아서
3. 화면에서 섭씨 온도를 입력 받아
4. 화면에 출력

```python
# 1. 계산 확인
a = 25
b = ((9/5)*a) + 32
print(b)
```

    60.8

```python
# 2. 함수 만들기
def conv_fahr(a):
    b = ((9/5) * a) + 32
    return b
```

```python
conv_fahr(25)
```

    77.0

```python
# 3. 함수 사용하기
x = 25
y = conv_fahr(x)
print(y)
```

    77.0

```python
# 4. 화면에서 입력 받아 함수 처리

def conv_fahr(a):
    b = ((9/5) * a) + 32
    return b

x = float(input("섭씨? "))
y = conv_fahr(x)
print("섭씨 %s는 화씨 %s 입니다"%(x,y))
```

    섭씨? 78
    섭씨 78.0는 화씨 172.4 입니다

## (실습 \_ 여러 데이터 처리)

x=[25, 20, 32 ,77]

```python
# 1. 처리할 데이터 확인 하기
x = [25, 20, 32, 77]

for i in x:
    print(i)
```

    25
    20
    32
    77

```python
y1 = []
for i in x:
    y = conv_fahr(i)
    y1.append(y)
```

```python
def myFunc():
    return "hello", 100, ['a', 'b']

first, second, third = myFunc()

print(first)
print(second)
print(third)
```

    hello
    100
    ['a', 'b']

## (실습. 리스트컴프리헨션으로 구현)

```python
y1 = [ conv_fahr(i) for i in x ]
print(y1)
```

    [77.0, 68.0, 89.6, 170.6]

```python
x
```

    [25, 20, 32, 77]

```python
y1
```

    [77.0, 68.0, 89.6, 170.6]

```python
[conv_fahr(a) for a in [25, 20, 32 ,77]]
```

    [77.0, 68.0, 89.6, 170.6]

```python
#실습 시작
# 방법 1
y2 = []
for i in x:
    y = conv_fahr(i)
    y2.append([i,y])
    print("i = ", i, "**", y2)
```

    i =  25 ** [[25, 77.0]]
    i =  20 ** [[25, 77.0], [20, 68.0]]
    i =  32 ** [[25, 77.0], [20, 68.0], [32, 89.6]]
    i =  77 ** [[25, 77.0], [20, 68.0], [32, 89.6], [77, 170.6]]

(실습)
[[25,77.0],[20,68.0]]

```python
# 방법 2
y2 = [[i,conv_fahr(i)] for i in x ]
y2
```

    [[25, 77.0], [20, 68.0], [32, 89.6], [77, 170.6]]

```python
# 방법 3
y2 = []
for i in zip(x,y1):
    y2.append(i)
    print(y2)
```

    [(25, 77.0)]
    [(25, 77.0), (20, 68.0)]
    [(25, 77.0), (20, 68.0), (32, 89.6)]
    [(25, 77.0), (20, 68.0), (32, 89.6), (77, 170.6)]

```python
# 방법 4
[i for i in zip(x,y1)]
```

    [(25, 77.0), (20, 68.0), (32, 89.6), (77, 170.6)]

```python
# 방법 5
[list(i) for i in zip(x,y1)]
```

    [[25, 77.0], [20, 68.0], [32, 89.6], [77, 170.6]]

## ( 실습. 딕셔너리 형식으로 데이터 만들기 )

```python
aa={}
aa[a] =123
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_107996\2036382021.py in <module>
          1 aa={}
    ----> 2 aa[a] =123


    NameError: name 'a' is not defined

```python
aa
```

    {16: 123}

```python
# 1.
y5 = {}
for i in x:
    y = conv_fahr(i)
    y5[i] = y
    print(y5)
```

    {25: 77.0}
    {25: 77.0, 20: 68.0}
    {25: 77.0, 20: 68.0, 32: 89.6}
    {25: 77.0, 20: 68.0, 32: 89.6, 77: 170.6}

```python
y5[20]
```

    68.0

```python
#2.
y6 = [list(i) for i in zip(x,y1)]; y6
```

    [[25, 77.0], [20, 68.0], [32, 89.6], [77, 170.6]]

```python
dict(y6)
```

    {25: 77.0, 20: 68.0, 32: 89.6, 77: 170.6}

#### 4. 클래스\_모듈

(클래스)

```python
class Calculator:
    txt="문자"
    def sum(self, a, b):
        c = a+b
        return c
```

```python
x = Calculator()
```

```python
x.sum(3,4)
```

    7

```python
x.txt

```

    '문자'

```python
class Family():
    def father(self):
        print("아빠")
    def mother(self):
        print("엄마")
    def son(self):
        print("아들")

class Look():
    def hansom(self):
        print("잘 생겼다")
    def pretty(self):
        print("이쁘다")
    def slim(self):
        print("날씬하다")

```

```python
we = Family()
```

```python
we.father()
```

    아빠

```python
class Employee:
    def __init__(self, name, salary):
        print("init!!!")
        self.name = name
        self.salary = salary
    def displayCount(self):
        print("hello")
```

```python
x = Employee("길동",30)
```

    init!!!

```python
dir(x)
```

    ['__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     'displayCount',
     'name',
     'salary']

```python
x.name
```

    '길동'

```python
x.salary
```

    30

```python
y = Employee("우영우", 30)
```

    init!!!

```python
y.name
```

    '우영우'

(패키지 모듈)

```python
import seaborn as sns
```

```python
import seaborn
```

```python
import matplotlib.pyplot as plt
```

```python
from matplotlib import pyplot as plt
```

```python
import matplotlib
```

# (패키지 설치)

1. pip install 패키지명
2. conda install 패키지명

### (실습. krx 주식 데이터 처리)

```python
from pykrx import stock
import pandas
```

```python
tickers = stock.get_market_ticker_list("20221104")
print(tickers[:10])
```

    ['095570', '006840', '027410', '282330', '138930', '001460', '001465', '001040', '079160', '00104K']

```python
len(tickers)
```

    941

```python
from pykrx import stock
```

```python
tickers = pykrx.stock.get_market_ticker_list("20221104") #??????
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_104692\599908128.py in <module>
    ----> 1 tickers = pykrx.stock.get_market_ticker_list("20221104") #??????


    NameError: name 'pykrx' is not defined

```python
pandas.DataFrame([])
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>

```python
from pandas import DataFrame
```

```python
DataFrame([])
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>

```python
pandas.DataFrame([])
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>

### ( 실습 시작)

```python
import selenium
```

```python
%%writefile mod1.py

class MyClass:
    def __init__(self):
        self.size=5

length=10

def sum(a,b):
    return a+b
def mult(a,b):
    return a*b
```

    Writing mod1.py

```python
import mod1
```

```python
dir(mod1)
```

    ['MyClass',
     '__builtins__',
     '__cached__',
     '__doc__',
     '__file__',
     '__loader__',
     '__name__',
     '__package__',
     '__spec__',
     'length',
     'mult',
     'sum']

```python
mod1.length
```

    10

```python
mod1.sum(2,3)
```

    5

```python
a = mod1.MyClass()
```

```python
a.size
```

    5

```python
import mod1 as md
```

```python
md.sum(1,3)
```

    4

```python
%%writefile c:/test/mod2.py

class MyClass:
    def __init__(self):
        self.size=5

length=10

def sum(a,b):
    return a+b
def mult(a,b):
    return a*b
```

    Writing c:/test/mod2.py

```python
import sys
sys.path.append("c:/test/")

import mod2
```

```python
if __name__ == "__main__":
    print("모듈 시작")
    print("__name__ = ", __name__)
    print("모듈 종료")
else:
    print("다른 모듈에 imported 되어 사용됨")
    print("__name__이 __main__ 이 아닌")
```

    모듈 시작
    __name__ =  __main__
    모듈 종료

```python
%%writefile main.py

def aa():
    print("AAAA")
if __name__ == "__main__":
    print("모듈 시작")
    print("__name__ = ", __name__)
    print("모듈 종료")
    print(aa())
else:
    print("다른 모듈에 imported 되어 사용됨")
    print("__name__이 __main__ 이 아닌")
```

    Writing main.py

```python
# 1. 직접 main.py 실행
%run main.py
```

    모듈 시작
    __name__ =  __main__
    모듈 종료
    AAAA
    None

```python
# 2. import로 확인
import main
```

    다른 모듈에 imported 되어 사용됨
    __name__이 __main__ 이 아닌

```python
main.aa()
```

    AAAA

```python
%%writefile calc.py
def add(a,b):
    return a+b
def mul(a,b):
    return a*b
def sub(a,b):
    return a-b

if __name__ == "__main__":
    print("** main **")
    print(add(10,20))
    print(mul(10,20))
    print(sub(10,20))
else:
    print("imported 동작")
```

    Writing calc.py

```python
!python calc.py
```

    ** main **
    30
    200
    -10

```python
import calc
calc.add(3,5)
```

    imported 동작
    8

```python

```

#### 5. pandas\_기초공부

```python
from pandas import Series, DataFrame
import pandas as pd
```

```python
print("ver = ", pd.__version__)
```

    ver =  1.4.4

1. Series 데이터 생성

```python
# Series 생성 1
fruit = Series(data=[2500,3800,1200,6000])
fruit
```

    0    2500
    1    3800
    2    1200
    3    6000
    dtype: int64

```python
fruit = Series(data=[2500,3800,1200,6000],
               index=["ap","ba","pe","ch"])
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
fruit.values
```

    array([2500, 3800, 1200, 6000], dtype=int64)

```python
fruit.index
```

    Index(['ap', 'ba', 'pe', 'ch'], dtype='object')

```python
# Series 생성 2

data = {"ap":2500, "ba":3800, "pe":1200, "ch":6000};data
```

    {'ap': 2500, 'ba': 3800, 'pe': 1200, 'ch': 6000}

```python
fruit = Series(data)
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
type(data)
```

    dict

```python
type(fruit)
```

    pandas.core.series.Series

```python
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
# Series 이름 붙이기
fruit.name = "fruitPrice"
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    Name: fruitPrice, dtype: int64

```python
fruit.index.name = "fruitName"
fruit
```

    fruitName
    ap    2500
    ba    3800
    pe    1200
    ch    6000
    Name: fruitPrice, dtype: int64

```python
fruit.index.name
```

    'fruitName'

Series()

```python
### 참고
a = [1,2,3,4]
a
```

    [1, 2, 3, 4]

```python
b = a;
b
```

    [1, 2, 3, 4]

```python
b[1] = 999
```

```python
b
```

    [1, 999, 3, 4]

```python
a
```

    [1, 999, 3, 4]

```python
c = a.copy()
```

```python
c[1] = 999
c
```

    [1, 999, 3, 4]

```python
a
```

    [1, 999, 3, 4]

2. DataFrame 생성

```python
data = {"name":["ap","ba","ch","pe"],
        "price":[2500,3800,6000,1200],
        "num":[10,5,3,8]}
data
```

    {'name': ['ap', 'ba', 'ch', 'pe'],
     'price': [2500, 3800, 6000, 1200],
     'num': [10, 5, 3, 8]}

```python
df = DataFrame(data)
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
      <th>name</th>
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ap</td>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ba</td>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ch</td>
      <td>6000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pe</td>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = DataFrame(data, columns=["price","num11","name"])
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
      <th>price</th>
      <th>num11</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>NaN</td>
      <td>ap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3800</td>
      <td>NaN</td>
      <td>ba</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6000</td>
      <td>NaN</td>
      <td>ch</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
      <td>NaN</td>
      <td>pe</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = DataFrame(data, columns=["price","num","name"])
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
      <th>price</th>
      <th>num</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>10</td>
      <td>ap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3800</td>
      <td>5</td>
      <td>ba</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6000</td>
      <td>3</td>
      <td>ch</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
      <td>8</td>
      <td>pe</td>
    </tr>
  </tbody>
</table>
</div>

```python
# 특정 항목 추출
df["name"]
```

    0    ap
    1    ba
    2    ch
    3    pe
    Name: name, dtype: object

```python
df.name
```

    0    ap
    1    ba
    2    ch
    3    pe
    Name: name, dtype: object

```python
# 컬럼 추가
df["year"] = 2022
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
      <th>price</th>
      <th>num</th>
      <th>name</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>10</td>
      <td>ap</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3800</td>
      <td>5</td>
      <td>ba</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6000</td>
      <td>3</td>
      <td>ch</td>
      <td>2022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
      <td>8</td>
      <td>pe</td>
      <td>2022</td>
    </tr>
  </tbody>
</table>
</div>

```python
variable = Series([4,2,1], index=[0,2,3])
variable
```

    0    4
    2    2
    3    1
    dtype: int64

```python
df["stock"] = variable
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
      <th>price</th>
      <th>num</th>
      <th>name</th>
      <th>year</th>
      <th>stock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>10</td>
      <td>ap</td>
      <td>2022</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3800</td>
      <td>5</td>
      <td>ba</td>
      <td>2022</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6000</td>
      <td>3</td>
      <td>ch</td>
      <td>2022</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
      <td>8</td>
      <td>pe</td>
      <td>2022</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

# 항목 삭제

```python
fruit = Series(data=[2500,3800,1200,6000],
               index=["ap","ba","pe","ch"])
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
fruit.drop("ba")
```

    ap    2500
    pe    1200
    ch    6000
    dtype: int64

```python
new_fruit = fruit.drop("ba")
```

```python
print(fruit)
print("-"*40)
print(new_fruit)
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64
    ----------------------------------------
    ap    2500
    pe    1200
    ch    6000
    dtype: int64

# dataframe 삭제

```python
data = {"name":["ap","ba","ch","pe"],
        "price":[2500,3800,6000,1200],
        "num":[10,5,3,8]}

df = DataFrame(data, columns=["price","num","name"])
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
      <th>price</th>
      <th>num</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2500</td>
      <td>10</td>
      <td>ap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3800</td>
      <td>5</td>
      <td>ba</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6000</td>
      <td>3</td>
      <td>ch</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200</td>
      <td>8</td>
      <td>pe</td>
    </tr>
  </tbody>
</table>
</div>

```python
data = {"name":["ap","ba","ch","pe"],
        "price":[2500,3800,6000,1200],
        "num":[10,5,3,8]}
name = data["name"]
df1 = DataFrame(data, index=name, columns=["price","num"])
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
df2 = df1.drop(["ap","ch"])
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
df1.drop("num", axis=1)
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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>

```python
df3 = df1.drop("num", axis=1)
df3
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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>

```python
df4 = df1.copy()
df4.drop("num",axis=1, inplace=True)
df4
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
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>

# 항목 추출 slice

```python
fruit = Series(data=[2500,3800,1200,6000],
               index=["ap","ba","pe","ch"])
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
fruit["ap":"pe"]
```

    ap    2500
    ba    3800
    pe    1200
    dtype: int64

```python
data = {"name":["ap","ba","ch","pe"],
        "price":[2500,3800,6000,1200],
        "num":[10,5,3,8]}
name = data["name"]
df = DataFrame(data, index=name, columns=["price","num"])
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
df["price"]
```

    ap    2500
    ba    3800
    ch    6000
    pe    1200
    Name: price, dtype: int64

```python
df["ap":"ba"]
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

# Series 연산

```python
fruit1 = Series(data=[5,9,10,3],
               index=["ap","ba","pe","ch"])
fruit2 = Series(data=[3,2,9,5,10],
               index=["ap","or","ba","ch","ma"])
print(fruit1)
print("-"*40)
print(fruit2)
```

    ap     5
    ba     9
    pe    10
    ch     3
    dtype: int64
    ----------------------------------------
    ap     3
    or     2
    ba     9
    ch     5
    ma    10
    dtype: int64

```python
fruit1 + fruit2
```

    ap     8.0
    ba    18.0
    ch     8.0
    ma     NaN
    or     NaN
    pe     NaN
    dtype: float64

```python
df1 = DataFrame({"oh":[4,8,3,5], "te":[0,1,2,3]},
                columns = ["oh","te"],
                index = ["ap","ba","ch","pe"]); print(df1)
print("-"*50)
df2 = DataFrame({"oh":[3,0,2,1,7], "co":[5,4,3,6,0]},
                columns = ["oh","co"],
                index = ["ap","or","ba","ch","ma"]); print(df2)
```

        oh  te
    ap   4   0
    ba   8   1
    ch   3   2
    pe   5   3
    --------------------------------------------------
        oh  co
    ap   3   5
    or   0   4
    ba   2   3
    ch   1   6
    ma   7   0

```python
df1 + df2
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
      <th>co</th>
      <th>oh</th>
      <th>te</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>NaN</td>
      <td>7.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>NaN</td>
      <td>10.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ma</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>or</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

# 정렬

```python
fruit
```

    ap    2500
    ba    3800
    pe    1200
    ch    6000
    dtype: int64

```python
fruit.sort_values(ascending=False)
```

    ch    6000
    ba    3800
    ap    2500
    pe    1200
    dtype: int64

```python
fruit.sort_index()
```

    ap    2500
    ba    3800
    ch    6000
    pe    1200
    dtype: int64

```python
df.sort_index()
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.sort_index(axis=1)
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
      <th>num</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ap</th>
      <td>10</td>
      <td>2500</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>5</td>
      <td>3800</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>3</td>
      <td>6000</td>
    </tr>
    <tr>
      <th>pe</th>
      <td>8</td>
      <td>1200</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.sort_values(["price"])
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
      <th>price</th>
      <th>num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pe</th>
      <td>1200</td>
      <td>8</td>
    </tr>
    <tr>
      <th>ap</th>
      <td>2500</td>
      <td>10</td>
    </tr>
    <tr>
      <th>ba</th>
      <td>3800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>ch</th>
      <td>6000</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
