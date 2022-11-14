---
layout: single
title: "2202.11.02.파이썬 기초 데이터처리 기술(2)"
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

1. 데이터 입출력
2. 함수 모듈
3. 조건문 반복문
4. 조건문 반복문 실습
5. 데이터 입출력 실습

#### 1. 데이터 입출력

```python
# 데이터 출력 1
fp = open("c:/test/frut.txt", "wt")
fp

fruit_price = {"ap":3500, "pe":2500, "ch":5000}

for item in fruit_price.items():
    print(item, file=fp)

fp.close()
```

```python
#데이터 출력 2
#   \화이트 스페이스
fp = open("c:/test/frut2.txt", "wt")
fp.write("ap 3500\n")
fp.write("pe 2500\n")
fp.write("ch 5000\n")
fp.close()
```

```python
#데이터 출력 3
#   \화이트 스페이스
fp = open("c:/test/frut3.txt", "wt")
fp.write("ap 3500")
fp.write("pe 2500")
fp.write("ch 5000")
fp.close()
```

```python
#데이터 출력 4
fp = open("c:/test/frut4.txt", "wt")
fp.write("ap 3500\t")
fp.write("pe 2500\t")
fp.write("ch 5000\t")
fp.close()
```

```python
#데이터 출력 5, csv 형식 출력
import csv

with open("c:/test/fruit5.csv","w", newline="") as fp:
    writer = csv.writer(fp, delimiter=",")
    writer.writerow(["ap"])
    writer.writerow(["pe"])
    writer.writerow(["ch"])
    writer.writerow(["ba"])
```

```python
fp = open("c:/test/frut.txt", "rt")
fp.readlines()
```

    ['ap 3500\n', 'pe 2500\n', 'ch 5000\n']

```python
fp
```

    <_io.TextIOWrapper name='c:/test/frut.txt' mode='rt' encoding='cp949'>

```python
## csv 파일 읽기

import csv
fp = open("c:/test/fruit5.csv", "rt")

```

```python
csvread = csv.reader(fp)

fruit = []
for i in csvread:
    fruit.append(csvread)
    print(i)
```

    ['ap']
    ['pe']
    ['ch']
    ['ba']

```python
fruit = []
for i in csv.reader(fp):
    fruit.append(csv.reader(fp))
    print(i)
```

```python

```

#### 2. 함수 모듈

### (함수)

1. 인자없이 실행되는 함수

```python
def hello():
    print("함수 실행!!!")
```

```python
hello()
```

    함수 실행!!!

```python
def hello():
    print("함수 실행!!!")
    for i in range(0:10)

```

      File "C:\Users\ehdal\AppData\Local\Temp\ipykernel_38756\1444227468.py", line 3
        for i in range(0:10)
                        ^
    SyntaxError: invalid syntax

```python
def sum(a,b):
    c= a+b
    print(c)
```

```python
a=3; b=2
sum(a,b)
```

    5

```python
res=sum(a,b)
print(res)
```

    5
    None

```python
def sum1(a,b):
    c = a+b
    print("def c=",c)
    return a+b
```

```python
x = 3; y = 2
sum1(x,y)
```

    def c= 5





    5

```python
res1 = sum1(x,y)
```

    def c= 5

```python
res1
```

    5

```python
sum1(30,50)
```

    def c= 80





    80

```python
def sum_many(*args):
    print(args)
    sum=0
    for i in args:
        sum = sum + i
    return sum
```

```python
def sum_many1(*aa):
    print(aa)
    sum = 0
    for i in aa:
        sum = sum + i
    return sum
```

```python
sum_many(1,2,3,4,5)
```

    15

```python
sum_many(1,2,3,4,5,6,7,8,9,10)
```

    55

```python
sum_many1(1,3,5,7)
```

    (1, 3, 5, 7)





    16

keyword arg 처리방식

```python
def person(**kwargs):
    print(kwargs)
```

```python
person(이름="woo", 키 =176)
```

    {'이름': 'woo', '키': 176}

```python
def person1(**kwargs):
    for key, value in kwargs.items():
        print("{} is {}".format(key,value))
```

```python
person1(이름="woo")
```

    이름 is woo

```python
def std_arg(arg):
    print("std_arg=", arg)
def pos_only_arg(arg, /):
    print("pos_only_arg=", arg)
def kwd_only_arg(*, arg):
    print(arg)
def combined_example(pos_only, /, standard, *, kwd_only):
    print(pos_only, standard, kwd_only)
```

```python
std_arg(a=2)
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_38756\1052478511.py in <module>
    ----> 1 std_arg(a=2)


    TypeError: std_arg() got an unexpected keyword argument 'a'

```python
pos_only_arg(1)
```

    pos_only_arg= 1

```python
pos_only_arg(arg=1)
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_38756\2836004375.py in <module>
    ----> 1 pos_only_arg(arg=1)


    TypeError: pos_only_arg() got some positional-only arguments passed as keyword arguments: 'arg'

```python
kwd_only_arg(arg=3)
```

    3

```python
pos_only_arg(arg=1)
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_38756\2836004375.py in <module>
    ----> 1 pos_only_arg(arg=1)


    TypeError: pos_only_arg() got some positional-only arguments passed as keyword arguments: 'arg'

```python
combined_example(1,2,kwd_only=3)
```

    1 2 3

```python

```

#### 3. 조건문 반복문

## (조건문)

1. if Then

```python
python = True
if python:
    print("T")
```

    T

```python
# 2.if Then Else
python = True
if python:
    print("True")
else:
    print("False")
```

    True

```python
python = -1
if python:
    print("True")
else:
    print("False")
```

    True

```python
python = True
pycharm = True
if python:
    print("1st True")
    print("11111")
elif pycharm:
    print("1st False, 2nd True")
else:
    print("False")
```

    1st True
    11111

```python

```

#### 4. 조건문 반복문 실습

## (실습\_01. 조건문 홀수 짝수 판별 )

1. 화면에서 숫자를 입력 받아 변수 x에 저장한다
2. 입력받은 숫자가 홀수면 "홀수", 짝수면 "짝수를 출력
3. 출력예) 3은 홀수 입니다.

```python
# 화면에서 입력 받을 때
x = int(input("숫자? "))
type(x)
if (x%2 ==0):
    print(x, "는 짝수")
else:
    print(x, "는 홀수")
```

    숫자? 12
    12 는 짝수

```python
# 이것도 됨
if (x%2 ==1):
    print(x, "는 홀수")
else:
    print(x, "는 짝수")
```

    12 는 짝수

```python
# 요것도 됨
if (x%2 != 0):
    print(x, "는 홀수")
else:
    print(x, "는 짝수")
```

    12 는 짝수

## (실습\_02. 성적 등급 판정 )

    화면에서 성적을 입력 받아 변수 score에 저장
    입력받은 숫자가
        80점 이상이면 "A"등급
        70점 이상 ~ 80점 미만이면 "B" 등급
        60점 이상 ~ 70점 미만이면 "C" 등급
        60점 미만이면 "D" 등급
    출력 예) 점수는 50점, 등급은 D등급

```python
score = int(input("점수? "))
if score > 80:
    degree = "A"
elif score > 70:
    degree = "B"
elif score > 60:
    degree = "C"
else:
    degree = "D"

print("점수는 ", score, "점, 등급은 = ", degree, "등급")
```

    점수? 34
    점수는  34 점, 등급은 =  D 등급

## ( 실습. 리스트 반복문 )

    score = [60, 73, 65, 89]
    채점
        80 보다 크면 "A"
        70 보다 크고 80보다 작거나 같으면 "B"
        60 보다 크고 70보다 작거나 같으면 "C"
        60 보다 작으면 "D"
    출력 -> 1번 학생 점수는 60점, 등급은 D등급

```python
scores = [60, 73, 65, 89]

no = 0

for score in scores:

    if score >= 80:
        degree = "A"
    elif (score > 60 and score <=70):
        degree = "C"
    elif (score > 70 and score <=80):
        degree = "B"
    else:
        degree = "D"

    no = no + 1

    print(no, "학생의 점수는 ", score, "점, 등급은 = ", degree, "등급")

#    break
```

    1 학생의 점수는  60 점, 등급은 =  D 등급
    2 학생의 점수는  73 점, 등급은 =  B 등급
    3 학생의 점수는  65 점, 등급은 =  C 등급
    4 학생의 점수는  89 점, 등급은 =  A 등급

## ( 과일 가격 찾기 )

    fruit_dict = {"ap":3500, "pe":2500, "ch":5000}
    과일가격이 2500원이 과일은?

```python
fruit_dict = {"ap":3500, "pe":2500, "ch":5000, "or":2500}

for item in fruit_dict.items():
#     print(item)
#     print(item[1])
    if item[1] == 2500:
        print("2500원 과일은 ", item[0], " 이다!!")
```

    2500원 과일은  pe  이다!!
    2500원 과일은  or  이다!!

## ( 과일명 찾기 )

    fruit_dict = {"ap":3500, "pe":2500, "ch":5000, "or":2000}
    3000원 이상인 과일을 찾아서 high_price 에 넣고 (리스트형)
    3000원 이하인 과일은 찾아서 low_price에 넣고 (리스트형)
        리스트변수에 추가하기
    high_price, low_price를 출력하세요.

```python
fruit_dict = {"ap":3500, "pe":2500, "ch":5000, "or":2000}
high_price=[]
low_price=[]

for item in fruit_dict.items():
    if item[1] > 3000:
        high_price.append(item[0])
    else:
        low_price.append(item[0])

print("high = ", high_price)
print("low = ", low_price)
```

    high =  ['ap', 'ch']
    low =  ['pe', 'or']

## ( 과일 갯수 찾기 )

    fruit_dict = {"ap":[3500,10], "pe":[2500,5], "ch":[5000,3], "or":[2000,5]}
    3000원 이상인 과일 갯수 high 변수에 누적
    3000원 이하인 과일 갯수 low 변수에 누적
    high, low를 출력하세요.

```python
fruit_dict = {"ap":[3500,10], "pe":[2500,5], "ch":[5000,3], "or":[2000,5]}
high=0
low=0

for item in fruit_dict.items():
    if item[1][0] > 3000:
        high = high + item[1][1]
    else:
        low = low + item[1][1]

print("high = ", high)
print("low = ", low)
```

    high =  13
    low =  10

## jupyter notebook 매직 커맨드

    1 ipynb 화일 실행
    2 .py 실행
        %run *.py
        !python *.py
        직접 폴더에서 실행 -> python *.py

```python
%%writefile c:/test/test.py

fruit_dict = {"ap":[3500,10], "pe":[2500,5], "ch":[5000,3], "or":[2000,5]}
high=0 ; low=0

for item in fruit_dict.items():
    if item[1][0] > 3000:
        high = high + item[1][1]
    else:
        low = low + item[1][1]

print("high = ", high)
print("low = ", low)
```

    Overwriting c:/test/test.py

```python
%run c:/test/test.py
```

    high =  13
    low =  10

```python
!python c:/test/test.py
```

    high =  13
    low =  10

### ( 리스트 컴프리헨션 )

### 아직도 리스트 컴프리헨션은 어려워서 찾아서 하네요

```python
# 전체 요소에 제곱된 리스트 만들기
square = []
for i in num:
    square.append(i**2)
print(square)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_102520\3854347002.py in <module>
          1 # 전체 요소에 제곱된 리스트 만들기
          2 square = []
    ----> 3 for i in num:
          4     square.append(i**2)
          5 print(square)


    NameError: name 'num' is not defined

```python
[i**2 for i in num]
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_102520\3957084216.py in <module>
    ----> 1 [i**2 for i in num]


    NameError: name 'num' is not defined

```python
# num의 5보다 >= 요소에 한해 제곱을 적용
square = []
for i in num:
    if i >= 5:
        square.append(i**2)
print(square)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_102520\1432341150.py in <module>
          1 # num의 5보다 >= 요소에 한해 제곱을 적용
          2 square = []
    ----> 3 for i in num:
          4     if i >= 5:
          5         square.append(i**2)


    NameError: name 'num' is not defined

```python
[i**2 for i in num if i >= 5 ]
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_102520\828959731.py in <module>
    ----> 1 [i**2 for i in num if i >= 5 ]


    NameError: name 'num' is not defined

```python

```

#### 5. 데이터 입출력 실습

## ( 실습\_02. 슬라이싱으로 문자열 나누기 )

    주민등록 번호에서 생년월일을 추출 하시오.
    변수에 주민등록번호를 넣는다
    변수에서 생년월일을 추출한다.(slicing)

```python
sn = "910625-1005822"
```

```python
sn[:6]
```

    '910625'

```python
sn.split("-")[0]
```

    '910625'

```python
yymmdd = sn[:6];yymmdd
```

    '910625'

## ( 실습\_03. 슬라이싱으로 문자열 나누기 )

    변수 a에는 “20010331Rainy” 가 있다.
    날짜는 date변수에,기온은 weather에 나누어 넣으시오.

```python
a = "20010331Rainy"
```

```python
date = a[:8]
```

```python
weather = a[8:]
```

```python
print(date)
print(weather)
```

    20010331
    Rainy

### 날짜는 2001년 3월 31일 입니다.

### 날씨는 Rainy 입니다.

```python
# 년
date[:4]
```

    '2001'

```python
# 월
date[4:6]
```

    '03'

```python
# 일
date[6:]
```

    '31'

```python
print("날짜는 ", date[:4],"년", date[4:6],"월", date[6:],"일 입니다")
```

    날짜는  2001 년 03 월 31 일 입니다

```python
print("날짜는 {}년 {}월 {}일 입니다".format(date[:4], date[4:6], date[6:]))
```

    날짜는 2001년 03월 31일 입니다

```python
print("날짜는 {1}월 {2}월 {0}년 입니다".format(date[:4], date[4:6], date[6:]))
```

    날짜는 03월 31월 2001년 입니다

```python
# 다양하게 쓸수 있다
print("날짜는 %s년 %s월 %s일 입니다"%(date[:4], date[4:6], date[6:]))
```

    날짜는 2001년 03월 31일 입니다

```python
print("날짜는 %d년 %d월 %d일 입니다"%(int(date[:4]), int(date[4:6]), int(date[6:])))
```

    날짜는 2001년 3월 31일 입니다

```python
print("날짜는 "+ date[:4]+"년", date[4:6],"월", date[6:],"일 입니다")
```

    날짜는 2001년 03 월 31 일 입니다

```python
print("날짜는 "+ date[:4]+"년",
      date[4:6],"월", date[6:],"일 입니다")
```

    날짜는 2001년 03 월 31 일 입니다

## ( 실습\_06. 리스트 인덱싱과 슬라이싱 )

    [1,2,['a','b',['Life','is']]] 값을 갖는
    리스트 a를 생성하시오.
    a에서'Life' 값을 추출 하시오.

```python
a = [1,2,["a","b",["Life","is"]]];a
```

    [1, 2, ['a', 'b', ['Life', 'is']]]

```python
len(a[2])
```

    3

```python
a[2][2][0]
```

    'Life'

```python

```
