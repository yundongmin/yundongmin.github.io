---
layout: single
title: "2202.11.02.파이썬 기초 데이터처리 기술(1)"
categories: python_basic
tag: [python, jupyter]
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

## 코드

```python
# 출처 -> 02.Python_초급.pdf
```

```python
temp = 2
```

```python
temp
```

    2

```python
print(temp)
```

    2

### 변수 이름 부여 규칙

1. 대/소문자(한글포함), 숫자, \_ 로 구성 (O)
2. 숫자가 제일앞에 나올수 없음 (X)
3. 공백을 포함해서는 안됨 (X)
4. "예약어"를 사용할 수 없다.

```python
변수 = "A"
```

```python
변수
```

    'A'

```python
# 이런식으로 규칙을 맞추지 않으면 에러가 난다
1aa = 1
```

      File "C:\Users\ehdal\AppData\Local\Temp\ipykernel_85100\273852455.py", line 1
        1aa = 1
         ^
    SyntaxError: invalid syntax

```python
a bcc=2
```

      File "C:\Users\ehdal\AppData\Local\Temp\ipykernel_85100\1200223300.py", line 1
        a bcc=2
          ^
    SyntaxError: invalid syntax

```python
or = 123
```

      File "C:\Users\ehdal\AppData\Local\Temp\ipykernel_85100\298450986.py", line 1
        or = 123
        ^
    SyntaxError: invalid syntax

```python
#이런것들은 변수명으로 쓰면 안된다
import keyword
print(keyword.kwlist)
```

    ['False', 'None', 'True', '__peg_parser__', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']

### ( 숫자형 변수 )

```python
# 사칙연산
a = 7
b = 3
print(a+b)
print(a-b)
print(a*b)
print(a/b)
```

    10
    4
    21
    2.3333333333333335

```python
a = 3;b=7
print("a+b=", a+b)
print("a-b={}".format(a+b))
print("a*b= {}*{}={}".format(a,b,a*b))
print("a/b= %d/%d=%f"%(a,b,a/b))
```

    a+b= 10
    a-b=10
    a*b= 3*7=21
    a/b= 3/7=0.428571

```python
a=7 ; b=3
print("몫=", a//b)
print("나머지=",a%b)
```

    몫= 2
    나머지= 1

```python
a**b
```

    343

```python
a*a*a
```

    343

```python
type(a)
```

    int

```python
dir(a)
```

    ['__abs__',
     '__add__',
     '__and__',
     '__bool__',
     '__ceil__',
     '__class__',
     '__delattr__',
     '__dir__',
     '__divmod__',
     '__doc__',
     '__eq__',
     '__float__',
     '__floor__',
     '__floordiv__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__index__',
     '__init__',
     '__init_subclass__',
     '__int__',
     '__invert__',
     '__le__',
     '__lshift__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__neg__',
     '__new__',
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
     '__rlshift__',
     '__rmod__',
     '__rmul__',
     '__ror__',
     '__round__',
     '__rpow__',
     '__rrshift__',
     '__rshift__',
     '__rsub__',
     '__rtruediv__',
     '__rxor__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__sub__',
     '__subclasshook__',
     '__truediv__',
     '__trunc__',
     '__xor__',
     'as_integer_ratio',
     'bit_length',
     'conjugate',
     'denominator',
     'from_bytes',
     'imag',
     'numerator',
     'real',
     'to_bytes']

```python
a.__sizeof__()
```

    28

### ( 문자열 변수 )

```python
#문자열 변수에 값 할당 합니다.
temp = "python is easy"; temp
```

    'python is easy'

```python
print(temp)
```

    python is easy

```python
type(temp)
```

    str

```python
len(temp)
```

    14

```python
temp
```

    'python is easy'

```python
temp[3]
```

    'h'

```python
# slice[st:en]
temp[0:5]
```

    'pytho'

```python
temp[0:5:2]
```

    'pto'

```python
temp[-4:-2]
```

    'ea'

```python
temp
```

    'python is easy'

```python
temp.split(" ")
```

    ['python', 'is', 'easy']

```python
aa = "localhost:8888/notebooks/0.jswoo/221102_python_기초.ipynb"
```

```python
aa.split("/")
```

    ['localhost:8888', 'notebooks', '0.jswoo', '221102_python_기초.ipynb']

```python
aa.split("/")[-1]
```

    '221102_python_기초.ipynb'

```python
bb = aa.split("/")
```

```python
"#".join(bb)
```

    'localhost:8888#notebooks#0.jswoo#221102_python_기초.ipynb'

```python
a1 = 1
a2 = 2
a3 = 3
```

### ( 데이터 구조 )

1. 리스트
2. 튜플
3. 딕셔너리

```python
# 리스트 생성
number = ["one","two","three","four"];number
```

    ['one', 'two', 'three', 'four']

```python
type(number)
```

    list

```python
num_tup = ("1","2","3","4"); num_tup
```

    ('1', '2', '3', '4')

```python
list(num_tup)
```

    ['1', '2', '3', '4']

```python
len(number)
```

    4

```python
number[5-2]
```

    'four'

```python
number[0] = 111 ; number
```

    [111, 'two', 'three', 'four']

```python
number[2:4]
```

    ['three', 'four']

```python
# 리스트에 값 추가
number.append("five");number
```

    [111, 'two', 'three', 'four', 'five']

```python
# 리스트에 요소 삭제
number.remove("three");number
```

    [111, 'two', 'four', 'five']

```python
aa = 123; aa
```

    123

```python
del aa
```

```python
num1 = ["1","2","3","4","5"]
num2 = ["6","7","8","9"]
```

```python
print("num1 = ", num1)
print('num2 =', num2)
```

    num1 =  ['1', '2', '3', '4', '5']
    num2 = ['6', '7', '8', '9']

```python
num1.extend(num2)
```

```python
num1
```

    ['1', '2', '3', '4', '5', '6', '7', '8', '9']

# 리스트 순서

num3 = sorted(number)

```python
number[0] = "one"
```

```python
sorted(number)
```

    ['five', 'four', 'one', 'two']

```python
number[-1] = "three"
```

```python
sorted(number)
```

    ['four', 'one', 'three', 'two']

```python
number
```

    ['one', 'two', 'four', 'three']

```python
num3 = sorted(number);num3
```

    ['four', 'one', 'three', 'two']

```python
ab = [1,2,["a","b","c"],[3,4]];ab
```

    [1, 2, ['a', 'b', 'c'], [3, 4]]

## ( 튜플 )

1. 생성-> 변수 = (1,2,3,4)

```python
tup = (1,2,3,4);tup
```

    (1, 2, 3, 4)

```python
type(tup)
```

    tuple

```python
tup[0] = 9
```

    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_85100\3580733124.py in <module>
    ----> 1 tup[0] = 9


    TypeError: 'tuple' object does not support item assignment

```python
tup_list = list(tup)
```

```python
tup_list[0] = 9
tup = tuple(tup_list)
tup
```

    (9, 2, 3, 4)

## ( 딕셔너리 )

1. 생성 변수 = {key1:val1, key2:val2}

```python
fp = {"ap":3500, "pe":2500, "ch":5000}
fp
```

    {'ap': 3500, 'pe': 2500, 'ch': 5000}

```python
type(fp)
```

    dict

```python
fp.keys()
```

    dict_keys(['ap', 'pe', 'ch'])

```python
fp.values()
```

    dict_values([3500, 2500, 5000])

```python
fp.items()
```

    dict_items([('ap', 3500), ('pe', 2500), ('ch', 5000)])

```python
fp1 = {"a":[1,2,3], "b":[4,5,6], "c":[7,8,9,0]}
fp1
```

    {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9, 0]}

```python
fp["ap"]
```

    3500

```python
fp.get("ap")
```

    3500

```python

```

## 이미지 목차

### 이미지 세부 목차1

이미지입니다.

### 이미지 세부 목차2

이미지입니다.{: .notice}

### 이미지 세부 목차3

이미지입니다.

![샘플 이미지 입니다.](https://images.unsplash.com/photo-1579353977828-2a4eab540b9a?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8c2FtcGxlfGVufDB8fDB8fA%3D%3D&ixlib=rb-1.2.1&w=1000&q=80)

# 와우2

안녕하세요?
