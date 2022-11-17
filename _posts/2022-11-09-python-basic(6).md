---
layout: single
title: "2202.11.09.파이썬 기초 데이터처리 기술(6)"
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

1. 데이터수집*네이버*뉴스속보\_함수만들기
2. 데이터수집 bs4

#### 1. 데이터수집*네이버*뉴스속보\_함수만들기

### ( 네이버 뉴스기사 수집 )

1. url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"
2. 웹 수집

   - 1단계) requests로 url정보 수집
   - 2단계) bs로 html parser 적옹
   - 3단계) 이미지 링크정보 수집
   - 4단계) 기사링크, 제목 정보 수집
   - 5단계) 기사내용 정보 수집
   - 6단계) 통합 모듈로 정리

3. 이미지 링크정보 수집

```python
import requests
from bs4 import BeautifulSoup
headers = { "User-Agent":"Mozilla/5.0"}
url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"

# 해당 url requests.get
html = requests.get(url, headers=headers)

soup = BeautifulSoup(html.text)  # "html.parser"
type06 = soup.find("ul",{"class":"type06_headline"})
dl = type06.find_all("dl")

# 기사 이미지 수집
for item2 in dl:
    try:
        img = item2.find("dt",{"class":"photo"}).find("img")
        print("img = ", img["src"])
        print("-"*30)
    except:
        print("no image")
```

    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583723.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583722.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583721.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/009/2022/11/17/5046949.jpg?type=nf106_72
    ------------------------------
    no image
    no image
    img =  https://imgnews.pstatic.net/image/origin/009/2022/11/17/5046946.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/009/2022/11/17/5046944.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/009/2022/11/17/5046943.jpg?type=nf106_72
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/009/2022/11/17/5046942.jpg?type=nf106_72
    ------------------------------

2. 기사 링크정보, 제목정보 수집

```python
# 기사 링크주소 & 제목 수집
dl = type06.find_all("dl")
for item2 in dl:
    link = item2.find("dt", {"class" : ""}).find("a")
    print("링크주소 = ", link["href"])
    print("제목 = ", link.text.replace("\t", "").replace("\n", "")[1:len(link.text)+1])
```

    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583723?sid=001
    제목 =  올해 1세대 1주택자 22만명에 종부세 2천400억원 고지할듯
    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583722?sid=001
    제목 =  코스피, 외인·기관 매도에 2,440대로 하락…환율 14.1원 올라(종합)
    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583721?sid=001
    제목 =  이재용·최태원·정의선 등 총출동…빈 살만과 환담 시작
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046949?sid=001
    제목 =  마라도나 '신의 손' 축구공 런던 경매서 31억원에 팔려
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046948?sid=001
    제목 =  부 음
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046947?sid=001
    제목 =  인 사
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046946?sid=001
    제목 =  이재성 LG전자 부사장 은탑훈장
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046944?sid=001
    제목 =  [매경춘추] 견디는 힘, 여행!
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046943?sid=001
    제목 =  박재인 필립스코리아 대표
    링크주소 =  https://n.news.naver.com/mnews/article/009/0005046942?sid=001
    제목 =  이디야커피 '메이트' 200명에 장학금

3. 기사 내용 정보 수집

```python
# 기사 내용 수집

dl = type06.find_all("dl")
for item2 in dl:
    try:
        content = item2.find("dd")
        print("내용 = ", content.text.replace("\t","").replace("\n","").split("…")[0])
    except:
        print("No Content")
```

    내용 =  1세대 1주택 과세 대상 2017년 3.6만명→22만명, 6배 이상 증가 올해 1세대 1주택자 22만명은 2천400억원 상당의
    내용 =  코스피가 17일 외국인과 기관의 매도세에 2,440대로 주저앉았다. 이날 코스피는 전 거래일보다 34.55포인트(1.39%) 내
    내용 =  김동관·박정원·이재현·정기선·이해욱도 합류
    내용 =  1986년 멕시코월드컵에서 디에고 마라도나(아르헨티나)가 '신의 손'이 함께해 골을 넣었다고 언급했던 경기에 사용됐던 축구공이
    내용 =  ▲왕윤수씨 별세, 왕종명(MBC 워싱턴 지국장) 왕종미씨(대구 전자공고 교사) 부친상, 이성용씨(애플하우스 인테리어 대표) 장인
    내용 =  ■한국건설기술연구원 ◇보직인사 △건설인증센터장 강성훈 △기술사업화실장 김중배 ■리버티코리아포스트 △편집국 경제부 팀장 임정혁매일경제신문A33면 1분전
    내용 =  이재성 LG전자 에어솔루션사업부장(부사장·사진)이 국내 가전 산업을 발전시킨 공로를 인정받아 은탑산업훈장을 받았다. 이 부사장은
    내용 =  검사로 재직할 때 여행은 큰마음 먹고 시작해야 하는 이벤트였다. 짧은 여행이라도 가려면 진행 중인 사건에 지장이 없는지 꼼꼼히
    내용 =  필립스코리아가 17일 박재인 신임 대표이사(사장·사진)를 선임했다고 밝혔다. 박 사장은 앞서 글로벌 의료기기 기업 애보트의 국내
    내용 =  매일경제신문A33면 1분전

4. 통합

- 이미지링크정보, 기사링크정보, 기사제목, 기사내용 정보
- 한번에 수집

```python
import requests
from bs4 import BeautifulSoup
headers = { "User-Agent":"Mozilla/5.0"}
url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"

# 해당 url requests.get
html = requests.get(url, headers=headers)

soup = BeautifulSoup(html.text)  # "html.parser"
type06 = soup.find("ul",{"class":"type06_headline"})
dl = type06.find_all("dl")

# 기사 이미지 수집
for item2 in dl:
    try:
        img = item2.find("dt",{"class":"photo"}).find("img")
        print("img = ", img["src"])
        link = item2.find("dt", {"class" : ""}).find("a")
        print("링크주소 = ", link["href"])
        print("제목 = ", link.text.replace("\t", "").replace("\n", "")[1:len(link.text)+1])
        content = item2.find("dd")
        print("내용 = ", content.text.replace("\t","").replace("\n","").split("…")[0])
        print("-"*30)
    except:
        print("no image")
```

    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583726.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583726?sid=001
    제목 =  [영상] 세 번째 '코로나 수능'…"예년 출제기조 유지"
    내용 =  2023학년도 대학수학능력시험(수능) 출제위원장인 박윤봉 충남대 교수는 올해 수능에서 예년 출제기조를 유지했으며 선택과목에 따른
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583725.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583725?sid=001
    제목 =  에쓰오일 9조 '샤힌 프로젝트' 첫발…세계 최대 스팀크래커 구축(종합)
    내용 =  울산 2단계 석유화학 프로젝트
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/001/2022/11/17/13583724.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/001/0013583724?sid=001
    제목 =  (3rd LD) S Korea-Saudi Arabia-biz talks
    내용 =  Biz leaders of S. Korea, Saudi Arabia discuss future cooperation SEOUL,
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/032/2022/11/17/3186953.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/032/0003186953?sid=001
    제목 =  수교 30주년 맞아 한·중 서화작품 한 자리에…베이징서 국제순회전 ‘화운한풍’ 개최
    내용 =  한·중 수교 30주년을 기념해 양국 서화 작가들의 작품을 한 자리에서 감상할 수 있는 전시회가 중국 베이징에서 열린다. 베이징시
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/021/2022/11/17/2541949.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/021/0002541949?sid=001
    제목 =  사우디 왕세자가 방한 기간 눕는 이 침대…홍보 효과 톡톡
    내용 =  빈 살만 왕세자, 시몬스 최고가 라인 ‘뷰티레스트 블랙’ 이용 1900만∼3500만 원 호가
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/005/2022/11/17/1567373.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/005/0001567373?sid=001
    제목 =  고명진 목사도 국회로…차금법 반대 1인 시위 이어가
    내용 =  포괄적 차별금지법에 반대하는 대형교회 목회자들의 1인 시위가 잇따르고 있다. 이재훈 온누리교회 목사와 이찬수 분당우리교회 목사
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/087/2022/11/17/935579.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/087/0000935579?sid=001
    제목 =  [포토뉴스]국제위러브유운동본부 전기요 및 마스크 전달
    내용 =  국제위러브유운동본부 춘천지부(지부장:최승원)는 17일 시청 1층 로비에서 취약가구를 위한 전기요 50세트 및 마스크 700매를
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/092/2022/11/17/2274010.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/092/0002274010?sid=001
    제목 =  3.7㎓ 대역 주파수 분배 놓고 통신사들 신경전
    내용 =  3.7㎓ 대역 주파수 할당을 두고 통신 3사가 갈등을 빚고 있다. SK텔레콤은 3.7㎓ 대역 추가 할당 요청에 LG유플러스는 반
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/119/2022/11/17/2658948.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/119/0002658948?sid=001
    제목 =  [시황] 코스피, 美 증시 하락에 외인 매도세로 2450선 내줘
    내용 =  코스피지수가 미국 증시 하락과 외국인 순매도 영향에 1%대 하락율을 보이며 2450선을 내줬다. 코스닥지수도 740선을 하회했고
    ------------------------------
    img =  https://imgnews.pstatic.net/image/origin/021/2022/11/17/2541948.jpg?type=nf106_72
    링크주소 =  https://n.news.naver.com/mnews/article/021/0002541948?sid=001
    제목 =  필립스코리아, 박재인 신임 대표 선임…“헬스케어 전문가”
    내용 =  필립스코리아는 박재인 신임 대표를 선임했다고 17일 밝혔다. 박 신임 사장은 필립스코리아의 대표 직무와 전문 헬스케어 솔루션을
    ------------------------------

5. 수집 데이터로 데이터 프레임 만들기
   - "이미지링크","기사링크","기사제목","기사내용"
   - 기사 리스트 데이터를 만들어
   - 기사별 데이터를 추가
   - 최종 리스트 데이터로 데이터 프레임 만들기

```python
## 네이버 뉴스 속보를 수집
import requests
from bs4 import BeautifulSoup
headers = { "User-Agent":"Mozilla/5.0"}
url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"

# 해당 url requests.get
html = requests.get(url, headers=headers).text
soup = BeautifulSoup(html)
type06 = soup.find("ul",{"class":"type06_headline"})

dl = type06.find_all("dl")

news_page = []

for item2 in dl:
    try:
        # img_src 방법 1
        img = item2.find("dt",{"class":"photo"}).find("img")
        img_src = img["src"]

        # img_src 방법 2
        img_src = item2.find("dt",{"class":"photo"}).find("img")["src"]

        # 기사 주소 & 타이틀
        link = item2.find("dt", {"class" : ""}).find("a")
        link_href = link["href"]
        link_text = link.text.replace("\t","").replace("\r\n","").lstrip()

        # 내용
        content_text = item2.find("dd").text.replace("\n","").split("…")[0]

        news_page.append([img_src, link_href, link_text, content_text])
    except:
        print(" No image or Content ")

# 데이터 프레임 만들기
import pandas as pd
news_df = pd.DataFrame(news_page,
                      columns = ["이미지링크","기사링크","기사제목","기사내용"])
news_df.head(2)
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
      <th>이미지링크</th>
      <th>기사링크</th>
      <th>기사제목</th>
      <th>기사내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>[영상] 세 번째 '코로나 수능'…"예년 출제기조 유지"\n</td>
      <td>2023학년도 대학수학능력시험(수능) 출제위원장인 박윤봉 충남대 교수는 올해 수능에...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>에쓰오일 9조 '샤힌 프로젝트' 첫발…세계 최대 스팀크래커 구축(종합)\n</td>
      <td>울산 2단계 석유화학 프로젝트</td>
    </tr>
  </tbody>
</table>
</div>

6. 함수 만들어서 처리하기

```python
import requests
from bs4 import BeautifulSoup
headers = { "User-Agent":"Mozilla/5.0"}

url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"

def get_page(url):
    # 해당 url requests.get
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html)
    type06 = soup.find("ul",{"class":"type06_headline"})
    dl = type06.find_all("dl")
    return dl

def page_to_items(dl):
    news_page = []
    for item2 in dl:
        try:
            # img_src 방법 1
            img = item2.find("dt",{"class":"photo"}).find("img")
            img_src = img["src"]

            # img_src 방법 2
            img_src = item2.find("dt",{"class":"photo"}).find("img")["src"]

            # 기사 주소 & 타이틀
            link = item2.find("dt", {"class" : ""}).find("a")
            link_href = link["href"]
            link_text = link.text.replace("\t","").replace("\r\n","").lstrip()
            # 내용
            content_text = item2.find("dd").text.replace("\n","").split("…")[0]
            news_page.append([img_src, link_href, link_text, content_text])
        except:
            print(" No image or Content ")
    return news_page

get_dl = get_page(url)
news_page = page_to_items(get_dl)

# 데이터 프레임 만들기
import pandas as pd
news_df = pd.DataFrame(news_page,
                      columns = ["이미지링크","기사링크","기사제목","기사내용"])
news_df.head(3)
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
      <th>이미지링크</th>
      <th>기사링크</th>
      <th>기사제목</th>
      <th>기사내용</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>[영상] 세 번째 '코로나 수능'…"예년 출제기조 유지"\n</td>
      <td>2023학년도 대학수학능력시험(수능) 출제위원장인 박윤봉 충남대 교수는 올해 수능에...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>에쓰오일 9조 '샤힌 프로젝트' 첫발…세계 최대 스팀크래커 구축(종합)\n</td>
      <td>울산 2단계 석유화학 프로젝트</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>(3rd LD) S Korea-Saudi Arabia-biz talks\n</td>
      <td>Biz leaders of S. Korea, Saudi Arabia discuss ...</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

#### 2. 데이터수집 bs4

```python
#  https://www.crummy.com/software/BeautifulSoup/bs4/doc/
```

```python
html_doc = """<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""
```

```python
type(html_doc)
```

    str

```python
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')
```

```python
type(soup)
```

    bs4.BeautifulSoup

```python
dir(soup)
```

    ['ASCII_SPACES',
     'DEFAULT_BUILDER_FEATURES',
     'DEFAULT_INTERESTING_STRING_TYPES',
     'NO_PARSER_SPECIFIED_WARNING',
     'ROOT_TAG_NAME',
     '__bool__',
     '__call__',
     '__class__',
     '__contains__',
     '__copy__',
     '__delattr__',
     '__delitem__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattr__',
     '__getattribute__',
     '__getitem__',
     '__getstate__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__setitem__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__unicode__',
     '__weakref__',
     '_all_strings',
     '_decode_markup',
     '_feed',
     '_find_all',
     '_find_one',
     '_is_xml',
     '_lastRecursiveChild',
     '_last_descendant',
     '_linkage_fixer',
     '_markup_is_url',
     '_markup_resembles_filename',
     '_most_recent_element',
     '_namespaces',
     '_popToTag',
     '_should_pretty_print',
     'append',
     'attrs',
     'builder',
     'can_be_empty_element',
     'cdata_list_attributes',
     'childGenerator',
     'children',
     'clear',
     'contains_replacement_characters',
     'contents',
     'currentTag',
     'current_data',
     'declared_html_encoding',
     'decode',
     'decode_contents',
     'decompose',
     'decomposed',
     'default',
     'descendants',
     'element_classes',
     'encode',
     'encode_contents',
     'endData',
     'extend',
     'extract',
     'fetchNextSiblings',
     'fetchParents',
     'fetchPrevious',
     'fetchPreviousSiblings',
     'find',
     'findAll',
     'findAllNext',
     'findAllPrevious',
     'findChild',
     'findChildren',
     'findNext',
     'findNextSibling',
     'findNextSiblings',
     'findParent',
     'findParents',
     'findPrevious',
     'findPreviousSibling',
     'findPreviousSiblings',
     'find_all',
     'find_all_next',
     'find_all_previous',
     'find_next',
     'find_next_sibling',
     'find_next_siblings',
     'find_parent',
     'find_parents',
     'find_previous',
     'find_previous_sibling',
     'find_previous_siblings',
     'format_string',
     'formatter_for_name',
     'get',
     'getText',
     'get_attribute_list',
     'get_text',
     'handle_data',
     'handle_endtag',
     'handle_starttag',
     'has_attr',
     'has_key',
     'hidden',
     'index',
     'insert',
     'insert_after',
     'insert_before',
     'interesting_string_types',
     'isSelfClosing',
     'is_empty_element',
     'is_xml',
     'known_xml',
     'markup',
     'name',
     'namespace',
     'new_string',
     'new_tag',
     'next',
     'nextGenerator',
     'nextSibling',
     'nextSiblingGenerator',
     'next_element',
     'next_elements',
     'next_sibling',
     'next_siblings',
     'object_was_parsed',
     'open_tag_counter',
     'original_encoding',
     'parent',
     'parentGenerator',
     'parents',
     'parse_only',
     'parserClass',
     'parser_class',
     'popTag',
     'prefix',
     'preserve_whitespace_tag_stack',
     'preserve_whitespace_tags',
     'prettify',
     'previous',
     'previousGenerator',
     'previousSibling',
     'previousSiblingGenerator',
     'previous_element',
     'previous_elements',
     'previous_sibling',
     'previous_siblings',
     'pushTag',
     'recursiveChildGenerator',
     'renderContents',
     'replaceWith',
     'replaceWithChildren',
     'replace_with',
     'replace_with_children',
     'reset',
     'select',
     'select_one',
     'setup',
     'smooth',
     'string',
     'string_container',
     'string_container_stack',
     'strings',
     'stripped_strings',
     'tagStack',
     'text',
     'unwrap',
     'wrap']

```python
print(soup)
```

    <html><head><title>The Dormouse's story</title></head>
    <body>
    <p class="title"><b>The Dormouse's story</b></p>
    <p class="story">Once upon a time there were three little sisters; and their names were
    <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a> and
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>;
    and they lived at the bottom of a well.</p>
    <p class="story">...</p>
    </body></html>

```python
print(soup.prettify())
```

    <html>
     <head>
      <title>
       The Dormouse's story
      </title>
     </head>
     <body>
      <p class="title">
       <b>
        The Dormouse's story
       </b>
      </p>
      <p class="story">
       Once upon a time there were three little sisters; and their names were
       <a class="sister" href="http://example.com/elsie" id="link1">
        Elsie
       </a>
       ,
       <a class="sister" href="http://example.com/lacie" id="link2">
        Lacie
       </a>
       and
       <a class="sister" href="http://example.com/tillie" id="link3">
        Tillie
       </a>
       ;
    and they lived at the bottom of a well.
      </p>
      <p class="story">
       ...
      </p>
     </body>
    </html>

```python
soup.title
```

    <title>The Dormouse's story</title>

```python
soup.title.name
```

    'title'

```python
soup.title.string
```

    "The Dormouse's story"

```python
soup.title.parent.name
```

    'head'

```python
soup.p
```

    <p class="title"><b>The Dormouse's story</b></p>

```python
soup.p['class']
```

    ['title']

```python
soup.a
```

    <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

```python
soup.a['class']
```

    ['sister']

```python
soup.find_all('a')
```

    [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

```python
res = soup.find_all('a')
res
```

    [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>,
     <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>,
     <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]

```python
for item in res:
    print(item)
```

    <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>
    <a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>
    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

```python
soup.find(id="link3")
```

    <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>

```python
soup.find(class_="sister")
```

    <a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>

```python
for link in soup.find_all('a'):
    print(link.get('href'))
```

    http://example.com/elsie
    http://example.com/lacie
    http://example.com/tillie

```python
for link in soup.find_all('a'):
    print(link['href'])
    print(link.text)
```

    http://example.com/elsie
    Elsie
    http://example.com/lacie
    Lacie
    http://example.com/tillie
    Tillie

```python
print(soup.get_text())
```

    The Dormouse's story

    The Dormouse's story
    Once upon a time there were three little sisters; and their names were
    Elsie,
    Lacie and
    Tillie;
    and they lived at the bottom of a well.
    ...

```python
soup.html.head
```

    <head><title>The Dormouse's story</title></head>

```python
soup.html.body.p
```

    <p class="title"><b>The Dormouse's story</b></p>

```python

```
