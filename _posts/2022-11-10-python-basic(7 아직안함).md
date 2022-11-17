---
layout: single
title: "2202.11.10.파이썬 기초 데이터처리 기술(7)"
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

1. 데이터수집 네이버뉴스 여러날짜적용 및 DB 등록
2. 데이터수집 selenium
3. python db연동

#### 1. 데이터수집 네이버뉴스 여러날짜적용 및 DB 등록

### ( 실습. 네이버 뉴스 수집 여러 페이지 )

1. 날짜는 22.11.08 ~ 22.11.10
   - 날짜별 페이지는 3page (1,2,3)
   - df 컬럼 추가
     - 날짜 컬럼, page 컬럼
2. 통합 df로 만들기

```python
import pandas as pd

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
```

```python
url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"
news_date = pd.date_range("2022-11-08", periods=3)

init = 0

for wk_date in news_date:
    for wk_page in (range(0,3)):
        str_date = wk_date.strftime("%Y%m%d")
        target_url = url + "&date=" + str_date + "&page=" + str(wk_page)
        print(target_url)

        get_dl = get_page(url)
        news_page = page_to_items(get_dl)

        news_page_df = pd.DataFrame(news_page,
                              columns = ["이미지링크","기사링크","기사제목",
                                         "기사내용"])
        news_page_df["날짜"] = wk_date
        news_page_df["page"] = wk_page

        if init == 0:
            total_df = news_page_df.copy()
            init = 1
        else:
            total_df = pd.concat([total_df, news_page_df], ignore_index=True)

        print("total_df = ", total_df.shape)

# 컬럼순서 바꾸기
total_df = total_df[["날짜","page","이미지링크","기사링크","기사제목","기사내용"]]
# df to csv 만들기
total_df.to_csv("c:/test/naver_news_page.csv",
                index=False, encoding="utf-8-sig") #cp949, EUC-KR, utf-8, utf-8-sig

print("total_df = ", total_df.shape)
```

    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221108&page=0
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (4, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221108&page=1
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (8, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221108&page=2
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (12, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221109&page=0
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (16, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221109&page=1
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (20, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221109&page=2
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (24, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221110&page=0
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (28, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221110&page=1
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (32, 6)
    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221110&page=2
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
    total_df =  (36, 6)
    total_df =  (36, 6)

```python
url = "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001"
news_date = pd.date_range("2022-11-08", periods=3)

total_df = pd.DataFrame([],
                      columns = ["이미지링크","기사링크","기사제목",
                                 "기사내용", "날짜","page"])
for wk_date in news_date:
    for wk_page in (range(0,3)):
        str_date = wk_date.strftime("%Y%m%d")
        target_url = url + "&date=" + str_date + "&page=" + str(wk_page)
        print(target_url)

        get_dl = get_page(url)
        news_page = page_to_items(get_dl)

        news_page_df = pd.DataFrame(news_page,
                              columns = ["이미지링크","기사링크","기사제목",
                                         "기사내용"])
        news_page_df["날짜"] = wk_date
        news_page_df["page"] = wk_page

        total_df = pd.concat([total_df, news_page_df], ignore_index=True)

        total_df.to_csv("c:/jswoo/naver_news_page.csv",
                         index=False, encoding="utf-8-sig") #cp949, EUC-KR, utf-8, utf-8-sig

        print("new_page = ", total_df.shape)

# 컬럼순서 바꾸기
total_df = total_df[["날짜","page","이미지링크","기사링크","기사제목","기사내용"]]
# df to csv 만들기
total_df.to_csv("c:/test/naver_news_page.csv",
                index=False, encoding="utf-8-sig") #cp949, EUC-KR, utf-8, utf-8-sig

print("total_df = ", total_df.shape)
```

    https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20221108&page=0
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content
     No image or Content



    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_55856\818229747.py in <module>
         22         total_df = pd.concat([total_df, news_page_df], ignore_index=True)
         23
    ---> 24         total_df.to_csv("c:/jswoo/naver_news_page.csv",
         25                          index=False, encoding="utf-8-sig") #cp949, EUC-KR, utf-8, utf-8-sig
         26


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in to_csv(self, path_or_buf, sep, na_rep, float_format, columns, header, index, index_label, mode, encoding, compression, quoting, quotechar, line_terminator, chunksize, date_format, doublequote, escapechar, decimal, errors, storage_options)
       3549         )
       3550
    -> 3551         return DataFrameRenderer(formatter).to_csv(
       3552             path_or_buf,
       3553             line_terminator=line_terminator,


    ~\anaconda3\lib\site-packages\pandas\io\formats\format.py in to_csv(self, path_or_buf, encoding, sep, columns, index_label, mode, compression, quoting, quotechar, line_terminator, chunksize, date_format, doublequote, escapechar, errors, storage_options)
       1178             formatter=self.fmt,
       1179         )
    -> 1180         csv_formatter.save()
       1181
       1182         if created_buffer:


    ~\anaconda3\lib\site-packages\pandas\io\formats\csvs.py in save(self)
        239         """
        240         # apply compression and byte/text conversion
    --> 241         with get_handle(
        242             self.filepath_or_buffer,
        243             self.mode,


    ~\anaconda3\lib\site-packages\pandas\io\common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        692     # Only for write methods
        693     if "r" not in mode and is_path:
    --> 694         check_parent_directory(str(handle))
        695
        696     if compression:


    ~\anaconda3\lib\site-packages\pandas\io\common.py in check_parent_directory(path)
        566     parent = Path(path).parent
        567     if not parent.is_dir():
    --> 568         raise OSError(rf"Cannot save file into a non-existent directory: '{parent}'")
        569
        570


    OSError: Cannot save file into a non-existent directory: 'c:\jswoo'

```python
total_df.head(2)
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
      <th>날짜</th>
      <th>page</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://imgnews.pstatic.net/image/origin/001/2...</td>
      <td>https://n.news.naver.com/mnews/article/001/001...</td>
      <td>[의회소식] 경남 아동급식 지원 조례안, 상임위 통과\n</td>
      <td>경남도의회 문화복지위원회는 제400회 정례회 기간인 17일 상임위원회를 열어 국민의...</td>
      <td>2022-11-08</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://imgnews.pstatic.net/image/origin/009/2...</td>
      <td>https://n.news.naver.com/mnews/article/009/000...</td>
      <td>국어 '기초대사량' 지문 진땀 … 수학 미적분 과목 까다로워\n</td>
      <td>수능 난이도 분석 국어 난도 작년보다 떨어져 '언어와 매체' 체감난도 높을듯 수학 ...</td>
      <td>2022-11-08</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
total_df["이미지링크"].str.len().max()
```

    82

### DB에 넣는다.

1. DBMS명 -> newsdb
2. TABLE명 -> naver_news
   - news_id, news_date, news_page, news_img_link, news_link, news_title, news_content
   - primary key -> news_id
3. 실습
   - Table을 생성한다
   - total_df의 컬럼명을 table의 컬럼명과 일치 시킨다.
   - df.to_sql() 실행한다.

```python
total_df.columns
```

    Index(['이미지링크', '기사링크', '기사제목', '기사내용', '날짜', 'page'], dtype='object')

1. table 생성

```python
import pymysql

db = pymysql.connect(host="localhost", port=3306,
                     user="root", passwd = "pass",
                     db = "newsdb", charset="utf8")

cursor = db.cursor()

sql_str = """
    CREATE TABLE IF NOT EXISTS naver_news (
           news_id         int(2),
           news_date       date,
           news_page       int(2),
           news_img_link   varchar(100),
           news_link       varchar(100),
           news_title      varchar(100),
           news_content    varchar(100),
           primary key(news_id),
           index secondary (news_date, news_page)
    );
"""

cursor.execute(sql_str)

db.commit()
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_55856\3283575533.py in <module>
          1 import pymysql
          2
    ----> 3 db = pymysql.connect(host="localhost", port=3306,
          4                      user="root", passwd = "pass",
          5                      db = "newsdb", charset="utf8")


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        351             self._sock = None
        352         else:
    --> 353             self.connect()
        354
        355     def __enter__(self):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        631
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634
        635             if self.sql_mode is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        919                 and plugin_name is not None
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:
        923                 # send legacy handshake


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1016
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()
       1020         return pkt


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")

```python
news_df = total_df.copy()
```

```python
news_df["news_content"].str.len().max()
```

#2. 컬럼명과 순서 맞추기

```python
news_df = news_df.reset_index()
news_df.columns = ["news_id","news_date","news_page","news_img_link",
                    "news_link","news_title","news_content"]
news_df.head(2)
```

3. df.to_sql() 실행

```python
from sqlalchemy import create_engine

# MariaDB Connector using pymysql
pymysql.install_as_MySQLdb()

import MySQLdb

engine = create_engine("mysql://root:pass@127.0.0.1/newsdb",encoding="utf-8")

news_df.to_sql(name="naver_news",con=engine,
               if_exists="append",index=False)
```

```python
### ( 이미지 다운로드 )
```

```python
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
createFolder('c:/jswoo/img_down')
```

```python
import urllib.request
for row in range(0,len(total_df)):
    img_file = total_df.iloc[row,2].split("?")
    img_name = img_file[0].split("/")[-1]
    down_filename = "c:/jswoo/img_down/" + img_name   # ./img_down/132334.jpg

    # 이미지 링크로 사진 다운 받기
    url = img_file[0]
    urllib.request.urlretrieve(url, down_filename)

    print("down = ", url )
print("*"*30)

print(" 완료!!! ")
```

### ( 웹 수집할때 )

1. table 태그로 구성시 ( 반드시는 안님)

```python
url1 = "https://movie.naver.com/movie/point/af/list.naver?&page=1"

read_html = pd.read_html(url1)
df = read_html[0]

df.head(3)
```

```python

```

#### 2. 데이터수집 selenium

```python
# https://selenium-python.readthedocs.io/getting-started.html
```

### ( Selenium 시작 하기 )

1. selenium 설치 -> pip install selenium
2. 브라우저에 맞는 드라이브 다운로드
   - 107ver -> chromedriver.exe
   - c:/test/

```python
from selenium.webdriver import Chrome
driver = Chrome("c:/test/chromedriver.exe")
```

    C:\Users\ehdal\AppData\Local\Temp\ipykernel_56996\3525337058.py:2: DeprecationWarning: executable_path has been deprecated, please pass in a Service object
      driver = Chrome("c:/test/chromedriver.exe")

```python
url = "https://selenium.dev"
driver.get(url)
```

```python
driver.current_url
```

    'https://www.selenium.dev/'

```python
#dir(driver)
```

```python
driver.get("http://www.python.org")
```

```python
from selenium.webdriver.common.by import By
elem = driver.find_element(By.NAME, "q")
```

```python
elem.clear()
```

```python
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_56996\3608860890.py in <module>
          1 elem.send_keys("pycon")
    ----> 2 elem.send_keys(Keys.RETURN)


    NameError: name 'Keys' is not defined

```python
from selenium.webdriver.common.keys import Keys
elem.send_keys(Keys.RETURN)
```

```python
my_id = "jswoo100@empas.com"
my_pw = "akrhr701!@!@"
```

### ( nate login )

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

driver = webdriver.Chrome("c:/test/chromedriver.exe")

url = "https://www.nate.com/"

driver.get(url)
```

    C:\Users\ehdal\AppData\Local\Temp\ipykernel_56996\678002887.py:5: DeprecationWarning: executable_path has been deprecated, please pass in a Service object
      driver = webdriver.Chrome("c:/test/chromedriver.exe")

```python
elem = driver.find_element(By.NAME, "ID")
elem.send_keys(my_id)
```

```python
elem = driver.find_element(By.NAME, "PASSDM")
elem.send_keys(my_pw, Keys.ENTER)
```

```python
elem.clear()
elem = driver.find_element(By.ID, "ID")
elem.send_keys(my_id)
```

    ---------------------------------------------------------------------------

    StaleElementReferenceException            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_56996\2995432120.py in <module>
    ----> 1 elem.clear()
          2 elem = driver.find_element(By.ID, "ID")
          3 elem.send_keys(my_id)


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in clear(self)
        114     def clear(self) -> None:
        115         """Clears the text if it's a text entry element."""
    --> 116         self._execute(Command.CLEAR_ELEMENT)
        117
        118     def get_property(self, name) -> str | bool | WebElement | dict:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in _execute(self, command, params)
        408             params = {}
        409         params["id"] = self._id
    --> 410         return self._parent.execute(command, params)
        411
        412     def find_element(self, by=By.ID, value=None) -> WebElement:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in execute(self, driver_command, params)
        442         response = self.command_executor.execute(driver_command, params)
        443         if response:
    --> 444             self.error_handler.check_response(response)
        445             response["value"] = self._unwrap_value(response.get("value", None))
        446             return response


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\errorhandler.py in check_response(self, response)
        247                 alert_text = value["alert"].get("text")
        248             raise exception_class(message, screen, stacktrace, alert_text)  # type: ignore[call-arg]  # mypy is not smart enough here
    --> 249         raise exception_class(message, screen, stacktrace)


    StaleElementReferenceException: Message: stale element reference: element is not attached to the page document
      (Session info: chrome=107.0.5304.107)
    Stacktrace:
    Backtrace:
    	Ordinal0 [0x0064ACD3+2075859]
    	Ordinal0 [0x005DEE61+1633889]
    	Ordinal0 [0x004DB7BD+571325]
    	Ordinal0 [0x004DE374+582516]
    	Ordinal0 [0x004DE225+582181]
    	Ordinal0 [0x004DE4C0+582848]
    	Ordinal0 [0x0050C654+771668]
    	Ordinal0 [0x00502FDC+733148]
    	Ordinal0 [0x0052731C+881436]
    	Ordinal0 [0x005015BF+726463]
    	Ordinal0 [0x00527534+881972]
    	Ordinal0 [0x0053B56A+963946]
    	Ordinal0 [0x00527136+880950]
    	Ordinal0 [0x004FFEFD+720637]
    	Ordinal0 [0x00500F3F+724799]
    	GetHandleVerifier [0x008FEED2+2769538]
    	GetHandleVerifier [0x008F0D95+2711877]
    	GetHandleVerifier [0x006DA03A+521194]
    	GetHandleVerifier [0x006D8DA0+516432]
    	Ordinal0 [0x005E682C+1665068]
    	Ordinal0 [0x005EB128+1683752]
    	Ordinal0 [0x005EB215+1683989]
    	Ordinal0 [0x005F6484+1729668]
    	BaseThreadInitThunk [0x75BF7BA9+25]
    	RtlInitializeExceptionChain [0x76FEBB9B+107]
    	RtlClearBits [0x76FEBB1F+191]

```python
elem.clear()
elem = driver.find_element(By.XPATH, "//*[@id='ID']")
elem.send_keys(my_id)
```

    ---------------------------------------------------------------------------

    StaleElementReferenceException            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_56996\4056264029.py in <module>
    ----> 1 elem.clear()
          2 elem = driver.find_element(By.XPATH, "//*[@id='ID']")
          3 elem.send_keys(my_id)


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in clear(self)
        114     def clear(self) -> None:
        115         """Clears the text if it's a text entry element."""
    --> 116         self._execute(Command.CLEAR_ELEMENT)
        117
        118     def get_property(self, name) -> str | bool | WebElement | dict:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in _execute(self, command, params)
        408             params = {}
        409         params["id"] = self._id
    --> 410         return self._parent.execute(command, params)
        411
        412     def find_element(self, by=By.ID, value=None) -> WebElement:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in execute(self, driver_command, params)
        442         response = self.command_executor.execute(driver_command, params)
        443         if response:
    --> 444             self.error_handler.check_response(response)
        445             response["value"] = self._unwrap_value(response.get("value", None))
        446             return response


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\errorhandler.py in check_response(self, response)
        247                 alert_text = value["alert"].get("text")
        248             raise exception_class(message, screen, stacktrace, alert_text)  # type: ignore[call-arg]  # mypy is not smart enough here
    --> 249         raise exception_class(message, screen, stacktrace)


    StaleElementReferenceException: Message: stale element reference: element is not attached to the page document
      (Session info: chrome=107.0.5304.107)
    Stacktrace:
    Backtrace:
    	Ordinal0 [0x0064ACD3+2075859]
    	Ordinal0 [0x005DEE61+1633889]
    	Ordinal0 [0x004DB7BD+571325]
    	Ordinal0 [0x004DE374+582516]
    	Ordinal0 [0x004DE225+582181]
    	Ordinal0 [0x004DE4C0+582848]
    	Ordinal0 [0x0050C654+771668]
    	Ordinal0 [0x00502FDC+733148]
    	Ordinal0 [0x0052731C+881436]
    	Ordinal0 [0x005015BF+726463]
    	Ordinal0 [0x00527534+881972]
    	Ordinal0 [0x0053B56A+963946]
    	Ordinal0 [0x00527136+880950]
    	Ordinal0 [0x004FFEFD+720637]
    	Ordinal0 [0x00500F3F+724799]
    	GetHandleVerifier [0x008FEED2+2769538]
    	GetHandleVerifier [0x008F0D95+2711877]
    	GetHandleVerifier [0x006DA03A+521194]
    	GetHandleVerifier [0x006D8DA0+516432]
    	Ordinal0 [0x005E682C+1665068]
    	Ordinal0 [0x005EB128+1683752]
    	Ordinal0 [0x005EB215+1683989]
    	Ordinal0 [0x005F6484+1729668]
    	BaseThreadInitThunk [0x75BF7BA9+25]
    	RtlInitializeExceptionChain [0x76FEBB9B+107]
    	RtlClearBits [0x76FEBB1F+191]

```python
elem.clear()
elem = driver.find_element(By.CSS_SELECTOR, "#ID")
elem.send_keys(my_id)
```

    ---------------------------------------------------------------------------

    StaleElementReferenceException            Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_56996\3921589247.py in <module>
    ----> 1 elem.clear()
          2 elem = driver.find_element(By.CSS_SELECTOR, "#ID")
          3 elem.send_keys(my_id)


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in clear(self)
        114     def clear(self) -> None:
        115         """Clears the text if it's a text entry element."""
    --> 116         self._execute(Command.CLEAR_ELEMENT)
        117
        118     def get_property(self, name) -> str | bool | WebElement | dict:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webelement.py in _execute(self, command, params)
        408             params = {}
        409         params["id"] = self._id
    --> 410         return self._parent.execute(command, params)
        411
        412     def find_element(self, by=By.ID, value=None) -> WebElement:


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\webdriver.py in execute(self, driver_command, params)
        442         response = self.command_executor.execute(driver_command, params)
        443         if response:
    --> 444             self.error_handler.check_response(response)
        445             response["value"] = self._unwrap_value(response.get("value", None))
        446             return response


    ~\anaconda3\lib\site-packages\selenium\webdriver\remote\errorhandler.py in check_response(self, response)
        247                 alert_text = value["alert"].get("text")
        248             raise exception_class(message, screen, stacktrace, alert_text)  # type: ignore[call-arg]  # mypy is not smart enough here
    --> 249         raise exception_class(message, screen, stacktrace)


    StaleElementReferenceException: Message: stale element reference: element is not attached to the page document
      (Session info: chrome=107.0.5304.107)
    Stacktrace:
    Backtrace:
    	Ordinal0 [0x0064ACD3+2075859]
    	Ordinal0 [0x005DEE61+1633889]
    	Ordinal0 [0x004DB7BD+571325]
    	Ordinal0 [0x004DE374+582516]
    	Ordinal0 [0x004DE225+582181]
    	Ordinal0 [0x004DE4C0+582848]
    	Ordinal0 [0x0050C654+771668]
    	Ordinal0 [0x00502FDC+733148]
    	Ordinal0 [0x0052731C+881436]
    	Ordinal0 [0x005015BF+726463]
    	Ordinal0 [0x00527534+881972]
    	Ordinal0 [0x0053B56A+963946]
    	Ordinal0 [0x00527136+880950]
    	Ordinal0 [0x004FFEFD+720637]
    	Ordinal0 [0x00500F3F+724799]
    	GetHandleVerifier [0x008FEED2+2769538]
    	GetHandleVerifier [0x008F0D95+2711877]
    	GetHandleVerifier [0x006DA03A+521194]
    	GetHandleVerifier [0x006D8DA0+516432]
    	Ordinal0 [0x005E682C+1665068]
    	Ordinal0 [0x005EB128+1683752]
    	Ordinal0 [0x005EB215+1683989]
    	Ordinal0 [0x005F6484+1729668]
    	BaseThreadInitThunk [0x75BF7BA9+25]
    	RtlInitializeExceptionChain [0x76FEBB9B+107]
    	RtlClearBits [0x76FEBB1F+191]

```python
options = webdriver.ChromeOptions()
options.add_argument("headless")

dvier = webdriver.Chrome("c:/test/chromedriver.exe", chrome_options = options)
```

    C:\Users\ehdal\AppData\Local\Temp\ipykernel_56996\316414333.py:4: DeprecationWarning: executable_path has been deprecated, please pass in a Service object
      dvier = webdriver.Chrome("c:/test/chromedriver.exe", chrome_options = options)
    C:\Users\ehdal\AppData\Local\Temp\ipykernel_56996\316414333.py:4: DeprecationWarning: use options instead of chrome_options
      dvier = webdriver.Chrome("c:/test/chromedriver.exe", chrome_options = options)

```python
 # jupyter notebook warnings 메시지를 감출때
import warnings
warnings.filterwarnings(action="ignore") # action="default"
```

### ( 실습 )

1. 유튜브 사이트
2. 검색은 "파이썬 머신러닝" 검색후 결과 확인

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

driver = webdriver.Chrome("c:/test/chromedriver.exe")

url = "https://www.youtube.com/"

driver.get(url)


elem = driver.find_element(By.NAME, "search_query")

elem.send_keys("파이썬 머신러닝")
elem.send_keys(Keys.RETURN)
```

```python

```

#### 3. python db연동

### (1. python DB 연동 )

1. mariadb ( mysql )
2. 연결 -> pymysql

```python
import pymysql

db = pymysql.connect(host="localhost", port=3306,
                     user="root", passwd = "pass",
                     db = "newsdb", charset="utf8")

cursor = db.cursor()
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\3999767626.py in <module>
          1 import pymysql
          2
    ----> 3 db = pymysql.connect(host="localhost", port=3306,
          4                      user="root", passwd = "pass",
          5                      db = "newsdb", charset="utf8")


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        351             self._sock = None
        352         else:
    --> 353             self.connect()
        354
        355     def __enter__(self):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        631
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634
        635             if self.sql_mode is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        919                 and plugin_name is not None
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:
        923                 # send legacy handshake


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1016
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()
       1020         return pkt


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")

### ( 2. Table 생성 )

```python
sql_str = """
    CREATE TABLE IF NOT EXISTS test (
        no   int(2),
        name varchar(10),
        eng  int(3),
        math int(3),
        kor  int(3),
        primary key(no)
    )
"""

cursor.execute(sql_str)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\916471637.py in <module>
         10 """
         11
    ---> 12 cursor.execute(sql_str)


    NameError: name 'cursor' is not defined

```python
# data insert
sql_str = """
   INSERT INTO test values(4,'동동사',77,69,89)
"""
print(sql_str)

cursor.execute(sql_str)

db.commit()
```

       INSERT INTO test values(4,'동동사',77,69,89)




    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\1336709508.py in <module>
          5 print(sql_str)
          6
    ----> 7 cursor.execute(sql_str)
          8
          9 db.commit()


    NameError: name 'cursor' is not defined

```python
# 데이터 불러오기
sql_str = """
   SELECT * FROM test
"""
cursor.execute(sql_str)
cursor.fetchall()
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\4007139429.py in <module>
          3    SELECT * FROM test
          4 """
    ----> 5 cursor.execute(sql_str)
          6 cursor.fetchall()


    NameError: name 'cursor' is not defined

```python
# df to sql
```

```python
import pandas as pd

df = pd.DataFrame({"no":[11,12,13],
                   "name":["일일","일이","일삼"],
                   "eng":[61,71,81],
                   "math":[72,83,92],
                   "kor":[73,83,93]}
                   )
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
      <th>no</th>
      <th>name</th>
      <th>eng</th>
      <th>math</th>
      <th>kor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>일일</td>
      <td>61</td>
      <td>72</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>일이</td>
      <td>71</td>
      <td>83</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>일삼</td>
      <td>81</td>
      <td>92</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>

```python
import pymysql

from sqlalchemy import create_engine

pymysql.install_as_MySQLdb()
import MySQLdb

engine = create_engine('mysql://root:pass@127.0.0.1/newsdb', encoding="utf-8")
```

```python
df.to_sql(name="test", con=engine,if_exists="append", index=False)
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        142             try:
    --> 143                 return self._create_connection()
        144             except:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        255
    --> 256         return _ConnectionRecord(self)
        257


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        352         else:
    --> 353             self.connect()
        354


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")


    The above exception was the direct cause of the following exception:


    OperationalError                          Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\3646360395.py in <module>
    ----> 1 df.to_sql(name="test", con=engine,if_exists="append", index=False)


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in to_sql(self, name, con, schema, if_exists, index, index_label, chunksize, dtype, method)
       2949         from pandas.io import sql
       2950
    -> 2951         return sql.to_sql(
       2952             self,
       2953             name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in to_sql(frame, name, con, schema, if_exists, index, index_label, chunksize, dtype, method, engine, **engine_kwargs)
        696         )
        697
    --> 698     return pandas_sql.to_sql(
        699         frame,
        700         name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in to_sql(self, frame, name, if_exists, index, index_label, schema, chunksize, dtype, method, engine, **engine_kwargs)
       1730         sql_engine = get_engine(engine)
       1731
    -> 1732         table = self.prep_table(
       1733             frame=frame,
       1734             name=name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in prep_table(self, frame, name, if_exists, index, index_label, schema, dtype)
       1629             dtype=dtype,
       1630         )
    -> 1631         table.create()
       1632         return table
       1633


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in create(self)
        830
        831     def create(self):
    --> 832         if self.exists():
        833             if self.if_exists == "fail":
        834                 raise ValueError(f"Table '{self.name}' already exists.")


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in exists(self)
        814
        815     def exists(self):
    --> 816         return self.pd_sql.has_table(self.name, self.schema)
        817
        818     def sql_schema(self):


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in has_table(self, name, schema)
       1763             from sqlalchemy import inspect
       1764
    -> 1765             insp = inspect(self.connectable)
       1766             return insp.has_table(name, schema or self.meta.schema)
       1767         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\inspection.py in inspect(subject, raiseerr)
         62             if reg is True:
         63                 return subject
    ---> 64             ret = reg(subject)
         65             if ret is not None:
         66                 break


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _engine_insp(bind)
        180     @inspection._inspects(Engine)
        181     def _engine_insp(bind):
    --> 182         return Inspector._construct(Inspector._init_engine, bind)
        183
        184     @inspection._inspects(Connection)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _construct(cls, init, bind)
        115
        116         self = cls.__new__(cls)
    --> 117         init(self, bind)
        118         return self
        119


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _init_engine(self, engine)
        126     def _init_engine(self, engine):
        127         self.bind = self.engine = engine
    --> 128         engine.connect().close()
        129         self._op_context_requires_connect = True
        130         self.dialect = self.engine.dialect


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in connect(self, close_with_result)
       3232         """
       3233
    -> 3234         return self._connection_cls(self, close_with_result=close_with_result)
       3235
       3236     @util.deprecated(


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in __init__(self, engine, connection, close_with_result, _branch_from, _execution_options, _dispatch, _has_events, _allow_revalidate)
         94                 connection
         95                 if connection is not None
    ---> 96                 else engine.raw_connection()
         97             )
         98


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in raw_connection(self, _connection)
       3311
       3312         """
    -> 3313         return self._wrap_pool_connect(self.pool.connect, _connection)
       3314
       3315


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:
    -> 3283                 Connection._handle_dbapi_exception_noconnection(
       3284                     e, dialect, self
       3285                 )


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _handle_dbapi_exception_noconnection(cls, e, dialect, engine)
       2115             util.raise_(newraise, with_traceback=exc_info[2], from_=e)
       2116         elif should_wrap:
    -> 2117             util.raise_(
       2118                 sqlalchemy_exception, with_traceback=exc_info[2], from_=e
       2119             )


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3278         dialect = self.dialect
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        308
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311
        312     def _return_conn(self, record):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        866     def _checkout(cls, pool, threadconns=None, fairy=None):
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869
        870             fairy._pool = pool


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        474     @classmethod
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:
        478             dbapi_connection = rec.get_connection()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        144             except:
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:
        148             return self._do_get()


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        141         if self._inc_overflow():
        142             try:
    --> 143                 return self._create_connection()
        144             except:
        145                 with util.safe_reraise():


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        254         """Called by subclasses to create a new ConnectionRecord."""
        255
    --> 256         return _ConnectionRecord(self)
        257
        258     def _invalidate(self, connection, exception=None, _checkin=True):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        369         self.__pool = pool
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()
        373


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        664         except Exception as e:
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:
        668             # in SQLAlchemy 1.4 the first_connect event is not used by


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        659         try:
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)
        663             self.fresh = True


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        588                     if connection is not None:
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591
        592         creator = pop_kwarg("creator", connect)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        595     def connect(self, *cargs, **cparams):
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598
        599     def create_connect_args(self, url):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        351             self._sock = None
        352         else:
    --> 353             self.connect()
        354
        355     def __enter__(self):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        631
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634
        635             if self.sql_mode is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        919                 and plugin_name is not None
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:
        923                 # send legacy handshake


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1016
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()
       1020         return pkt


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (pymysql.err.OperationalError) (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
    (Background on this error at: https://sqlalche.me/e/14/e3q8)

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
      <th>no</th>
      <th>name</th>
      <th>eng</th>
      <th>math</th>
      <th>kor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>일일</td>
      <td>61</td>
      <td>72</td>
      <td>73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>12</td>
      <td>일이</td>
      <td>71</td>
      <td>83</td>
      <td>83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>일삼</td>
      <td>81</td>
      <td>92</td>
      <td>93</td>
    </tr>
  </tbody>
</table>
</div>

```python
sql_str = """
   select * from test
"""

pd.read_sql(sql_str, db)
```

    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\1933837676.py in <module>
          3 """
          4
    ----> 5 pd.read_sql(sql_str, db)


    NameError: name 'db' is not defined

```python
pd.read_sql(sql_str, engine)
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        142             try:
    --> 143                 return self._create_connection()
        144             except:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        255
    --> 256         return _ConnectionRecord(self)
        257


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        352         else:
    --> 353             self.connect()
        354


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")


    The above exception was the direct cause of the following exception:


    OperationalError                          Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\2511218155.py in <module>
    ----> 1 pd.read_sql(sql_str, engine)


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in read_sql(sql, con, index_col, coerce_float, params, parse_dates, columns, chunksize)
        591         )
        592     else:
    --> 593         return pandas_sql.read_query(
        594             sql,
        595             index_col=index_col,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in read_query(self, sql, index_col, coerce_float, parse_dates, params, chunksize, dtype)
       1558         args = _convert_params(sql, params)
       1559
    -> 1560         result = self.execute(*args)
       1561         columns = result.keys()
       1562


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in execute(self, *args, **kwargs)
       1403     def execute(self, *args, **kwargs):
       1404         """Simple passthrough to SQLAlchemy connectable"""
    -> 1405         return self.connectable.execution_options().execute(*args, **kwargs)
       1406
       1407     def read_table(


    <string> in execute(self, statement, *multiparams, **params)


    ~\anaconda3\lib\site-packages\sqlalchemy\util\deprecations.py in warned(fn, *args, **kwargs)
        400         if not skip_warning:
        401             _warn_with_version(message, version, wtype, stacklevel=3)
    --> 402         return fn(*args, **kwargs)
        403
        404     doc = func.__doc__ is not None and func.__doc__ or ""


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in execute(self, statement, *multiparams, **params)
       3173
       3174         """
    -> 3175         connection = self.connect(close_with_result=True)
       3176         return connection.execute(statement, *multiparams, **params)
       3177


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in connect(self, close_with_result)
       3232         """
       3233
    -> 3234         return self._connection_cls(self, close_with_result=close_with_result)
       3235
       3236     @util.deprecated(


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in __init__(self, engine, connection, close_with_result, _branch_from, _execution_options, _dispatch, _has_events, _allow_revalidate)
         94                 connection
         95                 if connection is not None
    ---> 96                 else engine.raw_connection()
         97             )
         98


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in raw_connection(self, _connection)
       3311
       3312         """
    -> 3313         return self._wrap_pool_connect(self.pool.connect, _connection)
       3314
       3315


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:
    -> 3283                 Connection._handle_dbapi_exception_noconnection(
       3284                     e, dialect, self
       3285                 )


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _handle_dbapi_exception_noconnection(cls, e, dialect, engine)
       2115             util.raise_(newraise, with_traceback=exc_info[2], from_=e)
       2116         elif should_wrap:
    -> 2117             util.raise_(
       2118                 sqlalchemy_exception, with_traceback=exc_info[2], from_=e
       2119             )


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3278         dialect = self.dialect
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        308
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311
        312     def _return_conn(self, record):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        866     def _checkout(cls, pool, threadconns=None, fairy=None):
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869
        870             fairy._pool = pool


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        474     @classmethod
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:
        478             dbapi_connection = rec.get_connection()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        144             except:
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:
        148             return self._do_get()


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        141         if self._inc_overflow():
        142             try:
    --> 143                 return self._create_connection()
        144             except:
        145                 with util.safe_reraise():


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        254         """Called by subclasses to create a new ConnectionRecord."""
        255
    --> 256         return _ConnectionRecord(self)
        257
        258     def _invalidate(self, connection, exception=None, _checkin=True):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        369         self.__pool = pool
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()
        373


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        664         except Exception as e:
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:
        668             # in SQLAlchemy 1.4 the first_connect event is not used by


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        659         try:
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)
        663             self.fresh = True


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        588                     if connection is not None:
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591
        592         creator = pop_kwarg("creator", connect)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        595     def connect(self, *cargs, **cparams):
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598
        599     def create_connect_args(self, url):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        351             self._sock = None
        352         else:
    --> 353             self.connect()
        354
        355     def __enter__(self):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        631
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634
        635             if self.sql_mode is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        919                 and plugin_name is not None
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:
        923                 # send legacy handshake


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1016
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()
       1020         return pkt


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (pymysql.err.OperationalError) (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
    (Background on this error at: https://sqlalche.me/e/14/e3q8)

```python
from sqlalchemy import create_engine

# MariaDB Connector using pymysql
pymysql.install_as_MySQLdb()

import MySQLdb

engine = create_engine("mysql://root:pass@127.0.0.1/newsdb",encoding="utf-8")
```

```python
df.to_sql(name="test",con=engine,
         if_exists="append", index=False)
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        142             try:
    --> 143                 return self._create_connection()
        144             except:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        255
    --> 256         return _ConnectionRecord(self)
        257


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        207         try:
    --> 208             raise exception
        209         finally:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        352         else:
    --> 353             self.connect()
        354


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1045, "Access denied for user 'root'@'localhost' (using password: YES)")


    The above exception was the direct cause of the following exception:


    OperationalError                          Traceback (most recent call last)

    ~\AppData\Local\Temp\ipykernel_58600\960457481.py in <module>
    ----> 1 df.to_sql(name="test",con=engine,
          2          if_exists="append", index=False)


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in to_sql(self, name, con, schema, if_exists, index, index_label, chunksize, dtype, method)
       2949         from pandas.io import sql
       2950
    -> 2951         return sql.to_sql(
       2952             self,
       2953             name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in to_sql(frame, name, con, schema, if_exists, index, index_label, chunksize, dtype, method, engine, **engine_kwargs)
        696         )
        697
    --> 698     return pandas_sql.to_sql(
        699         frame,
        700         name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in to_sql(self, frame, name, if_exists, index, index_label, schema, chunksize, dtype, method, engine, **engine_kwargs)
       1730         sql_engine = get_engine(engine)
       1731
    -> 1732         table = self.prep_table(
       1733             frame=frame,
       1734             name=name,


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in prep_table(self, frame, name, if_exists, index, index_label, schema, dtype)
       1629             dtype=dtype,
       1630         )
    -> 1631         table.create()
       1632         return table
       1633


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in create(self)
        830
        831     def create(self):
    --> 832         if self.exists():
        833             if self.if_exists == "fail":
        834                 raise ValueError(f"Table '{self.name}' already exists.")


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in exists(self)
        814
        815     def exists(self):
    --> 816         return self.pd_sql.has_table(self.name, self.schema)
        817
        818     def sql_schema(self):


    ~\anaconda3\lib\site-packages\pandas\io\sql.py in has_table(self, name, schema)
       1763             from sqlalchemy import inspect
       1764
    -> 1765             insp = inspect(self.connectable)
       1766             return insp.has_table(name, schema or self.meta.schema)
       1767         else:


    ~\anaconda3\lib\site-packages\sqlalchemy\inspection.py in inspect(subject, raiseerr)
         62             if reg is True:
         63                 return subject
    ---> 64             ret = reg(subject)
         65             if ret is not None:
         66                 break


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _engine_insp(bind)
        180     @inspection._inspects(Engine)
        181     def _engine_insp(bind):
    --> 182         return Inspector._construct(Inspector._init_engine, bind)
        183
        184     @inspection._inspects(Connection)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _construct(cls, init, bind)
        115
        116         self = cls.__new__(cls)
    --> 117         init(self, bind)
        118         return self
        119


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\reflection.py in _init_engine(self, engine)
        126     def _init_engine(self, engine):
        127         self.bind = self.engine = engine
    --> 128         engine.connect().close()
        129         self._op_context_requires_connect = True
        130         self.dialect = self.engine.dialect


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in connect(self, close_with_result)
       3232         """
       3233
    -> 3234         return self._connection_cls(self, close_with_result=close_with_result)
       3235
       3236     @util.deprecated(


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in __init__(self, engine, connection, close_with_result, _branch_from, _execution_options, _dispatch, _has_events, _allow_revalidate)
         94                 connection
         95                 if connection is not None
    ---> 96                 else engine.raw_connection()
         97             )
         98


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in raw_connection(self, _connection)
       3311
       3312         """
    -> 3313         return self._wrap_pool_connect(self.pool.connect, _connection)
       3314
       3315


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:
    -> 3283                 Connection._handle_dbapi_exception_noconnection(
       3284                     e, dialect, self
       3285                 )


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _handle_dbapi_exception_noconnection(cls, e, dialect, engine)
       2115             util.raise_(newraise, with_traceback=exc_info[2], from_=e)
       2116         elif should_wrap:
    -> 2117             util.raise_(
       2118                 sqlalchemy_exception, with_traceback=exc_info[2], from_=e
       2119             )


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\base.py in _wrap_pool_connect(self, fn, connection)
       3278         dialect = self.dialect
       3279         try:
    -> 3280             return fn()
       3281         except dialect.dbapi.Error as e:
       3282             if connection is None:


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in connect(self)
        308
        309         """
    --> 310         return _ConnectionFairy._checkout(self)
        311
        312     def _return_conn(self, record):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _checkout(cls, pool, threadconns, fairy)
        866     def _checkout(cls, pool, threadconns=None, fairy=None):
        867         if not fairy:
    --> 868             fairy = _ConnectionRecord.checkout(pool)
        869
        870             fairy._pool = pool


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in checkout(cls, pool)
        474     @classmethod
        475     def checkout(cls, pool):
    --> 476         rec = pool._do_get()
        477         try:
        478             dbapi_connection = rec.get_connection()


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        144             except:
        145                 with util.safe_reraise():
    --> 146                     self._dec_overflow()
        147         else:
        148             return self._do_get()


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\impl.py in _do_get(self)
        141         if self._inc_overflow():
        142             try:
    --> 143                 return self._create_connection()
        144             except:
        145                 with util.safe_reraise():


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in _create_connection(self)
        254         """Called by subclasses to create a new ConnectionRecord."""
        255
    --> 256         return _ConnectionRecord(self)
        257
        258     def _invalidate(self, connection, exception=None, _checkin=True):


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __init__(self, pool, connect)
        369         self.__pool = pool
        370         if connect:
    --> 371             self.__connect()
        372         self.finalize_callback = deque()
        373


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        664         except Exception as e:
        665             with util.safe_reraise():
    --> 666                 pool.logger.debug("Error on connect(): %s", e)
        667         else:
        668             # in SQLAlchemy 1.4 the first_connect event is not used by


    ~\anaconda3\lib\site-packages\sqlalchemy\util\langhelpers.py in __exit__(self, type_, value, traceback)
         68             self._exc_info = None  # remove potential circular references
         69             if not self.warn_only:
    ---> 70                 compat.raise_(
         71                     exc_value,
         72                     with_traceback=exc_tb,


    ~\anaconda3\lib\site-packages\sqlalchemy\util\compat.py in raise_(***failed resolving arguments***)
        206
        207         try:
    --> 208             raise exception
        209         finally:
        210             # credit to


    ~\anaconda3\lib\site-packages\sqlalchemy\pool\base.py in __connect(self)
        659         try:
        660             self.starttime = time.time()
    --> 661             self.dbapi_connection = connection = pool._invoke_creator(self)
        662             pool.logger.debug("Created new connection %r", connection)
        663             self.fresh = True


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\create.py in connect(connection_record)
        588                     if connection is not None:
        589                         return connection
    --> 590             return dialect.connect(*cargs, **cparams)
        591
        592         creator = pop_kwarg("creator", connect)


    ~\anaconda3\lib\site-packages\sqlalchemy\engine\default.py in connect(self, *cargs, **cparams)
        595     def connect(self, *cargs, **cparams):
        596         # inherits the docstring from interfaces.Dialect.connect
    --> 597         return self.dbapi.connect(*cargs, **cparams)
        598
        599     def create_connect_args(self, url):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in __init__(self, user, password, host, database, unix_socket, port, charset, sql_mode, read_default_file, conv, use_unicode, client_flag, cursorclass, init_command, connect_timeout, read_default_group, autocommit, local_infile, max_allowed_packet, defer_connect, auth_plugin_map, read_timeout, write_timeout, bind_address, binary_prefix, program_name, server_public_key, ssl, ssl_ca, ssl_cert, ssl_disabled, ssl_key, ssl_verify_cert, ssl_verify_identity, compress, named_pipe, passwd, db)
        351             self._sock = None
        352         else:
    --> 353             self.connect()
        354
        355     def __enter__(self):


    ~\anaconda3\lib\site-packages\pymysql\connections.py in connect(self, sock)
        631
        632             self._get_server_information()
    --> 633             self._request_authentication()
        634
        635             if self.sql_mode is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _request_authentication(self)
        919                 and plugin_name is not None
        920             ):
    --> 921                 auth_packet = self._process_auth(plugin_name, auth_packet)
        922             else:
        923                 # send legacy handshake


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _process_auth(self, plugin_name, auth_packet)
       1016
       1017         self.write_packet(data)
    -> 1018         pkt = self._read_packet()
       1019         pkt.check_error()
       1020         return pkt


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (pymysql.err.OperationalError) (1045, "Access denied for user 'root'@'localhost' (using password: YES)")
    (Background on this error at: https://sqlalche.me/e/14/e3q8)

```python

```
