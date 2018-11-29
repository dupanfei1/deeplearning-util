# -*- coding: utf-8 -*-

import requests as rq
import re
import time
import codecs
from multiprocessing.dummy import Pool,Queue #dummy子库是多线程库
import HTMLParser
unescape = HTMLParser.HTMLParser().unescape #用来实现对HTML字符的转义

tasks = Queue() #链接队列
tasks_pass = set() #已队列过的链接
results = {} #结果变量
count = 0 #爬取页面总数

tasks.put('/index.html') #把主页加入到链接队列
tasks_pass.add('/index.html') #把主页加入到已队列链接

def main(tasks):
    global results,count,tasks_pass #多线程可以很轻松地共享变量
    while True:
        url = tasks.get() #取出一个链接
        url = 'http://wap.xigushi.com'+url
        web = rq.get(url).content.decode('gbk') #这里的编码要看实际情形而定
        urls = re.findall('href="(/.*?)"', web) #查找所有站内链接
        for u in urls:
            if u not in tasks_pass: #把还没有队列过的链接加入队列
                tasks.put(u)
                tasks_pass.add(u)
        text = re.findall('<article>([\s\S]*?)</article>', web)
        #爬取我们所需要的信息，需要正则表达式知识来根据网页源代码而写
        if text:
            text = ' '.join([re.sub(u'[ \n\r\t\u3000]+', ' ', re.sub(u'<.*?>|\xa0', ' ', unescape(t))).strip() for t in text]) #对爬取的结果做一些简单的处理
            results[url] = text #加入到results中，保存为字典的好处是可以直接以url为键，实现去重
        count += 1
        if count % 100 == 0:
            print u'%s done.'%count

pool = Pool(4, main, (tasks,)) #多线程爬取，4是线程数
total = 0
while True: #这部分代码的意思是如果20秒内没有动静，那就结束脚本
    time.sleep(20)
    if len(tasks_pass) > total:
        total = len(tasks_pass)
    else:
        break

pool.terminate()
with codecs.open('results.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(results.values()))