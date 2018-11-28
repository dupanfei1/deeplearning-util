# -*- coding: utf-8 -*-
import csv
import os
from collections import Counter

# csv_reader1=csv.reader(open('densenet_test97.8.csv'))
# for row in csv_reader:
#     print(row[1])


result = []
for file in os.listdir('/home/lab548/Desktop/bd_tiangong/csv'):
    if os.path.splitext(file)[1] == '.csv':
        print file  # 打印所有csv格式的文件名
        result2 = []
        csv_reader = csv.reader(open(file))
        for row in csv_reader:
            result2.append(row[1])
        result.append(result2)
print 'hello'

com = []#综合结果
for i in range(1000):
    com2 = []
    for j in range(7):
        com2.append(result[j][i])
    com.append(com2)
print 'combine'


finalresult = []
for i in range(1000):
    counter_words = Counter(com[i])
    print(counter_words)
    most_counter = counter_words.most_common(1)
    finalresult.append(most_counter[0][0])

f = csv.reader(open("res/densenet_test96.9.csv"))
i = 0
# with open('res/write.csv', 'w') as f2:
#     for line in f:
#         print line[0]
#         writer = csv.writer(f2)
#         writer.writerows(line[0]+','+finalresult[i])
#         i = i+1
# f2.close()
import numpy as np
import pandas as pd

with open('res/write.txt', 'w') as f2:
    for line in f:
        print line[0]
        f2.writelines(line[0]+','+finalresult[i]+'\n')
        i = i+1
f2.close()

txt = np.loadtxt('res/write.txt')
txtDF = pd.DataFrame(txt)
txtDF.to_csv('res/write.csv',index=False)

# for filename in sorted(os.listdir(r"testB")):
#     b = filename.split('.')[0]
#     f.writelines('%s\n' % filename)
# f.close()
# print("train1 finish")
