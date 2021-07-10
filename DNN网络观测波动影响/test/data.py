import re
import linecache

# 512个数据
s = linecache.getline('dnn11.mt0', 4)
print(s)

strAfter = re.sub(' +', ',', s)
data = strAfter.split(',')
print(data)
# 把string类型转换为lsit
#strAfter = re.sub(' +', ',', s)
#listStr = strAfter.split(',')
#print(listStr)
# listStr = list(strAfter)
# print(listStr)
