import re
import linecache

# 512个数据
s = linecache.getline('dnn1.mt0', 1)
print(s)
# 把string类型转换为lsit
#strAfter = re.sub(' +', ',', s)
#listStr = strAfter.split(',')
#print(listStr)
# listStr = list(strAfter)
# print(listStr)
# draw_hist(listStr,'AreasList','Area','number',50.0,250,0.0,100)   # 直方图展示
# draw_hist(listStr,'perimeterList','Area','number',1,3,0.1,30)
