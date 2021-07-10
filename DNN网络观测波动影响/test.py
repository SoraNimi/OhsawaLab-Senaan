#-*- encoding=utf-8 -*-
import datetime
import re
import linecache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd


def deal(listStr):
    # list转dataframe
    df = pd.DataFrame(listStr, columns=['L0 Voltage'])

    # 保存到本地excel
    df.to_excel("fluction1.xlsx", index=False)

if __name__ == '__main__':
    s = linecache.getline('dnn11.mt0', 4)
    print(s)
    strAfter = re.sub(' +', ',', s)
    data = strAfter.split(',')
    print(data)    # 把string类型转换为lsit
    strAfter = re.sub(' +', ',', s)
    data = strAfter.split(',')
    print(data)
    deal(data)