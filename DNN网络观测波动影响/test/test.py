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
    df.to_excel("Input number1.xlsx", index=False)

if __name__ == '__main__':
    s = linecache.getline('dnn11.mt0', 4)
    print(s)
    strAfter = re.sub(' +', ',', s)
    data = strAfter.split(',')
    print(data)
    datalist = data[1:513]
    print(datalist)
    deal(datalist)