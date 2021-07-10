# -*- encoding=utf-8 -*-
import datetime
import re
import linecache
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pandas as pd


def deal(listStr, i):
    # list转dataframe
    column = 'L2Voltage';
    df = pd.DataFrame(listStr, columns=[column])

    # 保存到本地excel
    IndexFile = 'L2Voltage'+'.xlsx'
    df.to_excel(IndexFile, index=False)


if __name__ == '__main__':
    dataall = []
    for i in range(1, 100):
        indexFile = "dnn" + str(i) + ".mt0"
        s = linecache.getline(indexFile, 5)
        print(s)
        strAfter = re.sub(' +', ',', s)
        data = strAfter.split(',')
        print(data)
        datalist = data[1025: 1537]
        print(datalist)
        dataall = dataall + datalist
    deal(dataall, i)
