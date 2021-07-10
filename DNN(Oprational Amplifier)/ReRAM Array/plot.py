# matplotlib中有很多可用的模块，我们使用pyplot模块
from matplotlib import pyplot

Vth = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
Accuracy = [98, 97, 87, 75, 38, 10, 10]
# 生成图表
pyplot.plot(Vth, Accuracy)
# 设置横坐标为year，纵坐标为population，标题为Population year correspondence
pyplot.xlabel('Vth')
pyplot.ylabel('Accuracy')
pyplot.title('Hardware Accuracy Result under Fluctuation')
# 设置纵坐标刻度
pyplot.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# 显示图表
pyplot.show()
