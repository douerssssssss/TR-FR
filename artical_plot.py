import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


###绘制评价指标图，跨数据集。

MAE1 = np.array([4.96, 6.27, 15.04, 3.13, 5.51, 10.2, 6.5,1.1])
#MSE1 = np.array([41.14,45.08,288,14.8,44.2,158,42.2,1.8])
RMSE1 = np.array([5.28,6.63,16.96,3.84,6.65,12.6,6.5,1.34])
LOA1 = np.array([7.07,4.99,9.65,7.5,11.3,25.2,6.5,2])

x_value = np.concatenate((MAE1,RMSE1,LOA1))


metrics = ['RRWaveNet','TransRR','ConvMixer','BiLSTM+Atten','BiLSTM','ResNet','RespNet','TR-FR']
model_name = ['MAE','RMSE','LOA']


# 绘制柱形图
fig, ax = plt.subplots()
index = np.arange(len(model_name))
bar_width = 0.1
opacity = 0.5

i = 0

while i < len(x_value):
    #la = metrics[i%8]
    ax.bar(i * bar_width, x_value[i], bar_width, alpha=opacity,)
    i = i+1

"""
for i, metric in enumerate(metrics):
    ll = i*len(model_name)
    pp = (i+1)*len(model_name)
    ax.bar(i*bar_width,x_value[i], bar_width, alpha=opacity, label=metric)
    #ax.bar(index + i * bar_width, x_value[i*len(model_name):(i+1)*len(model_name)], bar_width, alpha=opacity, label=metric)
"""


ax.set_xlabel('Model_Name')
ax.set_ylabel('Values')
ax.set_title('Mertics')
ax.set_xticks(index + bar_width)
ax.set_xticklabels(model_name)
ax.legend()

plt.tight_layout()
plt.show()