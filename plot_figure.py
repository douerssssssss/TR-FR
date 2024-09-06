"""
绘图文件，PPT展示所需要的所有图片代码
"""



################图片1,绘制条行图，分数据集，横坐标病人，纵坐标mae值，两组，蓝色appoint,橙色random，绿色上次的5个病人结果。

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


"""
csv_path = '/home/mininet/cha/data/mc.csv'


data_df = pd.read_csv(csv_path,sep=',',encoding='utf-8')
data = data_df.values
patient_x = data[:,0].tolist()
mae_appoint = data[:,1].tolist()
mae_random = data[:,3].tolist()


bar_width = 0.4  # 条形宽度
bar_distance = 1.0  # 相邻条形之间的距离

# 计算相邻条形的位置
positions1 = np.arange(len(patient_x)) * (bar_width + bar_distance)
positions2 = positions1 + bar_width
positions3 = positions2 + bar_width

# 创建条形图
plt.bar(positions1, mae_appoint, width=bar_width, label='MAE Appoint')
plt.bar(positions2, mae_random, width=bar_width, label='MAE Random')


# 设置横坐标刻度
plt.xticks((positions1 + positions2) / 2, patient_x)

# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel('Patients')
plt.ylabel('Mae_Values')
plt.title('Capnobase')

# 显示图形
plt.show()

"""

import matplotlib.pyplot as plt
import numpy as np
"""
# 生成示例数据
x = np.linspace(0, 10, 100)
loss1 = np.sin(x)
loss2 = np.cos(x)
loss3 = np.sin(2 * x)
loss4 = np.cos(2 * x)
loss5 = np.sin(3 * x)
loss6 = np.cos(3 * x)

# 画图
plt.plot(x, loss1, '--', color='blue', marker='^', label='Loss 1')
plt.plot(x, loss2, '-.', color='green', marker='s', label='Loss 2')
plt.plot(x, loss3, '-.', color='red', marker='o', label='Loss 3')
plt.plot(x, loss4, '--', color='purple', marker='D', label='Loss 4')
plt.plot(x, loss5, '-.', color='orange', marker='+', label='Loss 5')
plt.plot(x, loss6, '--', color='brown', marker='p', label='Loss 6')

# 添加图例和标签
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Loss values')
plt.title('Multiple Losses')

# 显示图形
plt.show()
"""
path4 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-18-13-30)_losses.csv'  # 0.6  0.4  ConcordanceCorrelationCoefficient_Loss


def read_csv(path):
    data_df = pd.read_csv(path, sep=',', encoding='utf-8')
    data = data_df['Val Loss'].values[0:200]

    return data

# loss plot

loss_path1 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-17-50-17)_losses.csv'  # 0    1
loss_path2 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-17-57-46)_losses.csv'  # 0.2  0.8
loss_path3 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-18-06-14)_losses.csv'  # 0.4  0.6
loss_path4 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-18-21-41)_losses.csv'  # 0.6  0.4
loss_path5 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-18-28-26)_losses.csv'  # 0.8  0.2
loss_path6 = '/home/mininet/PycharmProjects/pythonProject4/s4_dalition/result/loss/2024-03-06-18-34-35)_losses.csv'  # 1    0



loss1 = read_csv(loss_path1)
loss2 = read_csv(loss_path2)
loss3 = read_csv(loss_path3)
loss4 = read_csv(loss_path4)
loss5 = read_csv(loss_path5)
loss6 = read_csv(loss_path6)


plt.plot(loss1, '-', linewidth =1.0,color='maroon',label=r'$\alpha$'+'==0' +'   Loss 1')    # coral  sandybrown  greenyellow  deepskyblue  cyan  m hotpink  mediumpurple
plt.plot(loss2, '--', linewidth =1.0,color='deepskyblue', label=r'$\alpha$'+'==0.2' +'  Loss 2')
plt.plot(loss3, '-', linewidth =1.0,color='deeppink',  marker='+', label=r'$\alpha$'+'==0.4' +'  Loss 3')
plt.plot(loss4, '--', linewidth =1.0,color='olive', label=r'$\alpha$'+'==0.6' +'  Loss 4')
plt.plot(loss5, '-.', linewidth =1.0,color='orange', label=r'$\alpha$'+'==0.8' +'  Loss 5')
plt.plot(loss6, '-', linewidth =1.0,color='brown',  label=r'$\alpha$'+'==1' +'   Loss 6')

# 添加图例和标签
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Loss values')
plt.title('Loss')
plt.savefig('/home/mininet/PycharmProjects/pythonProject4/s4_dalition/figure/loss.png',dpi=1000,bbox_inches='tight')
# 显示图形
plt.show()



