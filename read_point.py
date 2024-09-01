import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data_path = "generated/airplane_vae/eval_points/sample/sample_000.npz"
data = dict(np.load(data_path)) #dict{'points': ndarray(2048,3),'splits: {ndarray:65},'paletten':{ndarray:(64,4)}}

xyz = data['points']
splits = data['splits']
paletten = data['palette']
# print(data)

def add_points(container, points, split, color: int or list or tuple):
    if type(color) is int or type(color) is tuple:
        color = (split.shape[0] - 1) * [color] #split.shape[0]=2，color=[255] / split.shape[0] =17
    for i in range(0, split.shape[0] - 1):
        c = color[i] #c =255 / c :(0.6196,0.0039,0.2588,1.0)
        if type(c) is int or type(c) is float:
            c = (c, c, c) #tuple(255,255,255)
        if type(c[0]) is int:
            c = [float(c[i]) / 255. for i in range(3)] #list:[1.0,1.0,1.0]
        # cur_points = clear_outside_points(points[split[i]: split[i + 1], :], bounds)
        # if len(cur_points) > 0:
        container.scatter(points[split[i]: split[i + 1], 0], points[split[i]: split[i + 1], 1], points[split[i]: split[i + 1], 2], marker='o', s=10, c=((c[0],c[1],c[2])),alpha=0.8)
        # plt.show()
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(projection='3d')
# ax = Axes3D(fig)
 # plt.axis('off')
ax.grid(False)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) #可以使用ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))来设置绘图区的背景颜色，其中参数(1.0, 1.0, 1.0, 1.0)表示白色，可以根据需要更改参数值来设置不同的背景颜色。
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))#去除背景板的颜色
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# Get rid of the spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))#去除线条的颜色
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
# Get rid of the ticks
ax.set_xticks([]) #设置无坐标轴刻度
ax.set_yticks([])
ax.set_zticks([])
    # ax_plot = ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='gray',s=80,alpha=0.1,marker='.')
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=80, c=feat, marker='o',cmap='rainbow')  # cmap='rainbow'与c搭配根据数值大小定颜色
# Get rid of the ticks
ax.set_xticks([])  # 设置无坐标轴刻度
ax.set_yticks([])
ax.set_zticks([])

add_points(ax,xyz,splits,paletten)


plt.show()
