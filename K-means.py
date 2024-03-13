import re
import pandas as pd
import numpy as np
import random
import math
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import copy
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']  # SimSun 是宋体的英文名
from sklearn.cluster import KMeans
# 从Excel文件中读取数据
file_path_coordinates = '坐标.xlsx'

coordinates_data = pd.read_excel(file_path_coordinates)


# 定义DMS格式转换为十进制度数的函数
def dms_to_dd(dms):
    parts = re.split('[°′″]+', dms)
    degrees = float(parts[0])
    minutes = 0
    seconds = 0
    if len(parts) > 1:
        minutes = float(parts[1])
    if len(parts) > 2:
        seconds = float(parts[2])
    dd = degrees + minutes / 60 + seconds / 3600
    return dd

# 将纬度和经度从DMS格式转换为十进制度数
coordinates_data['Latitude_DD'] = coordinates_data['latitude'].apply(dms_to_dd)
coordinates_data['Longitude_DD'] = coordinates_data['longitude'].apply(dms_to_dd)

# 随机生成数据
random.seed(2023)
n = len(coordinates_data)  # 风机数量
m =3
# 去掉最后一个坐标即出发点
coordinates_data = coordinates_data[:-1]
# 提取需要聚类的特征
features = coordinates_data[['Longitude_DD','Latitude_DD', ]]

# 使用K-means聚类将风机坐标分成船舶数量的类别
num_ships =3# 船舶数量
kmeans = KMeans(n_clusters=num_ships, random_state=0).fit(coordinates_data[['Longitude_DD','Latitude_DD']])
coordinates_data['Cluster'] = kmeans.labels_

# 将聚类结果加入数据框
coordinates_data['cluster'] = kmeans.labels_
# 获取每个聚类分组的风机坐标序号
cluster_groups = {}
for cluster_label in range(m):
    cluster_indices = list(coordinates_data[coordinates_data['cluster'] == cluster_label].index + 1)  # 加1是因为索引从0开始
    cluster_groups[cluster_label + 1] = cluster_indices

# 打印每个聚类分组的风机坐标序号
for cluster_label, indices in cluster_groups.items():
    print(f"船舶{cluster_label}的维修任务风机坐标序号：{indices}")
# 可视化聚类结果
plt.figure(figsize=(8, 6))
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F']
for cluster_label, color in zip(range(5), colors):
    cluster_data = coordinates_data[coordinates_data['cluster'] == cluster_label]
    plt.scatter( cluster_data['Longitude_DD'],cluster_data['Latitude_DD'], color=color, label=f'ship {cluster_label+1}task')
plt.grid(True)
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.title('Fan maintenance task assignment')
plt.legend()
plt.show()


