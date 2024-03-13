
import re
import pandas as pd
import numpy as np
import random
import math
import time
from itertools import permutations
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import copy
mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']  # SimSun 是宋体的英文名
from sklearn.cluster import KMeans
# 从Excel文件中读取数据
file_path_coordinates = '坐标.xlsx'
file_path_personnel = '人员.xlsx'
file_path_ships = '船舶信息.xlsx'

coordinates_data = pd.read_excel(file_path_coordinates)
personnel_data = pd.read_excel(file_path_personnel)
ships_data = pd.read_excel(file_path_ships)


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
n = len(coordinates_data)  # 风机数量
m = len(ships_data)  # 船舶数量
port = (coordinates_data['Longitude_DD'].iloc[-1], coordinates_data['Latitude_DD'].iloc[-1])  # 港口坐标（Excel中的最后一行）
# 找到港口的坐标序号数
port_idx = coordinates_data.index[coordinates_data['Longitude_DD'] == port[0]][0]
dist = [(coordinates_data['Longitude_DD'].iloc[i], coordinates_data['Latitude_DD'].iloc[i]) for i in
        range(n - 1)]  # 风机坐标
cluster_groups = {1: [19, 1, 2, 3, 13, 17], 3: [19, 4, 5, 6, 7, 8, 9], 2: [19, 10, 11, 12, 14, 15, 16, 18]}
cluster_sites = {1: [], 2: [], 3: []}
cluster_group = {1: [1, 2, 3, 13, 17], 2: [10, 11, 12, 14, 15, 16, 18], 3: [19, 4, 5, 6, 7, 8, 9]}
end_times_per_cluster = {}
for cluster, indices in cluster_group.items():
    end_times = [coordinates_data.loc[idx - 1, 'end_time'] for idx in indices]
    end_times_per_cluster[cluster] = end_times
max_end_times = {cluster: max(end_times) for cluster, end_times in end_times_per_cluster.items()}
# 创建一个字典用于存储每组风机的维修时间总和
total_maintenance_time = {}
# 遍历每个组，计算维修时间总和
for cluster, wind_farm_indices in cluster_group.items():
    total_time = 0  # 用于存储每组的维修时间总和
    for wind_farm_index in wind_farm_indices:
        # 从坐标数据中获取对应风机的维修时间
        maintenance_time = coordinates_data['repair_time'][wind_farm_index - 1]  # 风机序号从1开始，而索引从0开始
        total_time += maintenance_time
    total_maintenance_time[cluster] = total_time
# 将风机坐标按照分类分组
for cluster, indices in cluster_groups.items():
    for idx in indices:
        latitude = coordinates_data['Latitude_DD'].iloc[idx - 1]  # 索引从1开始，需要减1
        longitude = coordinates_data['Longitude_DD'].iloc[idx - 1]
        cluster_sites[cluster].append((longitude, latitude))

def calculate_sailing_cost(route, coords, ship_idx):
    """计算运维成本"""
    # 计算行驶公里数
    sailing_distance = 0
    for i in range(len(route) - 1):
        coord1 = coords[route[i]]
        coord2 = coords[route[i + 1]]
        sailing_distance += calculate_distance(coord1, coord2)
    # 计算运维成本（租赁费用 + 行驶公里数 * 油耗）
    leasing_cost = ships_data['租赁费用/万元'][ship_idx]
    fuel_cost = ships_data['油耗（万元每公里）'][ship_idx]
    sailing_cost = leasing_cost + sailing_distance * fuel_cost
    return sailing_cost

def calculate_punish_cost(route, coords, ship_idx):
    """计算惩罚成本"""
    punish = 0.24
    sailing_distance = 0
    for i in range(len(route) - 1):
        coord1 = coords[int(route[i])]  # 将浮点数索引转换为整数
        coord2 = coords[int(route[i + 1])]  # 将浮点数索引转换为整数
        sailing_distance += calculate_distance(coord1, coord2)

    speed = ships_data['速度（公里每小时）'][ship_idx]
    travel_time = sailing_distance / speed
    repair_time = total_maintenance_time[ship_idx + 1]
    total_time = repair_time + travel_time # 假设维修时间、航行时间
    deadline = max_end_times[ship_idx + 1]

    # 计算超时时间
    over_time = max(0, total_time - deadline)

    # 计算惩罚成本
    punish_cost = over_time * punish
    return punish_cost

def total_cost(route, coords, ship_idx):
    """计算总成本"""
    punish_cost = calculate_punish_cost(route, coords, ship_idx)

    # 计算总成本（船舶数量*租赁成本 + 行驶公里*油耗 + 超时时间*惩罚成本）
    leasing_cost = ships_data['租赁费用/万元'][ship_idx]
    fuel_cost = ships_data['油耗（万元每公里）'][ship_idx]

    total_cost = (leasing_cost + 2* fuel_cost * sum([calculate_distance(coords[route[i]], coords[route[i + 1]]) for i in range(len(route) - 1)])+ punish_cost * len(route))
    return total_cost


# 计算每艘船舶的行驶公里数
def calculate_distance(coord1, coord2):
    # 使用简化的球面距离公式计算两点间的距离（单位：公里）
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    radius = 6371  # 地球半径（单位：公里）
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    km = radius * c
    nautical_miles = km
    return nautical_miles

total_distances = {}
def total_distance(route, coords):
    """计算路径的总距离"""
    dist = 0
    for i in range(len(route) - 1):
        dist += calculate_distance(coords[route[i]], coords[route[i + 1]])
    return dist

def greedy_initial_solution(coords):
    """使用贪婪算法生成初始解"""
    unvisited = set(range(len(coords)))
    current = random.choice(list(unvisited))
    path = [current]
    unvisited.remove(current)
    while unvisited:
        nearest = min(unvisited, key=lambda x: calculate_distance(coords[current], coords[x]))
        path.append(nearest)
        current = nearest
        unvisited.remove(current)
    return path
def particle_swarm_initial_solution(coords, num_particles=50, num_iterations=100):
    best_positions = []
    best_fitness = float('inf')

    for _ in range(num_particles):
        particle_position = greedy_initial_solution(coords)  # 使用贪婪算法生成初始解
        particle_fitness = total_cost(particle_position, coords, ship_idx)  # 计算适应度（成本）

        if particle_fitness < best_fitness:
            best_fitness = particle_fitness
            best_positions = particle_position

    for _ in range(num_iterations):
        for i in range(num_particles):
            particle_position = greedy_initial_solution(coords)
            particle_fitness = total_cost(particle_position, coords, ship_idx)

            if particle_fitness < best_fitness:
                best_fitness = particle_fitness
                best_positions = particle_position

    return best_positions

def tournament_selection(population, fitness_values, tournament_size=3):
    """轮盘赌选择"""
    selected_indices = random.sample(range(len(population)), tournament_size)
    selected = [population[i] for i in selected_indices]
    selected_fitness = [fitness_values[i] for i in selected_indices]
    return selected[selected_fitness.index(min(selected_fitness))]
def order_crossover(parent1, parent2):
    """顺序交叉"""
    start = random.randint(0, len(parent1) - 1)
    end = random.randint(start, len(parent1))
    child = [-1] * len(parent1)
    child[start:end] = parent1[start:end]
    remaining = [item for item in parent2 if item not in child]
    idx = 0
    for i in range(len(child)):
        if child[i] == -1:
            child[i] = remaining[idx]
            idx += 1
    return child

def mutate(route, mutation_rate=0.01):
    """变异操作"""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


def remove_and_reinsert(route, coords):
    num_points_to_remove = random.randint(1, len(route) - 1)
    removed_indices = random.sample(range(len(route)), num_points_to_remove)
    removed_points = [route[idx] for idx in removed_indices]
    route = [point for point in route if point not in removed_points]

    # 随机选择插入位置
    insert_index = random.randint(0, len(route))
    route = route[:insert_index] + removed_points + route[insert_index:]

    return route


def adaptive_large_neighborhood_search(route, coords, ship_idx):
    best_route = route
    best_cost = total_cost(route, coords, ship_idx)

    for _ in range(5):  # 可根据实际情况调整迭代次数
        # 移除并重新插入操作
        route = remove_and_reinsert(route, coords)

        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                new_cost = total_cost(new_route, coords, ship_idx)

                if new_cost < best_cost:
                    best_cost = new_cost
                    best_route = new_route

    return best_route


# 定义GA算法函数
def genetic_algorithm(coords, ship_idx, population_size=50, generations=50, mutation_rate=0.001):
    population = [particle_swarm_initial_solution(sites) for _ in range(population_size)]
    costs_history = []  # 记录每代种群的成本
    for generation in range(generations):
        start_time = time.time()  # 记录每次迭代开始时间
        # 评估种群中每个个体的适应度（成本）
        fitness_values = [(total_cost(route, coords, ship_idx), total_distance(route, coords)) for route in population]

        # 记录每代最优个体的成本
        min_cost = min([cost[0] for cost in fitness_values])
        costs_history.append(min_cost)
        # 选择父代
        parents = [tournament_selection(population, fitness_values) for _ in range(population_size)]

        # 生成子代
        children = [order_crossover(parents[i], parents[i + 1]) for i in range(0, population_size, 2)]

        # 变异子代
        mutated_children = [mutate(child, mutation_rate) for child in children]

        # 更新种群
        population = mutated_children
        # 引入自适应大邻域搜索算法
        for i in range(len(population)):
            population[i] = adaptive_large_neighborhood_search(population[i], sites, ship_idx)

        end_time = time.time()  # 记录每次迭代结束时间
        iteration_time = end_time - start_time
        print(f"Iteration {generation + 1}: {iteration_time:.2f} seconds")
    # 找到最优路径
    best_route = min(population, key=lambda route: total_cost(route, coords, ship_idx))
    return best_route, costs_history

# 计算船舶的最优路径和成本变化历史
total_costs_history = []  # 记录船舶的总成本变化历史
best_routes = {}  # 记录每艘船舶的最优路径
best_costs = {}  # 记录每艘船舶的最优成本

for cluster, sites in cluster_sites.items():
    ship_idx = cluster - 1  # 船舶编号从0开始
    best_route, costs_history = genetic_algorithm(sites, ship_idx)
    best_routes[cluster] = best_route
    best_cost = total_cost(best_route, sites, ship_idx)
    best_costs[cluster] = best_cost
    total_costs_history.append(costs_history)

# 计算船舶的总成本随迭代次数的变化
total_costs = [sum(costs) for costs in zip(*total_costs_history)]
# 可视化船舶总成本随迭代次数的变化
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(total_costs) + 1), total_costs,label='operational and maintenance cost')
plt.xlabel('iterations',fontsize=24)
plt.ylabel('operational and maintenance cost',fontsize=24)
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimSun']  # SimSun 是宋体的英文名
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('iterations')
plt.ylabel('operational and maintenance cost')
plt.grid(True)
plt.legend()
plt.show()

# 输出每艘船舶的最优成本
for ship, cost in best_costs.items():
    print(f"{ship} 的最优成本: {cost:.2f}")
# 用坐标序号对应到坐标内容，输出最优路径的内容（减1）
ship_routes = {}
for cluster, route in best_routes.items():
    route_coords = [cluster_sites[cluster][i] for i in route]
    route_coords.append(route_coords[0])  # 回到起点
    route_indices = []
    for lon, lat in route_coords:
        idx = coordinates_data[(coordinates_data['Longitude_DD'] == lon) & (coordinates_data['Latitude_DD'] == lat)].index[0]
        route_indices.append(idx)  # 坐标序号从0开始
    ship_routes[f'ship{cluster}'] = route_indices

# 打印船舶最优路径
for ship, route in ship_routes.items():
    print(f"{ship}: {route}")
# 计算每艘船舶的行驶公里数
total_distances = {}
for ship, route in ship_routes.items():
    total_distance = 0
    for i in range(len(route) - 1):
        coord1 = (coordinates_data.loc[route[i]]['Latitude_DD'], coordinates_data.loc[route[i]]['Longitude_DD'])
        coord2 = (coordinates_data.loc[route[i + 1]]['Latitude_DD'], coordinates_data.loc[route[i + 1]]['Longitude_DD'])
        distance = calculate_distance(coord1, coord2)
        total_distance += distance
    total_distances[ship] = total_distance

# 输出每艘船舶的行驶公里数
for ship, distance in total_distances.items():
    print(f"{ship}的行驶公里数: {distance:.2f} 公里")

# 船舶航线数量
num_ship_routes =3
 # 每艘船对应的颜色
plt.figure(figsize=(12, 12))  # 调整图像大小

# 绘制故障风机
plt.scatter(coordinates_data['Longitude_DD'], coordinates_data['Latitude_DD'], color='b', label='Faulty fan', s=80)

# 绘制变压站
plt.scatter(port[0], port[1], c='r', marker='s', label='Inspection center', s=80)
plt.annotate('Start', (port[0], port[1]), fontsize=24)  # 添加变压站标签

# 标记故障风机序号
for i, d in enumerate(dist):
    plt.annotate(f'{i + 1}', (d[0], d[1]), fontsize=24)

# 绘制船舶维修路径，并添加箭头
prev = port
# 使用不同的颜色绘制每艘船的路线
colors = ['#8ECFC9', '#FFBE7A', '#FA7F6F', '#87CEFA', '#BEB8DC']
for i, (ship, route) in enumerate(ship_routes.items()):
    ship_coordinates = coordinates_data.loc[route][['Longitude_DD', 'Latitude_DD']]
    color_index = i % len(colors)

    plt.plot(
        ship_coordinates['Longitude_DD'],
        ship_coordinates['Latitude_DD'],
        marker='o',
        color=colors[color_index],
        label=f'{ship} inspection path'
    )

    for j in range(1, len(route)):
        dx = ship_coordinates['Longitude_DD'].iloc[j] - ship_coordinates['Longitude_DD'].iloc[j - 1]
        dy = ship_coordinates['Latitude_DD'].iloc[j] - ship_coordinates['Latitude_DD'].iloc[j - 1]

        plt.quiver(
            ship_coordinates['Longitude_DD'].iloc[j - 1],
            ship_coordinates['Latitude_DD'].iloc[j - 1],
            dx,
            dy,
            angles='xy',
            scale_units='xy',
            scale=1,
            color=colors[color_index]
        )
plt.xlabel('longitude', fontsize=24)
plt.ylabel('latitude', fontsize=24)
plt.title('Ship inspection path', fontsize=24)
plt.grid(True)
plt.legend(fontsize=16)
plt.show()