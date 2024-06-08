# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 20:19:24 2023
@author: Jerry Li
Description:  原始准确的（两个区域之间的各个三角形之间）代码——进行多个obj文件的读取和绘制。
ChangedParts: 读取160个obj文件。先读取14个的

各个区域与操作的rgb配色：
需要区分的 注射区域：(0.5, 0.5, 0) 橄榄绿
点击高亮颜色： (0.941, 0.5, 0.5) 浅珊瑚
模拟tau蛋白扩散颜色： (0.5， 0， 0)栗色 

整个绘图框的背景颜色：(0.1, 0.2, 0.4) 蓝色

"""
import vtk
import numpy as np
import pandas as pd
import copy
import os
from moviepy.editor import ImageSequenceClip
from scipy.spatial.distance import cdist

# 注射的话，规定的前NUM_INJECTED个三角形值设置为1.
NUM_INJECTED = 1
# 如果要进行注射，每个三角形的初始tau蛋白浓度。s
initial_concentration = 1/4000
# 需要用到的obj文件夹路径：
ObjFiles_FolderName = 'ObjFilesSplit/'
# 用于动画制作的关键帧存储路径：
frame_path = "AnimationFrames/"
# 欧几里得项系数改变：
Euclidean_changed = 0.9
internal_changed = 0.7
# 改变自身区域扩散的系数
b0_changed = 1

minValue = 0
maxValue = 0.01
color_value = 10000  #映射的精度
iteration_Number = 50 #迭代次数
# 设定浓度的上限值
#concentractions_threshold = 10e+5
#max_concentration = 200  
# 原始方程中 关于时间t的相关系数
t_t = -0.058069745 #-0.0052
# 用于存储文件内容的字典，方便直接在内存中调用。而不是每次读取相关数据，都得经过函数read_obj_file
obj_content = {}

#obj_files = ['left_894_RSPagl.obj', 'left_394_VISam.obj', 'left_564_MS.obj', 'right_894_RSPagl.obj', 'right_394_VISam.obj', 'right_564_MS.obj'] 
# 提前计算所有文件名
#file_names = [f.split('/')[-1].replace('.obj', '') for f in obj_files]
# 读取CSV文件
df = pd.read_csv('objFiles_sorted_name14.csv')
file_names = df['column_name'].tolist()
obj_files = [file + '.obj' for file in file_names]

obj_num = len(obj_files)
# injected_obj_files = ['iRSPagl', 'iCA1', 'iCA3', 'iDG']
# injected_obj_files = ['left_894_RSPagl.obj', 'left_382_CA1.obj', 'left_463_CA3.obj', 'left_726_DG.obj']
# injected_obj_files = ['left_564_MS.obj']
injected_obj_files = ['iAOB.obj']
#injected_obj_files = ['left_394_VISam.obj', 'left_564_MS.obj']
# 示例使用的参数
b_coefficients = pd.read_csv('Tau_coefficients/Tau_b_Values.csv', header=None).to_numpy()
b0 = b_coefficients[0]
b1 = b_coefficients[1]
b2 = b_coefficients[2]
b3 = b_coefficients[3]

w_r_ij = pd.read_csv('Tau_coefficients/Tau_W_r_Values.csv', header=None).to_numpy()
# anterograde即retrograde的转置
w_a_ij = w_r_ij.T
# 预计算exp(-w_r_ij)和exp(-w_a_ij)
exp_w_r_ij = np.exp(-w_r_ij)
exp_w_a_ij = np.exp(-w_a_ij)



D_ij = pd.read_csv('Tau_coefficients/Tau_D_Values.csv', header=None).to_numpy()

def calculate_triangle_centers(vertices, triangles):
    triangle_points = vertices[triangles]  # 一个三维数组，其中每个元素表示一个三角形的三个顶点
    centers = np.mean(triangle_points, axis=1)  # 直接沿着第二维（即每个三角形的顶点维）计算平均值
    return centers
    #return np.array([np.mean(vertices[triangle], axis=0) for triangle in triangles])

# 计算距离矩阵--使用NumPy的广播机制来计算每对三角形中心之间的距离
# 非常快速！高效！！！

def calculate_distance_matrix(centers):
    # 使用scipy.spatial.distance.cdist计算距离矩阵
    # centers是一个(N, 3)的数组，其中N是中心点的数量
    dist_matrix = cdist(centers, centers) / 10
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

# 通过欧氏距离，计算顺行和逆行扩散的权重
def calculate_euclidean_weights(distance_matrix):
    D_log_squared_region = np.log(distance_matrix ** 2)
    D_log_squared_region[distance_matrix == 0] = 0  # 替换距离为0的情况

    # 计算倒数，同样避免了使用 np.where
    D_region_original = 1. / D_log_squared_region
    D_region_original[D_log_squared_region == 0] = 0  # 替换原先为0的情况

    # 将大于2的值替换为2，小于0.1857的值替换为0
    D_region_upper = np.clip(D_region_original, 0, 2)  # 使用np.clip进行范围限制

    return D_region_upper
 
# 通过欧氏距离，计算欧几里得项的系数
def calculate_euclidean_sigmoid_coefficient(distance_matrix):
    D_log_squared_region = np.log(distance_matrix ** 2)
    D_log_squared_region[distance_matrix == 0] = 0  # 替换距离为0的情况

    # 计算倒数，同样避免了使用 np.where
    D_region_original = 1. / D_log_squared_region
    D_region_original[D_log_squared_region == 0] = 0  # 替换原先为0的情况

    # 将大于2的值替换为2，小于0.1857的值替换为0
    D_region = np.clip(D_region_original, 0, 2)  # 使用np.clip进行范围限制
    D_region[D_region < 0.1857] = 0

    # 构建sigmoid函数，同样避免了使用 np.where
    D_sigmoid = 1 / (1 + np.exp(-D_region))
    D_sigmoid[D_region == 0] = 0  # 替换原先为0的情况
    return D_sigmoid

# 读取OBJ文件+计算区域内部的关联系数 的函数
def read_obj_file(file_path):
    # 读取OBJ文件
    folder_file_path = ObjFiles_FolderName + file_path
    reader = vtk.vtkOBJReader()
    reader.SetFileName(folder_file_path)
    reader.Update()
    # 获取vtkPolyData对象
    polydata = reader.GetOutput()

    '''为获得区域内部各个三角形之间的距离关系系数——去获取相关的三角形的顶点、面等信息'''
    # 获取顶点数据
    # 可以通过直接转换 VTK 数据到 NumPy 数组来优化。VTK 提供了将其数据转换为 NumPy 数组的功能，这通常比手动循环更快。
    vertices = np.array(polydata.GetPoints().GetData())

    # 计算模型中心位置
    model_center = np.mean(vertices, axis=0)
    # 计算三角形数量
    num_triangles = polydata.GetNumberOfCells()
    # 获取三角形面的顶点索引
    triangles = []
    for i in range(num_triangles):
        cell = polydata.GetCell(i)
        indices = [cell.GetPointId(j) for j in range(3)]  # 假设每个单元都是三角形
        triangles.append(indices)

    centers = calculate_triangle_centers(vertices, triangles)
    # 计算距离矩阵
    distance_matrix = calculate_distance_matrix(centers)

    # 计算区域内部各三角形面片之间的欧几里得系数
    D_internal_region_sigmoid = calculate_euclidean_sigmoid_coefficient(distance_matrix)

    # 创建region_concentrations数组
    # ps：！这段之后：把internal、external合并的时候记得只用初始化一次
    region_concentrations = np.zeros(num_triangles)
    # 如果在injected_regions中，则设置相应区域为1：
    if file_path in injected_obj_files:
        # 假设先注射前NUM_INJECTED个小三角形面片。
        region_concentrations[:NUM_INJECTED] = initial_concentration   # 前NUM_INJECTED个元素设为1/4000
    # 创建region_external_concentrations数组
    region_external_concentrations = np.copy(region_concentrations)
    # 创建region_internal_concentrations数组
    region_internal_concentrations = np.copy(region_concentrations)
    
    return {
        'triangles_centers': centers,   #保存各个区域（obj文件）的各个三角形的中心位置，防止之后计算跨区域之间欧几里得系数时的重复读取与计算，提高代码运行速度。
        'num_triangles': num_triangles,
        'model_center': model_center,
        'region_whole_concentrations_sum': np.sum(region_concentrations),
        'original_concentrations': region_concentrations,
        'region_concentrations': region_concentrations,
        'region_external_concentrations': region_external_concentrations,
        'region_internal_concentrations': region_internal_concentrations,
        #'internal_region_distance_matrixes': distance_matrix,
        'D_internal_region_sigmoid': D_internal_region_sigmoid,
    }


# 计算内部区域的tau蛋白扩散（考虑各个三角形之间的影响）
def internal_spread(concentractions_i, D_internal_region_sigmoid, b0):
    sigmoid_term = D_internal_region_sigmoid * concentractions_i
    sum_term = np.sum(sigmoid_term, axis=1) * Euclidean_changed
    # 计算更新的region_1_internal_concentrations
    new_region_internal_concentrations = b0_changed * b0 * concentractions_i + sum_term
    
    return new_region_internal_concentrations * internal_changed

# 计算跨区域中的 欧氏距离扩散的tau蛋白浓度影响
def external_euclidean_spread(i, obj_contents):
    # 包含现在需要计算区域浓度的 区域各个三角形浓度；
    # 计算其他各个区域j整体的 对该区域i的各个三角形面的影响
    file_name_i = file_names[i]
    region_content_i = obj_contents[file_name_i]
    external_euclidean_concentrations_i = np.zeros(obj_contents[file_name_i]['num_triangles'])
    for j in range(obj_num):
        # 不考虑自己本身区域的 跨区域的欧几里得影响：（即i==j的情况直接跳过）
        if i != j:
            file_name_j = file_names[j]
            # 考虑对区域i的影响，故索引使用file_name_j中的 i
            # 且D等于0时，不考虑计算，结果为0 
            # 参考：np.where(D_internal_region_sigmoid != 0, np.exp(D_internal_region_sigmoid) * region_original_concentrations, 0)
            
            # 使用矩阵的向量乘法直接获得区域j各个三角形对区域i各个三角形的影响。
            #D_exp = np.exp(obj_contents[file_name_i]['D_external_region_sigmoid'][j])
            external_euclidean_concentrations_ji = region_content_i['D_external_region_sigmoid'][j].dot(obj_contents[file_name_j]['region_concentrations'])
            external_euclidean_concentrations_i += external_euclidean_concentrations_ji
    return b3[i] * external_euclidean_concentrations_i * Euclidean_changed

# 计算跨区域中的 逆行扩散的tau蛋白浓度影响:
# 计算时考虑索引w_r_ij[j][i], 表示其他区域向i的扩散
def external_retrograde_spread(i, obj_contents):
    file_name_i = file_names[i]
    external_retrograde_concentrations_i = np.zeros(obj_contents[file_name_i]['num_triangles'])
    
    for j in range(obj_num):
        if w_r_ij[j][i] != 0:
            # 顺行扩散只考虑存在轴突树突时，即W_r_ji 等于0时，则不予计算。
            # 计算retrograde逆行扩散项，如果w_r_ij != 0
            # 需要考虑自身区域的轴突树突，即i==j的情况。
            file_name_j = file_names[j]
            latter_half = (obj_content[file_name_j]['region_whole_concentrations_sum'] / (1 + exp_w_r_ij[j][i]))
            # 根据区域j到 区域i各个三角形的距离不同，设定不同的逆行扩散系数
            external_retrograde_concentrations_ji = obj_contents[file_name_j]['ante_retrograde_W_ij_weight'][i] * latter_half
            external_retrograde_concentrations_i += external_retrograde_concentrations_ji
            
    return b1[i] * external_retrograde_concentrations_i
    

# 计算跨区域中的 顺行扩散的tau蛋白浓度影响:
# 计算时考虑索引w_a_ij[j][i], 表示其他区域向i的扩散
def external_anterograde_spread(i, obj_contents):
    file_name_i = file_names[i]
    external_anterograde_concentrations_i = np.zeros(obj_contents[file_name_i]['num_triangles'])
    
    for j in range(obj_num):
        if w_a_ij[j][i] != 0:
            # 顺行扩散只考虑存在轴突树突时，即W_r_ji 等于0时不予计算。
            # 计算retrograde逆行扩散项，如果w_r_ij != 0
            #需要考虑自身区域的轴突树突，即i==j的情况。
            file_name_j = file_names[j]
            latter_half = (obj_contents[file_name_j]['region_whole_concentrations_sum'] / (1 + exp_w_a_ij[j][i]))
            # 根据区域j到 区域i各个三角形的距离不同，设定不同的逆行扩散系数
            external_anterograde_concentrations_ji = obj_contents[file_name_j]['ante_retrograde_W_ij_weight'][i] * latter_half
            external_anterograde_concentrations_i += external_anterograde_concentrations_ji
    return b2[i] * external_anterograde_concentrations_i
   
# 构建跨区域情况下 顺行扩散中的权重系数函数(对原始跨区域中 欧几里得系数归一化)
# 对多个数组（即obj_content[file_name_i]['D_external_region']中的所有数值）进行操作
def retrograde_W_ij_normalized(D_external_region):
    epsilon = 1e-10  # 防止除以零
    normalized_D_external_region = []
    for D_external_region_i in D_external_region:
        min_val = D_external_region_i.min()
        max_val = D_external_region_i.max()
        # 避免分母为零的情况
        range_val = max(max_val - min_val, epsilon)
        normalized_i = np.nan_to_num((D_external_region_i - min_val) / range_val)
        normalized_D_external_region.append(normalized_i)
    #normalized_D_external_region = [np.nan_to_num((D_external_region_i - D_external_region_i.min()) / (D_external_region_i.max() - D_external_region_i.min())) for D_external_region_i in D_external_region]
    # 之后再乘上e，来扩大差异？
    
    return normalized_D_external_region

'''
    计算初始注射条件下的第一次扩散的 内部和外部区域tau蛋白浓度，之后再进行迭代
'''
# 遍历文件中所有的obj文件
# 计算内部区域和外部区域的共同影响：
# 首先计算内部区域的影响，同时也读取了初始条件下的obj文件。
for i in range(obj_num):
    internal_concentrations_i = 0
    file_path_i = obj_files[i]
    file_name_i = file_names[i]

    # 可以通过首先将所有文件的内容读入内存，然后在需要时直接从内存中获取，从而避免重复读取。
    obj_content[file_name_i] = read_obj_file(file_path_i)
    
    region_content_i = obj_content[file_name_i]
    internal_concentrations_i = internal_spread(region_content_i['original_concentrations'], region_content_i['D_internal_region_sigmoid'], b0[i])
    region_content_i['region_internal_concentrations'] = internal_concentrations_i

'''    
    计算跨区域情况下，j中各个三角形与区域i之间的距离和欧几里得系数：
    以及 顺行扩散中的 不同三角形的tau蛋白扩散影响的权重系数。（由于是通过轴突末梢的扩散，因此该扩散也遵循靠近细胞体的扩散的较多）
'''
# 计算跨区域情况下，j中各个三角形与区域i之间的距离和欧几里得系数
# 该欧几里得系数是简化版的：整体区域i到j中三角形的距离
# 为节省内存，不再保留各个三角形之间的距离信息.
# 用于计算得到逆行、顺行扩散时各个小三角形的权重系数。
for i in range(obj_num):
    file_name_i = file_names[i]
    region_content_i = obj_content[file_name_i]
    file_name_i_center = region_content_i['model_center']
    # 这里不需要使用NumPy数组，因为要存储的是由不同长度的数组。
    #external_region_distance_matrixes_simplified_i = []
    #D_external_region_simplified_i = []
    D_external_region_simplified_i_initial = []
    
    for j in range(obj_num):
        file_name_j = file_names[j]
        region_content_j = obj_content[file_name_j]
        if j == i:
            num_triangles = region_content_j['num_triangles']
            zeros_array = np.zeros(num_triangles)
            external_region_distance_matrixes_simplified_ij = zeros_array
            #D_external_region_simplified_ij = zeros_array
            D_external_region_simplified_ij_initial = zeros_array
        else:
            # 使用 cdist 来计算距离
            # 确保 file_name_i_center 是二维数组
            file_name_i_center_2d = file_name_i_center[np.newaxis, :]
            external_region_distance_matrixes_simplified_ij = cdist(region_content_j['triangles_centers'], file_name_i_center_2d, 'euclidean').flatten() / 10
            D_external_region_simplified_ij_initial = calculate_euclidean_weights(external_region_distance_matrixes_simplified_ij)
            
        # 将区域j各个三角形到 区域i的欧几里得项更新至参数数组中去。
        #external_region_distance_matrixes_simplified_i.append(external_region_distance_matrixes_simplified_ij)
        #D_external_region_simplified_i.append(D_external_region_simplified_ij)
        D_external_region_simplified_i_initial.append(D_external_region_simplified_ij_initial)
    
    # 将每个区域i 对其他区域j的距离、欧几里得等信息保存在原字典中去。
    # ps:由于该列表中的数组 大小不同，因此不能直接将其转换成一个二维数组。
    #region_content_i['external_region_distance_matrixes_simplified'] = external_region_distance_matrixes_simplified_i
    #region_content_i['D_external_region_simplified'] = D_external_region_simplified_i
    region_content_i['D_external_region_simplified_initial'] = D_external_region_simplified_i_initial

# 计算跨区域情况下，j中各个三角形 与 i中各个三角形之间的距离和欧几里得系数。
# 为节省内存，不再保留各个三角形之间的距离信息
for i in range(obj_num):
    file_name_i = file_names[i]
    region_content_i = obj_content[file_name_i]

    num_triangles_i = region_content_i['num_triangles']
    triangles_centers_i = region_content_i['triangles_centers']
    # 这里不需要使用NumPy数组，因为要存储的是由不同长度的数组。
    #external_region_distance_matrixes_i = [] #包含的是一系列的 大小不同的二维数组。:i-0、i-1、i-2...
    D_external_region_i = []
    
    for j in range(obj_num):
        file_name_j = file_names[j]
        region_content_j = obj_content[file_name_j]

        num_triangles_j = region_content_j['num_triangles']
        triangles_centers_j = region_content_j['triangles_centers']
        
        if j == i:
            zeros_matrix = np.zeros((num_triangles_i, num_triangles_j))
            external_region_distance_matrixes_ij = zeros_matrix
            D_external_region_ij = zeros_matrix
        else:
            # 使用 cdist 计算距离矩阵
            external_region_distance_matrixes_ij = cdist(triangles_centers_i, triangles_centers_j) / 10
            D_external_region_ij = calculate_euclidean_sigmoid_coefficient(external_region_distance_matrixes_ij)
        
        #external_region_distance_matrixes_i.append(external_region_distance_matrixes_ij)
        D_external_region_i.append(D_external_region_ij)
    
    # 'external_region_distance_matrixes'，'D_external_region'储存的是多个二维数组。
    #region_content_i['external_region_distance_matrixes'] = external_region_distance_matrixes_i
    region_content_i['D_external_region_sigmoid'] = D_external_region_i
                


# 计算顺行扩散中的 不同三角形的tau蛋白扩散影响的权重系数(根据欧式距离来计算)  
for i in range(obj_num):
    file_name_i = obj_files[i].split('/')[-1].replace('.obj', '')
    region_content_i = obj_content[file_name_i]

    anterograde_W_ij_weight_i = retrograde_W_ij_normalized(region_content_i['D_external_region_simplified_initial'])
    region_content_i['ante_retrograde_W_ij_weight'] = anterograde_W_ij_weight_i

'''
计算 初始 注射条件下的第一次扩散的 跨区域tau蛋白浓度，之后再进行迭代
'''
#tt1 = np.array([-0.063203, -1.658832, -9.319879, -14.395618])
#tt2 = np.array([-0.05293649, -1.730772, -7.7784014, -12.183602]) 

# 计算跨区域的影响：
for i in range(obj_num):
    file_name_i = file_names[i]
    region_content_i = obj_content[file_name_i]
    
    region_content_i['external_euclidean_concentrations'] = external_euclidean_spread(i, obj_content)
    region_content_i['external_retrograde_concentrations'] = external_retrograde_spread(i, obj_content)
    region_content_i['external_anterograde_concentrations'] = external_anterograde_spread(i, obj_content)
    
    region_content_i['region_external_concentrations'] = region_content_i['external_euclidean_concentrations'] + region_content_i['external_retrograde_concentrations'] + region_content_i['external_anterograde_concentrations']
    region_content_i['region_concentrations']  = np.abs((region_content_i['region_internal_concentrations'] + region_content_i['region_external_concentrations']) * t_t)
    

'''
    计算迭代n次后的区域中的tau蛋白浓度
'''
def spread_iteration(obj_content):
    for i in range(obj_num):
        file_name_i = file_names[i]
        region_content_i = obj_content[file_name_i]  # 减少重复的数据访问

        # 直接在原有的字典里面更新数据
        original_concentrations = region_content_i['region_concentrations']
        
        # 计算内部区域的tau蛋白扩散
        region_internal_concentrations = internal_spread(original_concentrations, region_content_i['D_internal_region_sigmoid'], b0[i])
        
        # 计算跨区域的影响
        external_euclidean_concentrations = external_euclidean_spread(i, obj_content)
        external_retrograde_concentrations = external_retrograde_spread(i, obj_content)
        external_anterograde_concentrations = external_anterograde_spread(i, obj_content)
        region_external_concentrations = external_euclidean_concentrations + external_retrograde_concentrations + external_anterograde_concentrations
        region_concentrations = np.abs((region_internal_concentrations + region_external_concentrations) * t_t)
        
        #region_concentrations = np.minimum(region_concentrations, max_concentration)
        # 更新obj_content中的数据
        region_content_i['region_internal_concentrations'] = region_internal_concentrations
        region_content_i['external_euclidean_concentrations'] = external_euclidean_concentrations
        region_content_i['external_retrograde_concentrations'] = external_retrograde_concentrations
        region_content_i['external_anterograde_concentrations'] = external_anterograde_concentrations
        region_content_i['region_external_concentrations'] = region_external_concentrations
        region_content_i['region_concentrations'] =  np.log1p(region_concentrations)
        #region_content_i['region_concentrations'] = region_concentrations
        
    return obj_content

'''
    将小鼠大脑区域内 各个三角形的浓度标量值 映射在各个三角形上。
'''
#concentrations_map = 10e+8
#minValue = 0
#maxValue = 102400
#color_value = 10000
def scalar_map_draw(obj_contents, renderer, renderWindow):
    # 清空当前渲染器中的所有actor
    renderer.RemoveAllViewProps()
    
    # 创建查找表（颜色映射表）
    lookupTable = vtk.vtkLookupTable()
    lookupTable.SetNumberOfTableValues(color_value)
    lookupTable.SetRange(minValue, maxValue)

    # 创建RGB和透明度数组
    colors = np.zeros((color_value, 4))
    for i in range(color_value):
        value = minValue + (maxValue - minValue) * i / (color_value - 1)
        adjusted_value = np.power(value / maxValue, 2)  # 使用幂律映射放大数值差异

        r = 0.2 * (1.0 - adjusted_value)
        g = 0.7 + 0.3 * adjusted_value
        b = r
        alpha = adjusted_value

        colors[i] = [r, g, b, alpha]

    # 一次性设置颜色映射表
    for i, color in enumerate(colors):
        lookupTable.SetTableValue(i, *color)

    # 遍历并处理每个OBJ文件
    for obj_name, scalar_values in obj_contents.items():
        obj_path = ObjFiles_FolderName + obj_name + '.obj'
        
        # 读取OBJ文件
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_path)
        reader.Update()
        
        # 获取几何数据
        polydata = reader.GetOutput()

        # 创建标量值数组
        scalars = vtk.vtkFloatArray()
        scalars_map = scalar_values['region_concentrations']
        for value in scalars_map:
            scalars.InsertNextTuple1(value)
        
        # 将标量值数组添加到polydata中
        polydata.GetCellData().SetScalars(scalars)

        # 创建映射器和actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetLookupTable(lookupTable)
        mapper.SetScalarRange(minValue, maxValue)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # 设置actor的属性
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetEdgeColor(1, 1, 1)
        actor.GetProperty().SetLineWidth(0.01)

        # 将actor添加到渲染器
        renderer.AddActor(actor)

    # 设置渲染窗口为离屏渲染模式
    renderWindow.SetOffScreenRendering(1)
    renderer.SetBackground(0.1, 0.2, 0.4)  # 设置渲染器的背景颜色
    renderWindow.Render()
    
# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindow.SetSize(1920, 1080)  # 设置渲染窗口的尺寸为1080p：1920x1080 ; 720p：1280x720。
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# 创建窗口到图像的过滤器
windowToImageFilter = vtk.vtkWindowToImageFilter()
windowToImageFilter.SetInput(renderWindow)
windowToImageFilter.SetInputBufferTypeToRGB()
windowToImageFilter.ReadFrontBufferOff()

printIterations = [1] + [i * 10 + 4 for i in range(150)]
# 构建tau蛋白扩散的迭代n次的函数
def iteration_num(n, obj_content, iteration):
    for i in range(n):
        obj_content = spread_iteration(obj_content)
    #if np.max(obj_content['region_concentrations']) > concentractions_threshold:
        #xx = Euclidean_changed * 0.8
    #if iteration in printIterations:
    #    print(iteration, np.max(obj_content['left_894_RSPagl']['region_concentrations']))
    # 绘制并保存图像，不返回obj_content
    scalar_map_draw(obj_content, renderer, renderWindow)
    windowToImageFilter.Modified()
    windowToImageFilter.Update()
    
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(frame_path + f"frame_{iteration}.png")
    writer.SetInputConnection(windowToImageFilter.GetOutputPort())
    writer.Write()

# 预计算所有迭代的结果
iterations =[1] + [i for i in range(iteration_Number)]


# 直接从初始内容开始迭代，不使用precalculated_results字典
obj_content_current = copy.deepcopy(obj_content)

# 迭代并保存每一帧的图片
for i in range(len(iterations)):
    current_iteration = iterations[i]
    if i == 0:
        iteration_num(1, obj_content_current, current_iteration)  # 第一次迭代
    else:
        previous_iteration = iterations[i - 1]
        iteration_diff = current_iteration - previous_iteration
        iteration_num(iteration_diff, obj_content_current, current_iteration)  # 后续迭代
        #print(obj_content_current['region_concentrations'])

# 使用每次得到的迭代图片，创建扩散视频的函数。
def create_video(folder_path, output_video, fps=24):
    """
    从指定文件夹中的图片创建视频。
    
    参数:
    folder_path -- 包含图片的文件夹路径
    output_video -- 输出视频文件的名称
    fps -- 每秒帧数
    """
    # 自定义排序函数，确保文件名数字正确排序
    def sort_key(file_name):
        # 提取文件名中的数字，并转换为整数
        number_part = int(file_name.split('_')[1].split('.png')[0])
        return number_part
    # 获取所有图片文件的路径，并使用自定义排序函数排序
    image_files = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path), key=sort_key) if img.endswith(".png")]
    # 确保所有图片都包括在内
    print(f"总共有 {len(image_files)} 张图片，准备生成视频...")
    # 创建视频剪辑对象
    clip = ImageSequenceClip(image_files, fps=fps)
    # 写入视频文件
    clip.write_videofile(output_video, codec='libx264')

# 使用示例
# create_video('AnimationFrames', 'output_video.mp4', fps=24)
