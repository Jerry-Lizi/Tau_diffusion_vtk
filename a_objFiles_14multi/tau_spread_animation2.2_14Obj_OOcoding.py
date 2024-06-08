"""
Created on Thu Dec 22 09:42:54 2023

@author: Jerry Li

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


class ObjFileProcessor:
    def __init__(self, file_path):
        """
        初始化ObjFileProcessor类的实例。

        参数:
            file_path (str): OBJ文件的路径。
        """
        self.file_path = file_path
        self.vertices = None
        self.triangles = None

    def read_obj_file(self):
        """
        读取OBJ文件并提取顶点和三角形数据。
        """
        reader = vtk.vtkOBJReader()
        reader.SetFileName(self.file_path)
        reader.Update()

        polydata = reader.GetOutput()
        self.vertices, self.triangles = self.extract_geometry_data(polydata)

    def extract_geometry_data(self, polydata):
        """
        从vtkPolyData对象中提取顶点和三角形数据。

        参数:
            polydata (vtkPolyData): VTK的PolyData对象。

        返回:
            tuple: 包含顶点数组和三角形数组的元组。
        """
        vertices = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())], dtype=np.float32)
        triangles = []
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            if cell.GetCellType() == vtk.VTK_TRIANGLE:
                indices = [cell.GetPointId(j) for j in range(cell.GetNumberOfPoints())]
                triangles.append(indices)

        return vertices, triangles


class DistanceCalculator:
    @staticmethod
    def calculate_triangle_centers(vertices, triangles):
        """
        计算三角形的中心点。

        参数:
            vertices (np.array): 顶点数组。
            triangles (list): 三角形列表，其中每个元素是顶点索引的列表。

        返回:
            np.array: 各三角形中心点的坐标数组。
        """
        centers = np.empty((len(triangles), 3))
        for i, triangle in enumerate(triangles):
            centers[i] = np.mean(vertices[triangle], axis=0)
        return centers

    @staticmethod
    def calculate_distance_matrix(centers):
        """
        计算三角形中心点之间的欧几里得距离矩阵。

        参数:
            centers (np.array): 三角形中心点的坐标数组。

        返回:
            np.array: 中心点之间的距离矩阵。
        """
        # 计算每对中心点之间的差异
        diff = centers[:, np.newaxis] - centers
        # 计算距离矩阵（使用欧几里得距离公式）
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
        # 将矩阵对角线上的元素设置为0（每个点与自身的距离）
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix

class TauProteinSpreadSimulator:
    def __init__(self, vertices, triangles, distance_matrix):
        """
        初始化TauProteinSpreadSimulator类的实例。

        参数:
            vertices (np.array): 顶点数组。
            triangles (list): 三角形列表。
            distance_matrix (np.array): 三角形中心点之间的距离矩阵。
        """
        self.vertices = vertices
        self.triangles = triangles
        self.distance_matrix = distance_matrix
        # 初始化更多需要的属性，例如tau蛋白的初始分布等

    def calculate_spread_coefficients(self):
        """
        计算tau蛋白扩散的系数。
        """
        # 根据距离矩阵和其他参数计算扩散系数
        # 这可能涉及复杂的计算，具体取决于模型的细节
        # 返回计算得到的系数
        return coefficients

    def spread_iteration(self, current_distribution, coefficients):
        """
        执行一次tau蛋白扩散迭代。

        参数:
            current_distribution: 当前的tau蛋白分布。
            coefficients: 扩散系数。

        返回:
            更新后的tau蛋白分布。
        """
        # 实现扩散的逻辑
        # 这可能涉及到对当前分布的更新，根据扩散系数和其他参数
        updated_distribution = ...
        return updated_distribution

    def simulate_spread(self, iterations, initial_distribution):
        """
        模拟tau蛋白的扩散过程。

        参数:
            iterations (int): 迭代次数。
            initial_distribution: tau蛋白的初始分布。

        返回:
            list: 每次迭代后的tau蛋白分布。
        """
        distribution_history = [initial_distribution]
        coefficients = self.calculate_spread_coefficients()

        for _ in range(iterations):
            new_distribution = self.spread_iteration(distribution_history[-1], coefficients)
            distribution_history.append(new_distribution)

        return distribution_history


class Visualization:
    @staticmethod
    def draw_and_save_image(data, file_path, renderer_settings=None):
        """
        绘制并保存图像。

        参数:
            data: 需要绘制的数据。
            file_path (str): 保存图像的文件路径。
            renderer_settings: 渲染器的设置，用于自定义绘图。
        """
        # 设置VTK渲染器，绘制data
        renderer = vtk.vtkRenderer()
        if renderer_settings:
            # 应用自定义设置
            pass

        # 添加数据到渲染器
        # ...

        # 设置渲染窗口并保存图像
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.Render()

        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(file_path)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    @staticmethod
    def create_video(image_folder, output_video, fps=24):
        """
        从图像创建视频。

        参数:
            image_folder (str): 包含图像的文件夹。
            output_video (str): 输出视频的文件路径。
            fps (int): 每秒帧数。
        """
        image_files = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
        clip = ImageSequenceClip(image_files, fps=fps)
        clip.write_videofile(output_video, codec='libx264')
