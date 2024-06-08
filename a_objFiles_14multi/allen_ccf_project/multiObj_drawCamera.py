# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 08:36:57 2023
@author: Jerry Li
Description: 绘制所有的160个obj文件以找到 合适的 绘制角度。
ChangedParts: 现在的代码中，取消了 打印照相机坐标的 这一功能。
"""

import vtk
import os

# 文件夹路径
folder_path = "ObjFiles_allSplit"  # 替换为含有OBJ文件的文件夹路径
#folder_path = "ObjFilesSplit"
additional_folder_path = "ObjFiles_ignored"  # 替换为另一个含有OBJ文件的文件夹路径

# 创建渲染器和渲染窗口
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# 上次点击的actor
last_picked_actor = None

# 创建一个颜色映射表，用于高亮显示
highlightProp = vtk.vtkProperty()
highlightProp.SetColor(1, 0, 0)  # 高亮颜色设置为红色

# 定义设置actor为默认颜色的函数
def setActorToDefault(actor):
    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # 默认颜色设置为灰色
    actor.GetProperty().SetOpacity(1.0)  # 默认不透明

# 定义加载OBJ文件并添加到渲染器的函数
def loadObjFiles(folder_path, isTransparent):
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".obj"):
            full_path = os.path.join(folder_path, file_name)

            # 创建OBJ文件的读取器
            reader = vtk.vtkOBJReader()
            reader.SetFileName(full_path)
            reader.Update()

            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            # 创建actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            setActorToDefault(actor)  # 设置默认颜色

            # 如果需要透明效果
            #if isTransparent:
            actor.GetProperty().SetOpacity(0.3)  # 设置更透明

            # 将actor添加到渲染器中
            renderer.AddActor(actor)

# 加载两个文件夹中的OBJ文件
loadObjFiles(folder_path, False)
#loadObjFiles(additional_folder_path, True)

# 创建拾取器
picker = vtk.vtkPropPicker()

# 定义点击事件的回调函数
def onClick(event, obj):
    global last_picked_actor

    # 获取点击位置的坐标
    x, y = renderWindowInteractor.GetEventPosition()
    # 执行拾取操作
    picker.Pick(x, y, 0, renderer)

    # 获取拾取到的actor
    pickedActor = picker.GetActor()

    # 如果上次有actor被选中，则恢复其颜色
    if last_picked_actor:
        setActorToDefault(last_picked_actor)

    # 如果有actor被拾取，并且拾取到的不是透明的actor
    if pickedActor and pickedActor.GetProperty().GetOpacity() > 0.3:
        # 设置高亮属性
        pickedActor.SetProperty(highlightProp)
        # 更新上次点击的actor
        last_picked_actor = pickedActor
    elif not pickedActor:  # 如果没有拾取到actor，也需要重置上次点击的actor
        last_picked_actor = None

    # 重新绘制渲染窗口
    renderWindow.Render()

    # 打印摄像机参数
    #printCameraParameters()

# 定义一个函数来获取并打印摄像机参数
def printCameraParameters():
    camera = renderer.GetActiveCamera()
    print("Camera Position:", camera.GetPosition())
    print("Focal Point:", camera.GetFocalPoint())
    print("View Up:", camera.GetViewUp())

# 绑定鼠标左键点击事件
renderWindowInteractor.AddObserver("LeftButtonPressEvent", onClick)

# 开始交互
renderWindow.Render()
renderWindowInteractor.Start()