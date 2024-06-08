# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 16:15:07 2023

@author: Jerry Li
"""
import vtk
from moviepy.editor import ImageSequenceClip
import os

# 自定义排序函数，确保文件名数字正确排序
def sort_key(file_name):
    # 提取文件名中的数字，并转换为整数
    number_part = int(file_name.split('_')[1].split('.png')[0])
    return number_part

folder_path = 'AnimationFrames'

# 获取所有图片文件的路径，并使用自定义排序函数排序
image_files = [os.path.join(folder_path, img) for img in sorted(os.listdir(folder_path), key=sort_key) if img.endswith(".png")]

# 确保所有图片都包括在内
print(f"总共有 {len(image_files)} 张图片")

# 创建视频剪辑对象
clip = ImageSequenceClip(image_files, fps=24)  # 每秒24帧

# 设置输出文件名和格式
output_video = "output_video_simplified_160_24帧.mp4"

# 写入视频文件
clip.write_videofile(output_video, codec='libx264')


'''
ffmpeg_command = [
    "ffmpeg",
    "-framerate", str(framerate),  # 设置帧率
    "-i", f"{frame_path}/frame_%d.png",  # 输入文件的格式和路径
    "-c:v", "libx264",  # 使用x264编解码器
    "-pix_fmt", "yuv420p",  # 设置像素格式
    video_name  # 输出视频的文件名
]

try:
    result = subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode())
    print("Video created successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e.stderr.decode()}")
'''

'''
images = [img for img in os.listdir(frame_path) if img.endswith(".png")]
images.sort()  # 如果需要，根据命名规则对图片进行排序

# 读取一张图片以获取宽度和高度
frame = cv2.imread(os.path.join(frame_path, images[0]))
height, width, layers = frame.shape

# 创建视频编写器对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(frame_path, image)))

cv2.destroyAllWindows()
video.release()
'''