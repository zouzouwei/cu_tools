import time
import numpy as np
import pcl
import os
import os.path as osp
import open3d as o3d
import mmap

from pcl import PointCloud


def parse_points(data):
    num_points = len(data) // 4
    points = np.zeros((num_points, 4), dtype=np.float32)
    for i in range(num_points):
        points[i, 0] = data[i * 4]  # X
        points[i, 1] = data[i * 4 + 1]  # Y
        points[i, 2] = data[i * 4 + 2]  # Z
        points[i, 3] = data[i * 4 + 3]  # Intensity
    return points


def load_points1(filename):
    with open(filename, "r") as f:
        # 使用 mmap 将文件映射到内存中
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # 计算点云数据的元素大小
        element_size = 16  # 假设每个点由4个浮点数组成，每个浮点数占4个字节
        # 计算缓冲区大小，确保是元素大小的倍数
        buffer_size = (len(mm) // element_size) * element_size
        # 使用 np.frombuffer 从内存中读取数据
        points = np.frombuffer(mm[:buffer_size], dtype=np.float32).reshape(-1, 4)
    return points

def load_points2(pts_filename):                
    with open(pts_filename, "r") as f:
        points_list = []
        lines = f.readlines()
        data_start = False
        for line in lines:
            if line.startswith("DATA"):
                data_start = True
                continue
            if data_start:
                values = line.split()
                x = float(values[0])
                y = float(values[1])
                z = float(values[2])
                intensity = float(values[3])
                temp = 0.
                points_list.append([x, y, z, intensity,temp])
        points=np.array(points_list)

def load_points_o3d(pts_filename): 
    pcd = o3d.t.io.read_point_cloud(osp.join(pts_filename),format='pcd')
    pcd_intensity = pcd.point["intensity"] #强度
    pcd_points = pcd.point["positions"] #坐标
    pcd_intensity = pcd_intensity[:, :].numpy()
    pcd_points = pcd_points[:, :].numpy()

    # 创建一个全零的列，并将其拼接在pcd_points的第四列
    zeros = np.zeros((pcd_points.shape[0], 1), dtype=pcd_intensity.dtype)
    pcd_points_with_intensity = np.concatenate((pcd_points, pcd_intensity, zeros), axis=1)
    return pcd_points_with_intensity

def load_points_txt(pts_filename):
    with open(pts_filename) as pcd_file:
        point_list = []
        for i in range(11):
            pcd_file.readline()
        pcd = pcd_file.readlines()
        for pcd_line in pcd:
            pcd_line = pcd_line.split(" ")
            point_list.append(
                [float(pcd_line[0]), float(pcd_line[1]), float(pcd_line[2]), float(pcd_line[3][:-1]), 0.])
        points = np.array(point_list, dtype=np.float32).reshape(-1, 5)
    return points
# # 定义 pcd 文件路径
# pts_dir_file = "/media/heying/data/code/bevfusion/data/nuscenes/samples/2023_09_24/2023_09_24_changchun_ehs9_28/rslidar_points/"

# pcd_files_list = os.listdir(pts_dir_file)

# 开始计时
start_time = time.time()
idx = 0

# for pts_filename in pcd_files_list:
for i in range(20):
    # 读取 pcd 文件并转换为 numpy 数组
    # pcd_points = np.array(pcl.load_XYZI(osp.join(pts_dir_file,pts_filename)), dtype=np.float32)
    # 加载点云数据
    # pcd_points = load_points1(osp.join(pts_dir_file, pts_filename))

    pcd_points_txt = load_points_txt("1686898840341939.pcd")

    pcd_points_o3d = load_points_o3d("1686898840341939.pcd")

    # 计算两个numpy数组之间的差异
    difference = np.abs(pcd_points_txt - pcd_points_o3d)

    # # 打印差异的最大值、最小值和平均值
    print("最大差异：", np.max(difference))
    print("最小差异：", np.min(difference))
    print("平均差异：", np.mean(difference))

    idx += 1

    if idx >=10:
        break
# 结束计时
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time

# 打印执行时间
print(f"代码执行时间：{execution_time} 秒")


