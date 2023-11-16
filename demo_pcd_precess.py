import concurrent.futures
import pcl
import numpy as np
import time


def load_XYZI(pts_filename):
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     point_clouds = list(executor.map(pcl.load_XYZI, filepaths))
    # return np.array(point_clouds, dtype=np.float32)
    # futures = [(pcl.load_XYZI, filepath) for filepath in filepaths]

    pcd_points=np.array(pcl.load_XYZI(pts_filename), dtype=np.float32)
    return pcd_points

def load_XYZI_parallel(filepaths, num_workers=4):
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     point_clouds = list(executor.map(pcl.load_XYZI, filepaths))
    # return np.array(point_clouds, dtype=np.float32)
    print(filepaths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(pcl.load_XYZI, filepath) for filepath in filepaths]

    point_clouds = [future.result() for future in futures]
    pcd_points = np.concatenate(point_clouds).astype(np.float32)
    return pcd_points

filepaths = ['1686898840341939.pcd', '1686898841341630.pcd', '1686898842341104.pcd','1686898843342099.pcd']

start = time.time()
pcd_points = load_XYZI_parallel(filepaths, num_workers=4)
end = time.time()
print(end - start)


sin_start = time.time()
for i in filepaths:
    sin_pcd_points = load_XYZI(i)
sin_end = time.time()
print(sin_end - sin_start)
