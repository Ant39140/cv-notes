# import dataset
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tool, cv2, os, tqdm

MEM_BOUND = 100 * 2**20  # 限制数据内存占用为100MB
BYTE2M = 2**20
DS_STEP = 0.8

ROOT = os.path.dirname(__file__)

def RGBD2cloud(image:np.ndarray, depth:np.ndarray, F:np.ndarray, camera_frame:dict) -> tuple:
    """
    图片转世界坐标系下的点云
    image: BGR图(H*W*3)
    depth: 深度图(H*W)，传入前已经scale过了，像素值就是深度
    F: 内参(3*3)
    camera_frame: 相机坐标系位姿字典，键"x_axis"、"y_axis"、"z_axis"分别是相机坐标系坐标轴在世界坐标系下的单位向量，
    "T"则是相机坐标系原点在世界坐标系下的坐标。以上向量都是shape为(3)的np.ndarray
    return:
    points: 点的世界坐标(N*3)，其中N=H*W，列对应着xyz
    colors: 点的颜色(N*3)，其中N=H*W，列对应着RGB
    """
    #  先由像素坐标求出归一化坐标
    F_inv = np.linalg.inv(F)
    H, W = image.shape[:2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    uv_matrix = np.vstack((u.flatten(), v.flatten(), np.ones(H * W)))
    camera_normalized_position = (F_inv @ uv_matrix)

    #  利用矩阵的逐元素运算把归一化坐标转为相机坐标
    z_values = depth.flatten()
    camera_position = camera_normalized_position * z_values

    #  用相机坐标系坐标轴在世界坐标系下的单位向量合并成旋转矩阵
    R_carema2world = np.column_stack((camera_frame["x_axis"],
                                       camera_frame["y_axis"],
                                       camera_frame["z_axis"])
                                     )
    #  把旋转矩阵和平移矩阵合并写成4*4的矩阵
    T_camera2world = np.eye(4)
    T_camera2world[:3, :3] = R_carema2world
    T_camera2world[:3, 3] = camera_frame["T"]

    #  把相机坐标构成的矩阵也拓展成四维，从而能够进行矩阵乘法
    camera_4D_position = np.ones((4, H * W))
    camera_4D_position[:3, :] = camera_position

    #  把相机坐标转换为世界坐标
    world_4D_position = T_camera2world @ camera_4D_position

    #  对世界坐标构成的矩阵取前三行然后转置，以符合输出格式
    points = (world_4D_position[:3, :]).T

    RGB_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colors = RGB_im.reshape(-1, 3)

    return points, colors

if __name__ == "__main__":    
    data_path = os.path.join(ROOT, 'data')
    data_names = os.listdir(data_path)
    data_num = int(len(data_names) / 2)

    # 加载内参
    camera_param = tool.read_config_to_dict(os.path.join(ROOT, 'camera.txt'))
    F = np.array([[camera_param['fx'], 0, camera_param['cx']],
                  [0, camera_param['fy'], camera_param['cy']],
                  [0, 0, 1]])
    png_depth_scale = camera_param['png_depth_scale']

    # 加载相机位姿
    poses = []
    pose_path = os.path.join(ROOT, 'traj.txt')
    poses = tool.load_camera_frame(pose_path, data_num)

    # 加载第一组数据
    print(os.path.join(data_path, 'frame000000.jpg'))
    im_1st = cv2.imread(os.path.join(data_path, 'frame000000.jpg'))
    depth_1st = cv2.imread(os.path.join(data_path, 'depth000000.png'), cv2.IMREAD_UNCHANGED)
    depth_1st = depth_1st / png_depth_scale  # 看起来单位是m
    points, colors = RGBD2cloud(im_1st, depth_1st, F, poses[0])
    points = [points]
    colors = [colors]
    total_size = points[0].nbytes + colors[0].nbytes
    down_rate_acc = 1  # 目前累计的降采样率

    for i in tqdm.tqdm(range(data_num), desc=f'montage'):
        # 如果觉得所有图片都处理太慢了不方便调试可以去掉一下两行的注释
        # if i > 100:
        #     break

        now_im = cv2.imread(os.path.join(data_path, 'frame' + str(i * 10).zfill(6) + '.jpg'))
        now_depth = cv2.imread(os.path.join(data_path, 'depth' + str(i * 10).zfill(6) + '.png'), cv2.IMREAD_UNCHANGED)
        now_depth = now_depth / png_depth_scale
        now_points, now_color = RGBD2cloud(now_im, now_depth, F, poses[i])
        # 这里需要保证每帧降采样率一样
        now_points, now_color = tool.ds_point_ratio(now_points, down_rate_acc, now_color)
        points.append(now_points)
        colors.append(now_color)

        # 防止爆内存，限制点云大小
        total_size = sum([pt.nbytes for pt in points]) + sum([cl.nbytes for cl in colors])
        if total_size > MEM_BOUND:
            points, colors = tool.ds_point_ratio(points, DS_STEP, colors)
            down_rate_acc *= DS_STEP

    points = np.vstack(points)
    colors = np.vstack(colors)
    tool.visualize_with_color(points, colors)
