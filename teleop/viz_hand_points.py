import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_keypoints(filename, tip_indices):
    """
    加载包含 [index, x, y, z] 数据的 NumPy 文件，并进行 3D 可视化。
    :param filename (str): 要加载的 .npy 文件名。
    :param tip_indices (list): 需要高亮显示的点的索引（Unitree 期望的 [4, 9, 14]）。
    """
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"错误：文件未找到: {filename}")
        return

    if data.shape[1] != 4:
        print(f"错误：数据格式不正确。预期 shape 为 (N, 4)，实际为 {data.shape}")
        return

    # 提取索引和坐标
    indices = data[:, 0]  # 关键点索引
    X = data[:, 1]
    Y = data[:, 2]
    Z = data[:, 3]

    # 区分普通点和高亮的关键点
    normal_indices_mask = np.isin(indices, tip_indices, invert=True)
    X_norm, Y_norm, Z_norm = X[normal_indices_mask], Y[normal_indices_mask], Z[normal_indices_mask]
    
    tip_indices_mask = np.isin(indices, tip_indices)
    X_tip, Y_tip, Z_tip = X[tip_indices_mask], Y[tip_indices_mask], Z[tip_indices_mask]

    # 确定手腕原点坐标 (索引 0)
    wrist_index = 0
    if wrist_index in indices:
        # data[indices == wrist_index, 1:4] 找到索引 0 的 XYZ 坐标
        wrist_coords = data[indices == wrist_index, 1:4].flatten()
    else:
        # 兜底：如果手腕点不在，使用所有点的平均值
        wrist_coords = np.mean(data[:, 1:4], axis=0)

    
    wrist_idx = 0
    thumb_tip_idx = 4

    wrist_row = data[indices == wrist_idx]
    tip_row = data[indices == thumb_tip_idx]

    P0 = wrist_row[0, 1:]
    P4 = tip_row[0, 1:]
    
    distance = np.linalg.norm(P4 - P0)
    print("thumb length:", distance)

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 0. 明确绘制手腕原点 (Wrist Index 0)
    ax.scatter(wrist_coords[0], wrist_coords[1], wrist_coords[2], 
               c='blue', marker='X', s=200, label='Wrist (Origin 0)')

    # 1. 绘制所有普通点 (蓝色或灰色)
    ax.scatter(X_norm, Y_norm, Z_norm, c='gray', marker='o', s=20, label='Normal Joints')

    # 2. 绘制关键指尖点 (红色高亮)
    ax.scatter(X_tip, Y_tip, Z_tip, c='red', marker='o', s=100, label='Unitree Tips (Index: 4, 9, 14)')
    
    # 3. 标记关键轴（帮助理解坐标系）
    axis_len = 0.05 # 坐标轴长度
    
    # 绘制坐标轴：X (红色), Y (绿色), Z (蓝色)
    ax.quiver(wrist_coords[0], wrist_coords[1], wrist_coords[2], 
              axis_len, 0, 0, color='r', length=axis_len, arrow_length_ratio=0.2) # X
    ax.quiver(wrist_coords[0], wrist_coords[1], wrist_coords[2], 
              0, axis_len, 0, color='g', length=axis_len, arrow_length_ratio=0.2) # Y
    ax.quiver(wrist_coords[0], wrist_coords[1], wrist_coords[2], 
              0, 0, axis_len, color='b', length=axis_len, arrow_length_ratio=0.2) # Z
    ax.text(wrist_coords[0] + axis_len, wrist_coords[1], wrist_coords[2], 'X (Red)', color='r')
    ax.text(wrist_coords[0], wrist_coords[1] + axis_len, wrist_coords[2], 'Y (Green)', color='g')
    ax.text(wrist_coords[0], wrist_coords[1], wrist_coords[2] + axis_len, 'Z (Blue)', color='b')
    

    # 4. 标注所有点的索引 (可选：如果点太密可以注释掉)
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
         ax.text(x, y, z, f'{int(indices[i])}', color='black', fontsize=8)


    # 5. 设置图形标题和轴标签
    ax.set_title(f"3D Keypoint Visualization: {filename}\nUnitree Hand (Left, OpenXR Target)")
    ax.set_xlabel('X Axis (Red)')
    ax.set_ylabel('Y Axis (Green)')
    ax.set_zlabel('Z Axis (Blue)')
    
    # 确保坐标轴比例一致，避免手部形状变形
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    
    padding = max_range * 1.5 # 增加额外的填充，确保坐标轴标签和点完全显示
    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)
    
    ax.legend()
    # 调整初始视角，使其更易于查看手部结构
    ax.view_init(elev=20, azim=-120) 
    plt.show()


# --- 使用你的文件进行可视化 ---
filename = 'manus_left_target_points_with_idx_20251212_162116.npy'
unitree_tip_indices = [24, 5, 10] 

# filename = 'openxr_left_target_points_with_idx_20251212_151135.npy'
# unitree_tip_indices = [4, 9, 14] 

visualize_keypoints(filename, unitree_tip_indices)