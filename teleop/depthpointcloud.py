import os
import pickle
import time
import zlib

import cv2
import numpy as np
import open3d as o3d
import lzma

fx=389.07278
fy=389.07278
cx=321.61887
cy=238.43630

task_dir = "demos/default_task"
file_path = os.path.join(task_dir, "20250317_153137/depth")



def save_pc_from_lzma():
    depth_folder = "/home/yue/UnitreeRobotics/avp_teleoperate/teleop/data/task4_3/pick and place/pick_up_a_pillow/episode_0/depth"
    output_folder = "saved_org_point_clouds"
    os.makedirs(output_folder, exist_ok=True)

    depth_files = sorted(
        [
            os.path.join(depth_folder, f)
            for f in os.listdir(depth_folder)
            if f.endswith(".lzma")
        ]
    )

    for i, depth_path in enumerate(depth_files):
        if i != 0:
            continue

        print(f"Processing {depth_path} ...")

        with open(depth_path, "rb") as f_compressed:
            compressed_data = f_compressed.read()
        decompressed_data = lzma.decompress(compressed_data)
        depth_image = np.frombuffer(decompressed_data, dtype = np.uint16).reshape((480, 640))


        # depth_image = np.load(depth_path).astype(np.uint16)
        # with open(depth_path, "rb") as f:
        #     compressed_data = f.read()
        #     decompressed_data = zlib.decompress(compressed_data)
        #     depth_image = pickle.loads(decompressed_data)

        # depth_image = depth_image.astype(np.float32)

        print(
            f"Depth Image [{i}] - Min: {np.min(depth_image)}, Max: {np.max(depth_image)}"
        )

        height, width = depth_image.shape
        points = []

        for v in range(height):
            for u in range(width):
                Z = depth_image[v, u]

                if Z > 0:
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    if np.linalg.norm(np.array([X,Y,Z])) <= 1250:
                        points.append([X, Y, Z])

        points = np.array(points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        ply_filename = os.path.join(output_folder, f"frame_{i:04d}.ply")
        o3d.io.write_point_cloud(ply_filename, point_cloud)
        print(f"Point Cloud Saved to: {ply_filename}")

    print("All point cloud files saved.")


def play_pc(file_path):
    ply_folder = file_path
    ply_files = sorted(
        [
            os.path.join(ply_folder, f)
            for f in os.listdir(ply_folder)
            if f.endswith(".ply")
        ]
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_cloud = o3d.geometry.PointCloud()

    for ply_path in ply_files:
        print(f"Loading {ply_path} ...")

        point_cloud = o3d.io.read_point_cloud(ply_path)

        vis.clear_geometries()
        vis.add_geometry(point_cloud)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(1.0 / 30)

    vis.destroy_window()


def show_pc():
    file_path = os.path.join("saved_org_point_clouds", "frame_0000.ply")
    point_cloud = o3d.io.read_point_cloud(file_path)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1000.0, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([point_cloud, coord_frame])


def play_pc_pcd(pcd_folder):
    pcd_files = sorted(
        [
            os.path.join(pcd_folder, f)
            for f in os.listdir(pcd_folder)
            if f.endswith(".pcd")
        ]
    )

    if not pcd_files:
        print("No PCD Files Found!")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_cloud = o3d.geometry.PointCloud()

    for pcd_path in pcd_files:
        print(f"Loading {pcd_path} ...")

        point_cloud = o3d.io.read_point_cloud(pcd_path)

        vis.clear_geometries()
        vis.add_geometry(point_cloud)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(delay)

    vis.destroy_window()


if __name__ == "__main__":
    save_pc_from_lzma()
    # save_pc()
    # play_pc("saved_point_clouds")
    show_pc()
