import json
import os
import sys

from utils.logger import logger

FREQ = 30
DELAY = 1 / FREQ


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class DataMerger:
    def __init__(self, dirname) -> None:
        self.robot_data_path = os.path.join(dirname, "robot_data.jsonl")
        self.ik_data_path = os.path.join(dirname, "ik_data.jsonl")
        self.lidar_data_path = os.path.join(dirname, "lidar")
        self.output_path = os.path.join(dirname, "data.json")

    def _ik_is_ready(self, ik_data_list, time_key):
        closest_ik_entry = min(ik_data_list, key=lambda x: abs(x["armtime"] - time_key))
        if abs(closest_ik_entry["armtime"] - time_key) > DELAY / 2:
            return False, None
        return True, closest_ik_entry

    def _lidar_is_ready(self, lidar_time_list, time_key):
        closest_lidar_entry = min(lidar_time_list, key=lambda x: abs(x - time_key))
        if abs(closest_lidar_entry - time_key) > DELAY / 2:
            return False, None
        return True, closest_lidar_entry

    def merge_json(self):
        lidar_time_list = []

        lidar_files = [
            f
            for f in os.listdir(self.lidar_data_path)
            if os.path.isfile(os.path.join(self.lidar_data_path, f))
        ]

        for lidar_file_name in lidar_files:
            time_parts = lidar_file_name.split(".")[0:2]
            lidar_time_list.append(float(time_parts[0] + "." + time_parts[1]))

        logger.info("loading robot and IK data for merging.")
        robot_data_json_list = []
        with open(self.robot_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    robot_data_json_list.append(json.loads(line))

        ik_data_list = []
        with open(self.ik_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    ik_data_list.append(json.loads(line))

        ik_data_dict = {entry["armtime"]: entry for entry in ik_data_list}
        robot_data_dict = {entry["time"]: entry for entry in robot_data_json_list}

        if ik_data_list[0]["armtime"] > robot_data_json_list[0]["time"]:
            last_robot_data = None
        else:
            last_robot_data = ik_data_list[0]

        for motor_entry in robot_data_json_list:
            time_key = motor_entry["time"]
            ik_ready_flag, closest_ik_entry = self._ik_is_ready(ik_data_list, time_key)
            if ik_ready_flag and closest_ik_entry is not None:
                robot_data_dict[time_key]["actions"] = ik_data_dict[
                    closest_ik_entry["armtime"]
                ]
                last_robot_data = robot_data_dict[time_key]["actions"]
            else:
                robot_data_dict[time_key]["actions"] = last_robot_data

            # merge lidar path
            lidar_ready_flag, closest_lidar_time = self._lidar_is_ready(
                lidar_time_list, time_key
            )
            if lidar_ready_flag:
                robot_data_dict[time_key]["lidar"] = os.path.join(
                    "lidar", f"{closest_lidar_time}.pcd"
                )

        with open(self.output_path, "w") as f:
            json.dump(robot_data_json_list, f, indent=4)

        logger.info(f"Mergefile saved to {self.output_path}")
