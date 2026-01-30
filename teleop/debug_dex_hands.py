import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_._SportModeState_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from robot_control.remote_controller import RemoteController

from robot_control.robot_arm_joints import (
    G1_29_JointArmIndex,
    G1_29_JointLowerIndex,
    G1_29_BodyIndex,
    H1_2_JointArmIndex,
    H1_2_JointIndex,
)
import os
import sys
import json
import threading
import time
import traceback
from collections import deque
from multiprocessing import (Array, Event, Lock, Manager, Process, Queue,
                             shared_memory)
import csv
import mujoco
import numpy as np
import torch
from lidar import LidarProcess
from merger import DataMerger
from robot_control.robot_body import G1_29_BodyController
from robot_control.robot_body_ik import G1_29_BodyIK
from robot_control.robot_hand_inspire import Inspire_Controller
from robot_control.robot_hand_unitree import Dex3_1_Controller
from utils.logger import logger
from writers import IKDataWriter
from robot_control.compute_tau import GetTauer

from scipy.spatial.transform import Rotation


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



FREQ = 30
DELAY = 1 / FREQ


# dual_hand_data_lock = Lock()
# dual_hand_state_array = Array(
#     "d", 14, lock=False
# )  # [output] current left, right hand state(14) data.
# dual_hand_action_array = Array(
#     "d", 14, lock=False
# )  # [output] current left, right hand action(14) data.
# hand_shm = shared_memory.SharedMemory(
#     create=True, size=14 * np.dtype(np.float64).itemsize
# )
# hand_shm_array = np.ndarray(
#     (14,), dtype=np.float64, buffer=hand_shm.buf
# )

# hand_ctrl = Dex3_1_Controller(
#     hand_shm_array,
#     dual_hand_data_lock,
#     dual_hand_state_array,
#     dual_hand_action_array,
# )
kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
if __name__ == "__main__":
    merged_file_path = "/home/xiawei/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop/data/g1_1001/Basic/Pick_bottle_and_turn_and_pour_into_cup/episode_1/data.json"
    with open(merged_file_path, "r") as f:
        ChannelFactoryInitialize(0)
        # body_ctrl = G1_29_BodyController()
        # lowcmd_publisher = ChannelPublisher(kTopicLowCommand, LowCmd_)
        # lowcmd_publisher.Init()
        # lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
        # lowstate_subscriber.Init()
        # def get_mode_machine(self):
        #     """Return current dds mode machine."""
        #     return lowstate_subscriber.Read().mode_machine
        # crc = CRC()
        # msg = unitree_hg_msg_dds__LowCmd_()
        # msg.mode_pr = 0
        # msg.mode_machine = get_mode_machine()
        # # msg.mode_machine = get_mode_machine()
        # print("body_ctrl ok!")
        # body_ik = G1_29_BodyIK(Visualization=False)
        data_list = json.load(f)
        print("Using g1 controllers")
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array(
            "d", 14, lock=False
        )  # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array(
            "d", 14, lock=False
        )  # [output] current left, right hand action(14) data.
        hand_shm = shared_memory.SharedMemory(
            create=True, size=14 * np.dtype(np.float64).itemsize
        )
        hand_shm_array = np.ndarray(
            (14,), dtype=np.float64, buffer=hand_shm.buf
        )

        hand_ctrl = Dex3_1_Controller(
            hand_shm_array,
            dual_hand_data_lock,
            dual_hand_state_array,
            dual_hand_action_array,
        )
    time.sleep(3)
    try:
        interval = DELAY
        for i in range(0, len(data_list) - 1, 1):
            hand_poseList = data_list[i]["states"]["hand_state"]
            with dual_hand_data_lock:
                hand_shm_array[:] = hand_poseList
            time.sleep(1/30)
            

        
    except KeyboardInterrupt:
        print("Caught Ctrl+C, exiting gracefully...")
    finally:
        hand_shm.close()
        hand_shm.unlink()