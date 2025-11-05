import json
import os
import sys
import time
from multiprocessing import Array, Lock, shared_memory

import numpy as np

from robot_control.robot_arm import G1_29_ArmController, H1_2_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK, H1_2_ArmIK
from robot_control.robot_hand_unitree import Dex3_1_Controller
from robot_control.compute_tau import GetTauer



FREQ = 30
DELAY = 1 / FREQ


if __name__ == "__main__":
    merged_file_path = "/home/yue/UnitreeRobotics/avp_teleoperate/teleop/data/tasks_9_7/Experiments/experiments/episode_1/data.json"
    with open(merged_file_path, "r") as f:
        data_list = json.load(f)

    get_tauer = GetTauer()

    g1arm = G1_29_ArmController()
    g1arm.set_weight_to_1()

    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array("d", 14, lock=False)
    dual_hand_action_array = Array("d", 14, lock=False)
    hand_target_shm = shared_memory.SharedMemory(
        create=True, size=14 * np.dtype(np.float64).itemsize
    )
    hand_target_array = np.ndarray((14,), dtype=np.float64, buffer=hand_target_shm.buf)
    hand_target_array[:] = 0  # Initialize with zeros
    hand_shm = shared_memory.SharedMemory(
        create=True, size=14 * np.dtype(np.float64).itemsize
    )
    hand_shm_array = np.ndarray((14,), dtype=np.float64, buffer=hand_shm.buf)
    g1hand = Dex3_1_Controller(
        hand_shm_array,
        dual_hand_data_lock,
        dual_hand_state_array,
        dual_hand_action_array,
        hand_target_array=hand_target_array,
    )
    time.sleep(3)
    try:
        interval = DELAY
        for i in range(0, len(data_list) - 1, 1):
            q_poseList = data_list[i]["actions"]["sol_q"]
            q_tauff = data_list[i]["actions"]["tau_ff"]
            g1arm.ctrl_dual_arm(q_poseList, q_tauff)
            hand_poseList = data_list[i]["states"]["hand_state"]
            with dual_hand_data_lock:
                hand_shm_array[:] = hand_poseList
            time.sleep(interval)
            

        
    except KeyboardInterrupt:
        print("Caught Ctrl+C, exiting gracefully...")
    finally:
        hand_shm.close()
        hand_shm.unlink()
        hand_target_shm.close()
        hand_target_shm.unlink()
        g1arm.gradually_set_weight_to_0()
        g1arm.shutdown()