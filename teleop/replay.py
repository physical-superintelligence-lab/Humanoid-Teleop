import json
import os
import sys
import time
from multiprocessing import Array, Lock, shared_memory

import numpy as np

from robot_control.robot_arm import G1_29_ArmController, H1_2_ArmController
from robot_control.robot_hand_unitree import Dex3_1_Controller

FREQ = 30
DELAY = 1 / FREQ


def main():
    merged_file_path = "/home/yue/UnitreeRobotics/avp_teleoperate/teleop/data/tasks_5_5_g1/locomotion/task1/episode_0/data.json"
    with open(merged_file_path, "r") as f:
        data_list = json.load(f)

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

    if data_list[0]["robot_type"] == "g1":
        h1arm = G1_29_ArmController()
    elif data_list[0]["robot_type"] == "h1":
        h1arm = H1_2_ArmController()
    h1arm.set_weight_to_1()
    h1hand = Dex3_1_Controller(
        hand_shm_array,
        dual_hand_data_lock,
        dual_hand_state_array,
        dual_hand_action_array,
        hand_target_array=hand_target_array,  # Pass the shared memory for hand targets
    )

    time.sleep(1)

    last_time = None
    for i in range(0, len(data_list) - 1, 1):
        print(f"Replay step {i}")

        q_poseList = data_list[i]["states"]["arm_state"]

        with dual_hand_data_lock:
            hand_shm_array[:] = data_list[i]["states"]["hand_state"]

        q_tau_ff = (
            data_list[i]["actions"]["tau_ff"]
            if data_list[i]["actions"] is not None
            else np.zeros(14)
        )

        if last_time is not None:
            pass_time = time.time() - last_time
            time.sleep(max(0, time_interval - pass_time))

        h1arm.ctrl_dual_arm(q_poseList, q_tau_ff)

        if i != len(data_list) - 1:
            time_interval = data_list[i + 1]["time"] - data_list[i]["time"]
            last_time = time.time()

    print("Replay Complete!")
    
    hand_shm.close()
    hand_shm.unlink()
    hand_target_shm.close()
    hand_target_shm.unlink()
    h1arm.gradually_set_weight_to_0()
    h1arm.shutdown()

if __name__ == "__main__":
    main()
