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
from master_whole_body import RobotTaskmaster
from multiprocessing import Array, Event, Lock, shared_memory, Manager
import threading


FREQ = 30
DELAY = 1 / FREQ

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


if __name__ == "__main__":
    merged_file_path = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_10/data.json"
    with open(merged_file_path, "r") as f:
        data_list = json.load(f)

    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": "/home/replay"
    }
    robot_shm_array = Array("d", 512, lock=False)
    teleop_shm_array = Array("d", 64, lock=False)

    master = RobotTaskmaster(
        task_name="replay",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1"
    )

    get_tauer = GetTauer()

    try:
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("Initialize with standing pose...")
        time.sleep(20) 
        master.episode_kill_event.clear()
        last_pd_target = None

        for i in range(0, len(data_list) - 1):
            arm_poseList = data_list[i]["states"]["arm_state"]
            current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()
            # rpy = data_list[i]["states"]["imu"]["rpy"]
            # quat = data_list[i]["states"]["imu"]["quaternion"]
            # rpy = quatToEuler(quat)
            rpy = data_list[i]["actions"]["torso_rpy"]
            # master.torso_height = data_list[i]["states"]["odometry"]["position"][2]
            master.torso_height = data_list[i]["actions"]["torso_height"]
            master.torso_roll = rpy[0]
            master.torso_pitch = rpy[1]
            master.torso_yaw = rpy[2]

            hand_poseList = data_list[i]["states"]["hand_state"]

            master.get_ik_observation()
            pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
                left_wrist=None,
                right_wrist=None,
                current_lr_arm_q=current_lr_arm_q,
                current_lr_arm_dq=current_lr_arm_dq,
                observation=master.observation,
                extra_hist=master.extra_hist,
                is_teleop=False
            )
            master.last_action = np.concatenate([raw_action.copy(), (master.motorstate - master.default_dof_pos)[15:] / master.action_scale])

            pd_target[15:] = arm_poseList
            pd_tauff[15:] = get_tauer(np.array(arm_poseList))

            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_poseList

            ok = master.safelySetMotor(pd_target, last_pd_target, pd_tauff)
            if ok:
                last_pd_target = pd_target
            else:
                continue

            time.sleep(DELAY)
        
        print("Replay finished. Returning to standing pose...")
        master.episode_kill_event.set()
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("Caught Ctrl+C, exiting gracefully...")
    
    finally:
        master.stop()
        master.hand_shm.close()
        master.hand_shm.unlink()
        

