import requests
import json_numpy
import numpy as np
import os
import json
import cv2
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import Dex3_1_Controller
from robot_control.compute_tau import GetTauer
from multiprocessing import Array, Lock, shared_memory
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy
import zmq
import time

FREQ = 2
DELAY = 1 / FREQ

task_instruction = "Walk towards the purple front door and then stop to grab the black handle."
class RSCamera:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://192.168.123.164:5556")

    def get_frame(self):
        self.socket.send(b"get_frame")

        rgb_bytes, _, _ = self.socket.recv_multipart()

        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        rgb_image = cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)
        return rgb_image

def get_observation(g1arm, g1hand,camera):
    frame = camera.get_frame()
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    arm_q = g1arm.get_current_motor_q()
    hand_q = g1hand.get_current_dual_hand_q()
    qpos = np.concatenate([arm_q[15:29], hand_q])
    obs = {
        "image": frame,
        "qpos": qpos,
    }
    return obs


if __name__ == "__main__":
    url = "http://10.136.20.58:8014/act"
    unnorm_key = "humanoid_dataset/Grab_handle"
    # patch json-numpy
    json_numpy.patch()

    max_timesteps=500

    camera = RSCamera()
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
    try:
        # # initialize arm in the camera for Use eraser...
        # arm_state = np.array([0.375, 0.592, 0.113, -0.486, -0.481, -0.380, -0.510, -0.077, -0.171, -0.093, 1.214, -0.105, -0.619, -0.195])
        # hand_state = np.array([-0.149, 0.663, 0.608, -0.436, -0.661, -0.495, -0.432, -0.300, -0.499, -0.631, 0.691, 0.454, 0.561, 0.413])
        # initialize arm in the camera for Fold towel
        # arm_state = np.array([-0.349, 0.928, 0.140, -0.147, -0.122, 0.077, -0.408, -0.111, -0.261, -0.241, 1.219, -0.008, -0.318, -0.134])
        # hand_state = np.array([-0.303, 0.291, 0.811, -0.067, -0.446, -0.046, -0.441, -0.150, -0.647, -0.611, 0.636, 0.491, 0.506, 0.443])
        # with dual_hand_data_lock:
        #     hand_shm_array[:] = hand_state
        # q_tau_ff = get_tauer(arm_state)
        # g1arm.ctrl_dual_arm(arm_state, q_tau_ff)
        # print("Finished Initializing, now wait 5 seconds for inference start...")
        # time.sleep(5)
        for step in range(max_timesteps):
            print(f"Inference step {step}...")
            # time.sleep(1)
            obs = get_observation(g1arm, g1hand, camera)
            payload = {
            "image": obs["image"],
            "instruction": task_instruction,
            "unnorm_key": unnorm_key,
            }
            resp = requests.post(url, json=payload)
            pred_action = np.array(resp.json(), dtype=float)
            with dual_hand_data_lock:
                hand_shm_array[:] = pred_action[14:28]
            q_poseList = pred_action[:14]
            q_tau_ff = get_tauer(q_poseList)
            g1arm.ctrl_dual_arm(q_poseList, q_tau_ff)
            time.sleep(DELAY)
    except KeyboardInterrupt:
        print("Caught Ctrl+C, exiting gracefully...")
    finally:
        hand_shm.close()
        hand_shm.unlink()
        hand_target_shm.close()
        hand_target_shm.unlink()
        g1arm.gradually_set_weight_to_0()
        g1arm.shutdown()