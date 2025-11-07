import json
import os
import threading
import time

# from openpi_client import image_tools
# from openpi_client import websocket_client_policy
from multiprocessing import Array, Event, Lock, Manager, shared_memory

import cv2
import json_numpy
import numpy as np
import requests
import zmq
from master_whole_body import RobotTaskmaster
from robot_control.compute_tau import GetTauer
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import Dex3_1_Controller

FREQ = 3
DELAY = 1 / FREQ

task_instruction = (
    "Walk towards the purple front door and then stop to grab the black handle."
)

DATA_DIR = (
    "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_39"
)


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


# def get_observation(g1arm, g1hand,camera):
#     frame = camera.get_frame()
#     frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = frame.astype(np.uint8)

#     arm_q = g1arm.get_current_motor_q()
#     hand_q = g1hand.get_current_dual_hand_q()
#     qpos = np.concatenate([arm_q[15:29], hand_q])
#     obs = {
#         "image": frame,
#         "qpos": qpos,
#     }
#     return obs


def get_observation(camera):
    frame = camera.get_frame()
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    obs = {
        "image": frame,
    }
    return obs


def get_observation_with_gt(idx):
    img_name = os.path.join("color", f"frame_{idx:06d}.jpg")
    path = os.path.join(DATA_DIR, img_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    frame = cv2.imread(path, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    # frame = camera.get_frame()
    # frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = frame.astype(np.uint8)

    obs = {
        "image": img,
    }
    return obs


def action_request():
    global pred_action_buffer
    for step in range(max_timesteps):
        obs = get_observation_with_gt(step * 10)
        payload = {
            "image": obs["image"],
            "instruction": task_instruction,
            "unnorm_key": unnorm_key,
        }
        resp = requests.post(url, json=payload)
        with pred_action_lock:
            pred_action_buffer = np.array(resp.json()["action"], dtype=float)[0]
    kill_event.set()


def control_loop():
    global pred_action_buffer, kill_event
    while not kill_event.is_set():
        with pred_action_lock:
            action = None if pred_action_buffer is None else pred_action_buffer.copy()
        if action is not None:
            master.ctrl_whole_body(action)

        time.sleep(DELAY)


if __name__ == "__main__":
    pred_action_buffer = None
    pred_action_lock = threading.Lock()
    # url = "http://10.128.0.72:8014/act"
    url = "http://localhost:8080/act"
    unnorm_key = "humanoid_dataset/Grab_handle"
    # patch json-numpy
    json_numpy.patch()

    max_timesteps = 500

    camera = RSCamera()
    get_tauer = GetTauer()

    merged_file_path = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_10/data.json"

    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": "/home/replay",
    }
    robot_shm_array = Array("d", 512, lock=False)
    teleop_shm_array = Array("d", 64, lock=False)

    master = RobotTaskmaster(
        task_name="inference",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1",
    )

    get_tauer = GetTauer()
    kill_event = shared_data["kill_event"]

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
        stabilize_thread = threading.Thread(
            target=master.maintain_standing, daemon=True
        )
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("Initialize with standing pose...")
        time.sleep(10)
        master.episode_kill_event.clear()
        last_pd_target = None

        request_thread = threading.Thread(target=action_request, daemon=False)
        control_thread = threading.Thread(target=control_loop, daemon=False)

        request_thread.start()
        control_thread.start()

        request_thread.join(timeout=5.0)
        kill_event.set()
        control_thread.join(timeout=5.0)

        # ok = master.safelySetMotor(pd_target, last_pd_target, pd_tauff)
        # if ok:
        #     last_pd_target = pd_target
        # else:
        #     continue

        # time.sleep(DELAY*10)

        print("Replay finished. Returning to standing pose...")
        master.episode_kill_event.set()
        time.sleep(10)

    except KeyboardInterrupt:
        print("Caught Ctrl+C, exiting gracefully...")
    finally:
        request_thread.join(timeout=5.0)
        kill_event.set()
        control_thread.join(timeout=5.0)

        master.stop()
        master.hand_shm.close()
        master.hand_shm.unlink()

