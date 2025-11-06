import requests
import json_numpy
import numpy as np
import os
import json
import cv2
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_hand_unitree import Dex3_1_Controller
from robot_control.compute_tau import GetTauer
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy
from multiprocessing import Array, Event, Lock, shared_memory, Manager
from master_whole_body import RobotTaskmaster
import threading
import zmq
import time

FREQ = 3
DELAY = 1 / FREQ

task_instruction = "Walk towards the purple front door and then stop to grab the black handle."

DATA_DIR = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_39"


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


if __name__ == "__main__":
    # url = "http://10.128.0.72:8014/act"
    url = "http://localhost:8080/act"
    unnorm_key = "humanoid_dataset/Grab_handle"
    # patch json-numpy
    json_numpy.patch()

    max_timesteps=500

    camera = RSCamera()
    get_tauer = GetTauer()

    merged_file_path = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_10/data.json"

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
        task_name="inference",
        shared_data=shared_data,
        robot_shm_array=robot_shm_array,
        teleop_shm_array=teleop_shm_array,
        robot="g1"
    )

    get_tauer = GetTauer()

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
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("Initialize with standing pose...")
        time.sleep(10) 
        master.episode_kill_event.clear()
        last_pd_target = None

        for step in range(max_timesteps):
            print(f"Inference step {step}...")
            # time.sleep(1)
            # obs = get_observation(camera)
            obs = get_observation_with_gt(step*10)
            payload = {
            "image": obs["image"],
            "instruction": task_instruction,
            "unnorm_key": unnorm_key,
            }
            resp = requests.post(url, json=payload)
            # print("resp:", resp.json())
            pred_action = np.array(resp.json()["action"], dtype=float)
            pred_action = pred_action[0]
            arm_poseList = pred_action[:14]
            hand_poseList = pred_action[14:28]
            current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()
            master.torso_roll = pred_action[28]
            master.torso_pitch = pred_action[29]
            master.torso_yaw = pred_action[30]
            master.torso_height = pred_action[31]

            print("predicted torso r, p, y, h:", pred_action[28], pred_action[29], pred_action[30], pred_action[31])

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

            master.body_ctrl.ctrl_whole_body(pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15])
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
        master.stop()
        master.hand_shm.close()
        master.hand_shm.unlink()