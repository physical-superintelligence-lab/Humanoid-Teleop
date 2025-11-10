import os
import time
import threading
import json

import cv2
import numpy as np
import requests
import json_numpy

from multiprocessing import Array, Event
from master_whole_body import RobotTaskmaster
from robot_control.compute_tau import GetTauer
import zmq

# ---------------- 配置 ----------------
URL = "http://localhost:8014/act"  # 或 8080
UNNORM_KEY = "humanoid_dataset/Grab_handle"
TASK_INSTRUCTION = "Walk towards the purple front door and then stop to grab the black handle."

DATA_DIR = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_10"

FREQ_VLA = 3      # OpenVLA 请求频率
FREQ_CTRL = 100    # 控制频率 (Hz)
MAX_STEPS = 500

json_numpy.patch()


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


# ---------------- 工具函数 ----------------
def get_observation_with_gt(idx):
    img_name = os.path.join(DATA_DIR, "color", f"frame_{idx:06d}.jpg")
    if not os.path.exists(img_name):
        raise FileNotFoundError(f"Image not found: {img_name}")
    frame = cv2.imread(img_name, cv2.IMREAD_COLOR)
    # frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)
    return {"image": img}


def get_observation(camera):
    frame = camera.get_frame()
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    obs = {
        "image": frame,
    }
    return obs


# ---------------- 主逻辑 ----------------
def main():
    # 共享事件 & shm
    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": "/home/replay",
    }
    kill_event = shared_data["kill_event"]

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
    camera = RSCamera()

    # 共享 buffer：VLA 写入，控制 loop 读取
    pred_action_buffer = {"action": None}
    pred_action_lock = threading.Lock()

    running = Event()
    running.set()

    # -------- 线程1：请求 OpenVLA，写入 buffer --------
    def action_request_thread():
        s = requests.Session()
        for step in range(MAX_STEPS):
            if not running.is_set():
                break
            try:
                # obs = get_observation_with_gt(step * 10)
                obs = get_observation(camera)
                payload = {
                    "image": obs["image"],
                    "instruction": TASK_INSTRUCTION,
                    "unnorm_key": UNNORM_KEY,
                }
                resp = s.post(URL, json=payload)
                resp.raise_for_status()
                action = np.array(resp.json()["action"], dtype=float)[0]
                with pred_action_lock:
                    pred_action_buffer["action"] = action
                print(f"[VLA] step {step}, got action.")
            except Exception as e:
                print(f"[VLA] step {step} failed: {e}")
            # time.sleep(1.0 / FREQ_VLA)

        print("[VLA] Finished or stopped. Signaling kill_event.")
        kill_event.set()

    # -------- 辅助：根据 action 构造并下发电机命令 --------
    def apply_action_from_buffer(last_pd_target):
        with pred_action_lock:
            action = pred_action_buffer["action"].copy() if pred_action_buffer["action"] is not None else None

        if action is None:
            return last_pd_target  # 没有新指令就保持

        if action.shape[0] < 32:
            print("[CTRL] Invalid action shape:", action.shape)
            return last_pd_target

        # 解析 action: [0:14]=arm, [14:28]=hand, [28:32]=r,p,y,h
        arm = action[4:18]
        hand = action[18:32]
        rpyh = action[0:4]

        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        # 更新 torso 目标供 observation 使用
        master.torso_roll = rpyh[0]
        master.torso_pitch = rpyh[1]
        master.torso_yaw = rpyh[2]
        master.torso_height = rpyh[3]

        # 读当前机器人状态
        master.get_ik_observation()

        # 下肢 policy
        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )


        master.last_action = np.concatenate([
            raw_action.copy(),
            (master.motorstate - master.default_dof_pos)[15:] / master.action_scale,
        ])

        # 上肢关节来自 VLA
        pd_target[15:] = arm.astype(pd_target.dtype, copy=False)

        tau_arm = np.asarray(get_tauer(arm), dtype=np.float64).reshape(-1)

        pd_tauff[15:] = tau_arm.astype(pd_tauff.dtype, copy=False)

        # 手部
        with master.dual_hand_data_lock:
            master.hand_shm_array[:] = hand

        # ok = master.safelySetMotor(pd_target, last_pd_target, pd_tauff)
        # if not ok:
        #     print("[CTRL] safelySetMotor rejected step.")
        #     return last_pd_target
        master.body_ctrl.ctrl_whole_body(pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15])

        return pd_target

    # -------- 线程2：高频控制 loop --------
    def control_loop_thread():
        dt = 1.0 / FREQ_CTRL
        last_pd_target = None
        while running.is_set() and not kill_event.is_set():
            try:
                last_pd_target = apply_action_from_buffer(last_pd_target)
            except Exception as e:
                print("[CTRL] loop error:", e)
            time.sleep(dt)
        print("[CTRL] Control loop stopped.")

    try:
        # 1. 先站立 20 秒
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")
        time.sleep(40)
        master.episode_kill_event.clear()  # 停止站立控制，只留下面的控制线程写电机

        # 2. 启动双线程
        t_req = threading.Thread(target=action_request_thread, daemon=True)
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_req.start()
        t_ctrl.start()

        print("[MAIN] Running. Ctrl+C to stop.")
        # 主线程等待 kill_event（VLA结束）或 Ctrl+C
        while not kill_event.is_set():
            time.sleep(0.5)

        print("[MAIN] kill_event set, preparing to stop...")
        running.clear()
        time.sleep(0.5)  # 给线程一点时间收尾

        # 3. 可选：回到站立姿态
        master.episode_kill_event.set()
        print("[MAIN] Returning to standing pose for 5s...")
        time.sleep(5)
        master.episode_kill_event.clear()

    except KeyboardInterrupt:
        print("[MAIN] Caught Ctrl+C, exiting...")
        running.clear()
        kill_event.set()
    finally:
        shared_data["end_event"].set()
        master.stop()
        print("[MAIN] Shutdown complete.")

if __name__ == "__main__":
    main()
