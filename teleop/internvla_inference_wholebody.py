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

# DATA_DIR = "data/g1_1001/Basic/pick_dumpling_toy_and_turn_and_walk_and_squat_to_put_on_chair/episode_10"

FREQ_VLA = 30      # InternVLA 请求频率
FREQ_CTRL = 60    # 控制频率 (Hz)
MAX_STEPS = 500

ACTION_REPEAT = max(1, int(round(FREQ_CTRL / FREQ_VLA)))

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
    pred_action_buffer = {"actions": None, "idx": 0}
    pred_action_lock = threading.Lock()

    running = Event()
    running.set()

    sequence_done_event = Event()
    sequence_done_event.set() 

    # -------- 线程1：请求 OpenVLA，写入 buffer --------
    def action_request_thread():
        s = requests.Session()
        for step in range(MAX_STEPS):
            if not running.is_set():
                break

            sequence_done_event.wait()

            try:
                # obs = get_observation_with_gt(step * 16)
                obs = get_observation(camera)
                payload = {
                    "image": obs["image"],
                    "instruction": TASK_INSTRUCTION,
                    "unnorm_key": UNNORM_KEY,
                }
                resp = s.post(URL, json=payload)
                resp.raise_for_status()
                actions = np.array(resp.json()["action"], dtype=float)
                if len(actions.shape) != 2 or actions.shape[1] < 32:
                    print("[VLA] invalid action seq:", actions.shape)
                    continue

                with pred_action_lock:
                    pred_action_buffer["actions"] = actions
                    pred_action_buffer["idx"] = 0
                print(f"[VLA] Got sequence of {len(actions)} actions.")
                sequence_done_event.clear()
            except Exception as e:
                print(f"[VLA] step {step} failed: {e}")
            time.sleep(1.0 / FREQ_VLA)

        print("[VLA] Finished or stopped. Signaling kill_event.")
        kill_event.set()

    # -------- 辅助：根据 action 构造并下发电机命令 --------
    def apply_action_from_buffer(last_pd_target):
        # 1) 每个控制周期都先读取机器人当前状态
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        # 2) 读取当前 action buffer，看看这一 tick 是否有 VLA action 要用
        with pred_action_lock:
            actions = pred_action_buffer["actions"]
            idx = pred_action_buffer["idx"]

            action = None
            have_vla = False

            if actions is not None:
                real_idx = idx // ACTION_REPEAT
                if real_idx < len(actions):
                    # 本 tick 应该使用的 VLA 动作
                    action = actions[real_idx]
                    have_vla = True

                    # index 自增
                    pred_action_buffer["idx"] += 1

                    # 如果整个 sequence 播放完了，下次 allow 下一个 horizon
                    next_real_idx = pred_action_buffer["idx"] // ACTION_REPEAT
                    if next_real_idx >= len(actions):
                        pred_action_buffer["actions"] = None
                        pred_action_buffer["idx"] = 0
                        sequence_done_event.set()
                else:
                    # 安全兜底：已经超过序列长度
                    pred_action_buffer["actions"] = None
                    pred_action_buffer["idx"] = 0
                    sequence_done_event.set()

        # 3) 如果这一 tick 有来自 VLA 的 action，就更新 torso_* / arm / hand 指令
        arm_cmd = None
        hand_cmd = None
        if have_vla:
            if action.shape[0] < 36:
                print("[CTRL] Invalid action shape:", action.shape)
            else:

                vx = action[32]
                vy = action[33]
                vyaw = action[34]
                dyaw = action[35]


                rpyh   = action[28:32]
                arm_cmd = action[14:28]
                hand_cmd = action[:14]

                master.torso_roll   = rpyh[0]
                master.torso_pitch  = rpyh[1]
                master.torso_yaw    = rpyh[2]
                master.torso_height = rpyh[3]

                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.dyaw = dyaw

                master.prev_torso_roll   = master.torso_roll
                master.prev_torso_pitch  = master.torso_pitch
                master.prev_torso_yaw    = master.torso_yaw
                master.prev_torso_height = master.torso_height

                master.prev_vx   = master.vx
                master.prev_vy  = master.vy
                master.prev_vyaw    = master.vyaw
                master.prev_dyaw = master.dyaw

                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

                # print("vx, vy, vyaw, dyaw:", master.vx, master.vy, master.vyaw, master.dyaw)

                # print("action:", action)
        
        if not have_vla:
            master.torso_roll   = master.prev_torso_roll
            master.torso_pitch  = master.prev_torso_pitch
            master.torso_yaw    = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height

            arm_cmd = master.prev_arm
            hand_cmd = master.prev_hand

            master.vx = 0
            master.vy = 0
            master.vyaw = 0
            master.dyaw = 0
            # master.vx = master.prev_vx
            # master.vy = master.prev_vy
            # master.vyaw = master.prev_vyaw
            # master.dyaw = master.prev_dyaw
        
        # print("torso_yaw:", master.torso_yaw)
        # print("torso_height:", master.torso_height)



        # 4) 无论有没有新 action，**都要跑 IK + whole-body control**
        master.get_ik_observation()


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

        # 5) 如果这一 tick 有上肢 command，就覆盖 pd_target 中的上肢部分
        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm

        # 同样，如果这一 tick 有手的 command，就发给 hand
        if hand_cmd is not None:
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        # 6) 每个 90Hz tick 都要下到电机，不管有没有 VLA 新动作
        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

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

        time.sleep(30)
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
