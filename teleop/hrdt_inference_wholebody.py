import argparse
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
import json_numpy
import zmq

from multiprocessing import Array, Event

from master_whole_body import RobotTaskmaster
from robot_control.compute_tau import GetTauer

json_numpy.patch()


class RSCamera:
    def __init__(self, endpoint: str):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(endpoint)

    def get_frame(self) -> np.ndarray:
        self.socket.send(b"get_frame")
        rgb_bytes, _, _ = self.socket.recv_multipart()
        rgb_array = np.frombuffer(rgb_bytes, np.uint8)
        return cv2.imdecode(rgb_array, cv2.IMREAD_COLOR)

def get_observation_with_gt(data_dir: str, idx: int) -> tuple[dict, dict]:
    episode_dir = f"{data_dir}/episode_{idx:02d}"
    img_name = f"{episode_dir}/color/frame_{idx:06d}.jpg"
    frame = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {img_name}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)
    state_obs = {
        "arm_joints": np.zeros(14, dtype=np.float32),
        "hand_joints": np.zeros(14, dtype=np.float32),
        "prev_height": 0.0,
        "prev_rpy": np.zeros(3, dtype=np.float32),
        "prev_vx": 0.0,
        "prev_vy": 0.0,
        "prev_vyaw": 0.0,
        "prev_dyaw": 0.0,
    }
    return {"egocentric": img}, state_obs


def get_observation(camera: RSCamera, state: dict) -> tuple[dict, dict]:
    frame = camera.get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    img_obs = {"egocentric": img}
    state_obs = {
        "arm_joints": np.array(state["arm_joints"], dtype=np.float32),
        "hand_joints": np.array(state["hand_joints"], dtype=np.float32),
        "prev_height": float(state["prev_height"]),
        "prev_rpy": np.array(state["prev_rpy"], dtype=np.float32),
        "prev_vx": float(state["prev_vx"]),
        "prev_vy": float(state["prev_vy"]),
        "prev_vyaw": float(state["prev_vyaw"]),
        "prev_dyaw": float(state["prev_dyaw"]),
    }
    return img_obs, state_obs


def build_request(
    image_obs: dict,
    state_obs: dict,
    instruction: str,
) -> dict:
    return {
        "image": image_obs,
        "instruction": instruction,
        "history": {},
        "state": state_obs,
        "condition": {},
        "gt_action": None,
        "dataset_name": "realworld",
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HRDT real-world whole-body client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--camera-endpoint", default="tcp://192.168.123.164:5556")
    parser.add_argument("--instruction", default="Pick a dumpling toy and turn right, then walk towards the chair and place the toy on the chair.")
    parser.add_argument("--freq-vla", type=int, default=30)
    parser.add_argument("--freq-ctrl", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--use-gt", action="store_true", help="Use dataset observations instead of live camera.")
    parser.add_argument("--gt-dir", default=None, help="Path to a dataset folder with color/frame_*.jpg.")
    parser.add_argument("--gt-index", type=int, default=0)
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/act"
    freq_vla = args.freq_vla
    freq_ctrl = args.freq_ctrl
    action_repeat = max(1, int(round(freq_ctrl / freq_vla)))

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
    camera = None if args.use_gt else RSCamera(args.camera_endpoint)

    pred_action_buffer = {"actions": None, "idx": 0}
    pred_action_lock = threading.Lock()

    state_lock = threading.Lock()
    shared_robot_state = {"motor": None, "hand": None}

    running = Event()
    running.set()

    sequence_done_event = Event()
    sequence_done_event.set()

    def action_request_thread() -> None:
        session = requests.Session()
        for step in range(args.max_steps):
            if not running.is_set():
                break

            sequence_done_event.wait()

            try:
                with state_lock:
                    motor = shared_robot_state["motor"].copy() if shared_robot_state["motor"] is not None else None
                    hand = shared_robot_state["hand"].copy() if shared_robot_state["hand"] is not None else None

                if motor is None or hand is None:
                    print("[HRDT] Waiting for robot state...")
                    time.sleep(0.01)
                    continue

                arm_joints = motor[15:29]
                hand_joints = hand

                state = {
                    "arm_joints": arm_joints,
                    "hand_joints": hand_joints,
                    "prev_height": master.prev_torso_height,
                    "prev_rpy": np.array(
                        [master.prev_torso_roll, master.prev_torso_pitch, master.prev_torso_yaw],
                        dtype=np.float32,
                    ),
                    "prev_vx": master.prev_vx,
                    "prev_vy": master.prev_vy,
                    "prev_vyaw": master.prev_vyaw,
                    "prev_dyaw": master.prev_dyaw,
                }
                if args.use_gt:
                    if args.gt_dir is None:
                        raise ValueError("--gt-dir is required when --use-gt is set.")
                    img_obs, state_obs = get_observation_with_gt(args.gt_dir, args.gt_index)
                else:
                    img_obs, state_obs = get_observation(camera, state)
                request = build_request(img_obs, state_obs, args.instruction)

                resp = session.post(url, json=request, timeout=120.0)
                resp.raise_for_status()
                payload = resp.json()
                actions = np.asarray(payload["action"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != 36:
                    print(f"[HRDT] Invalid action shape: {actions.shape}")
                    continue

                with pred_action_lock:
                    pred_action_buffer["actions"] = actions
                    pred_action_buffer["idx"] = 0
                print(f"[HRDT] Got sequence of {len(actions)} actions.")
                sequence_done_event.clear()
            except Exception as exc:
                print(f"[HRDT] step {step} failed: {exc}")
            time.sleep(1.0 / freq_vla)

        print("[HRDT] Finished or stopped. Signaling kill_event.")
        kill_event.set()

    def apply_action_from_buffer(last_pd_target):
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        with state_lock:
            shared_robot_state["motor"] = master.motorstate.copy()
            shared_robot_state["hand"] = master.handstate.copy()

        with pred_action_lock:
            actions = pred_action_buffer["actions"]
            idx = pred_action_buffer["idx"]

            action = None
            have_action = False

            if actions is not None:
                real_idx = idx // action_repeat
                if real_idx < len(actions):
                    action = actions[real_idx]
                    have_action = True
                    pred_action_buffer["idx"] += 1

                    next_real_idx = pred_action_buffer["idx"] // action_repeat
                    if next_real_idx >= len(actions):
                        pred_action_buffer["actions"] = None
                        pred_action_buffer["idx"] = 0
                        sequence_done_event.set()
                else:
                    pred_action_buffer["actions"] = None
                    pred_action_buffer["idx"] = 0
                    sequence_done_event.set()

        arm_cmd = None
        hand_cmd = None
        if have_action:
            if action.shape[0] < 36:
                print("[CTRL] Invalid action shape:", action.shape)
            else:
                vx = action[0]
                vy = action[1]
                vyaw = action[2]
                dyaw = action[3]

                # vx = 0.35 if vx > 0.25 else 0
                # vy = 0 if abs(vy) < 0.3 else 0.5 * (1 if vy > 0 else -1)

                vx = 0
                vy = 0
                vyaw = 0


                rpyh = action[4:8]
                arm_cmd = action[8:22]
                hand_cmd = action[22:36]

                master.torso_roll = rpyh[1]
                master.torso_pitch = rpyh[2]
                master.torso_yaw = rpyh[3]
                master.torso_height = rpyh[0]
                print("height rpy:", master.torso_height, master.torso_roll, master.torso_pitch, master.torso_yaw)

                master.vx = vx
                master.vy = vy
                master.vyaw = vyaw
                master.dyaw = dyaw

                master.prev_torso_roll = master.torso_roll
                master.prev_torso_pitch = master.torso_pitch
                master.prev_torso_yaw = master.torso_yaw
                master.prev_torso_height = master.torso_height

                master.prev_vx = master.vx
                master.prev_vy = master.vy
                master.prev_vyaw = master.vyaw
                master.prev_dyaw = master.dyaw

                master.prev_arm = arm_cmd
                master.prev_hand = hand_cmd

        if not have_action:
            master.torso_roll = master.prev_torso_roll
            master.torso_pitch = master.prev_torso_pitch
            master.torso_yaw = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height
            print("height rpy:", master.torso_height, master.torso_roll, master.torso_pitch, master.torso_yaw)
            arm_cmd = master.prev_arm
            hand_cmd = master.prev_hand

            master.vx = master.prev_vx
            master.vy = 0
            master.vyaw = master.prev_vyaw
            master.dyaw = master.prev_dyaw

        master.get_ik_observation(record=False)

        pd_target, pd_tauff, raw_action = master.body_ik.solve_whole_body_ik(
            left_wrist=None,
            right_wrist=None,
            current_lr_arm_q=current_lr_arm_q,
            current_lr_arm_dq=current_lr_arm_dq,
            observation=master.observation,
            extra_hist=master.extra_hist,
            is_teleop=False,
        )

        master.last_action = np.concatenate(
            [raw_action.copy(), (master.motorstate - master.default_dof_pos)[15:] / master.action_scale]
        )

        if arm_cmd is not None:
            pd_target[15:] = arm_cmd
            tau_arm = np.asarray(get_tauer(arm_cmd), dtype=np.float64).reshape(-1)
            pd_tauff[15:] = tau_arm

        if hand_cmd is not None:
            with master.dual_hand_data_lock:
                master.hand_shm_array[:] = hand_cmd

        master.body_ctrl.ctrl_whole_body(
            pd_target[15:], pd_tauff[15:], pd_target[:15], pd_tauff[:15]
        )

        return pd_target

    def control_loop_thread() -> None:
        dt = 1.0 / freq_ctrl
        last_pd_target = None
        while running.is_set() and not kill_event.is_set():
            try:
                last_pd_target = apply_action_from_buffer(last_pd_target)
            except Exception as exc:
                print("[CTRL] loop error:", exc)
            time.sleep(dt)
        print("[CTRL] Control loop stopped.")

    try:
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")

        time.sleep(30)
        master.episode_kill_event.clear()
        master.reset_yaw_offset = True

        t_req = threading.Thread(target=action_request_thread, daemon=True)
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_req.start()
        t_ctrl.start()

        print("[MAIN] Running. Ctrl+C to stop.")
        while not kill_event.is_set():
            time.sleep(0.5)

        print("[MAIN] kill_event set, preparing to stop...")
        running.clear()
        time.sleep(0.5)

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
