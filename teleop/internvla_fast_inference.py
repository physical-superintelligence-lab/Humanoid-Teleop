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
from websocket import WebSocketApp

# ---------------- 配置 ----------------
UNNORM_KEY = "humanoid_dataset/Grab_handle"
TASK_INSTRUCTION = "Walk towards the purple front door and then stop to grab the black handle."

DATA_DIR = "data/g1_1001/Basic/squat_to_pick_a_box_and_stand_to_put_on_desk/episode_10"

FREQ_CTRL = 60    # 控制频率 (Hz)
OBS_SEND_INTERVAL = 0.01  # 发送间隔约200Hz

json_numpy.patch()

from base64 import b64encode, b64decode
from numpy.lib.format import dtype_to_descr, descr_to_dtype

# 共享事件 & shm
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


def get_observation(camera):
    frame = camera.get_frame()
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame.astype(np.uint8)

    obs = {
        "image": frame,
    }
    return obs


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
pred_action_buffer = {"actions": None}
pred_action_lock = threading.Lock()

running = Event()
running.set()


# ============ Serialization utilities ============
def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(dct["shape"]) if dct["shape"] else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


# ============ RTCClient ============
class RTCWebSocketClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self._running = True
        self._connected = threading.Event()
        self._ws = None
        self._send_lock = threading.Lock()
        self.start_time = time.time()  # ✅ 初始化
    
    def execute_action(self, action: np.ndarray):
        """
        Execute the action on the robot.
        TODO: Replace with your actual robot control code.
        """
        with pred_action_lock:
            pred_action_buffer["actions"] = action
        print(f"[client] Executing action: shape={action.shape}, first_3={action.flatten()[:3]}")
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        print("[client] Connected!")
        self._connected.set()

    def _on_message(self, ws, message):
        """Receive message from server (runs in WebSocketApp thread)"""
        interval = time.time() - self.start_time
        self.start_time = time.time()
        print(f"[client] recv_action interval: {interval} seconds")

        try:
            data = json.loads(message)
            action_data = data.get("action")
            version = data.get("version", -1)
            
            if action_data is not None:
                action = convert_numpy_in_dict(action_data, numpy_deserialize)
                if isinstance(action, np.ndarray):
                    self.execute_action(action)
                    print(f"[client] Received action, version={version}")
                    
        except Exception as e:
            print(f"[client] Message processing error: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error handler"""
        print(f"[client] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        print(f"[client] Connection closed: {close_status_code} - {close_msg}")
        self._running = False
        running.clear()  # ✅ 通知主程序停止
    
    def _send_thread(self):
        """Send observations at high frequency"""
        print("[client] Send thread started")
        
        # Wait for connection
        self._connected.wait()

        prev_tick = time.perf_counter()
        
        while self._running and running.is_set():  # ✅ 检查全局 running
            start = time.time()
            try:
                # Get observation
                obs = get_observation(camera)
                
                # Serialize
                payload = convert_numpy_in_dict(obs, numpy_serialize)
                message = json.dumps(payload)
                
                # Send (thread-safe)
                with self._send_lock:
                    if self._ws and self._ws.sock and self._ws.sock.connected:
                        self._ws.send(message)
                    else:
                        print("[client] WebSocket not connected, skipping send")
                        break
                        
            except Exception as e:
                print(f"[client] Send error: {e}")
                break
            
            # # Sleep for the remaining time
            # elapsed = time.time() - start
            # sleep_time = max(0, OBS_SEND_INTERVAL - elapsed)
            # print("sleep time:", sleep_time)
            # time.sleep(sleep_time)
            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"send thread interval: {interval} seconds")
        
        print("[client] Send thread stopped")

    def run(self): 
        """Main client loop - runs WebSocketApp in current thread"""
        print(f"[client] Connecting to {self.server_url}")

        # Create WebSocketApp
        self._ws = WebSocketApp(
            self.server_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Start send thread
        send_thread = threading.Thread(target=self._send_thread, daemon=True)
        send_thread.start()
        
        # Run WebSocketApp (blocks until connection closes)
        self._ws.run_forever()
        
        # Wait for send thread to finish
        self._running = False
        send_thread.join(timeout=2.0)
        
        print("[client] Client stopped")
    
    def stop(self):
        """Stop the client"""
        self._running = False
        if self._ws:
            self._ws.close()


# ---------------- 主逻辑 ----------------
def main(server_url):
    # -------- 辅助：根据 action 构造并下发电机命令 --------
    def apply_action_from_buffer(last_pd_target):
        # 1) 每个控制周期都先读取机器人当前状态
        current_lr_arm_q, current_lr_arm_dq = master.get_robot_data()

        have_vla = False

        # 2) 读取当前 action buffer，看看这一 tick 是否有 VLA action 要用
        with pred_action_lock:
            action = pred_action_buffer["actions"]

            if action is not None:
                have_vla = True
                action = action[0]


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

                # print("action:", action)
        
        if not have_vla:
            master.torso_roll   = master.prev_torso_roll
            master.torso_pitch  = master.prev_torso_pitch
            master.torso_yaw    = master.prev_torso_yaw
            master.torso_height = master.prev_torso_height

            master.vx = master.prev_vx
            master.vy = master.prev_vy
            master.vyaw = master.prev_vyaw
            master.dyaw = master.prev_dyaw

            arm_cmd = master.prev_arm
            hand_cmd = master.prev_hand
        
        print("torso_yaw:", master.torso_yaw)
        print("torso_height:", master.torso_height)


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

    # -------- 线程：WebSocket 客户端 --------
    def websocket_thread():
        client = RTCWebSocketClient(server_url=server_url)
        client.run()  # 阻塞直到连接关闭
        print("[WS] WebSocket thread stopped")

    try:
        # 1. 先站立 20 秒
        stabilize_thread = threading.Thread(target=master.maintain_standing, daemon=True)
        stabilize_thread.start()
        master.episode_kill_event.set()
        print("[MAIN] Initialize with standing pose...")
        time.sleep(40)
        master.episode_kill_event.clear()  # 停止站立控制，只留下面的控制线程写电机

        # 2. 启动双线程
        # t_req = threading.Thread(target=action_request_thread, daemon=True)
        ####
        t_ctrl = threading.Thread(target=control_loop_thread, daemon=True)
        t_ctrl.start()

        # 3. 启动 WebSocket 线程 ✅
        t_ws = threading.Thread(target=websocket_thread, daemon=True)
        t_ws.start()

        print("[MAIN] Running. Ctrl+C to stop.")
        
        # 4. 主线程等待
        while not kill_event.is_set() and running.is_set():
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8014)
    args = parser.parse_args()
    server_url=f"ws://{args.host}:{args.port}/ws"
    main(server_url)