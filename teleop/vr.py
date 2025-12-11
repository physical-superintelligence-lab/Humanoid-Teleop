import os
import sys
import time
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from pathlib import Path

import numpy as np
import yaml
from robot_control.dex_retargeting.retargeting_config import RetargetingConfig

from constants_vuer import tip_indices
from robot_control.hand_retargeting import HandRetargeting, HandType
from TeleVision import OpenTeleVision

import threading
import zmq

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from constants_vuer import (
    T_robot_openxr,
    T_to_unitree_hand,
    grd_yup2grd_zup,
    hand2inspire,
    hand2inspire_l_arm,
    hand2inspire_l_finger,
    hand2inspire_r_arm,
    hand2inspire_r_finger,
)
from motion_utils import fast_mat_inv, mat_update


class ManusSkeletonReceiver:

    def __init__(
        self,
        address="tcp://localhost:8000",
        left_glove_sn="85ab6e24",
        right_glove_sn="c152afa7",
    ):
        self.left_glove_sn = left_glove_sn
        self.right_glove_sn = right_glove_sn

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PULL)

        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(address)

        self._lock = threading.Lock()
        self._left_xyz = None   # shape (25,3)
        self._right_xyz = None  # shape (25,3)
        self._running = True

        self._prev_left_xyz = None
        self._prev_right_xyz = None


        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
    

    def _detect_activity(self):
        # debug 左手
        if self._prev_left_xyz is not None and self._left_xyz is not None:
            diffs = np.linalg.norm(self._left_xyz - self._prev_left_xyz, axis=1)
            print("LEFT diffs:", np.round(diffs, 4))  # 25 scalars
        self._prev_left_xyz = None if self._left_xyz is None else self._left_xyz.copy()

        # debug 右手
        if self._prev_right_xyz is not None and self._right_xyz is not None:
            diffs = np.linalg.norm(self._right_xyz - self._prev_right_xyz, axis=1)
            # print("RIGHT diffs:", np.round(diffs, 4))
        self._prev_right_xyz = None if self._right_xyz is None else self._right_xyz.copy()


    def _parse_skeleton_176(self, data_176):
        """
        data_176: list[str]，长度 176，第 0 个是 SN，后面 175 个 float:
        25 bones * (x,y,z,qx,qy,qz,qw) = 175
        返回: xyz, shape = (25,3)
        """
        # data_176[0] 是序列号，用来区分左右手
        floats = list(map(float, data_176[1:]))  # 175
        arr = np.asarray(floats, dtype=np.float32).reshape(25, 7)
        xyz = arr[:, :3]  # (25,3)
        return xyz

    def _loop(self):
        while self._running:
            try:
                msg = self.socket.recv()  # 阻塞直到有一条最新
            except zmq.error.ZMQError:
                break

            text = msg.decode("utf-8")
            parts = text.split(",")

            with self._lock:
                if len(parts) == 176:
                    sn = parts[0]
                    xyz = self._parse_skeleton_176(parts)
                    if sn == self.left_glove_sn:
                        self._left_xyz = xyz
                    elif sn == self.right_glove_sn:
                        self._right_xyz = xyz

                elif len(parts) == 352:
                    # 左右手各 176
                    left_parts = parts[0:176]
                    right_parts = parts[176:352]
                    left_sn = left_parts[0]
                    right_sn = right_parts[0]
                    left_xyz = self._parse_skeleton_176(left_parts)
                    right_xyz = self._parse_skeleton_176(right_parts)

                    # 不严格依赖 SN 顺序，按 SN 匹配
                    if left_sn == self.left_glove_sn:
                        self._left_xyz = left_xyz
                    elif left_sn == self.right_glove_sn:
                        self._right_xyz = left_xyz

                    if right_sn == self.left_glove_sn:
                        self._left_xyz = right_xyz
                    elif right_sn == self.right_glove_sn:
                        self._right_xyz = right_xyz
                
                # self._detect_activity()


    def get_latest(self):
        """在 Vuer 每帧 step() 里调用，拿到“当前一帧”的双手 xyz。"""
        with self._lock:
            if self._left_xyz is None or self._right_xyz is None:
                return None, None
            return self._left_xyz.copy(), self._right_xyz.copy()

    def stop(self):
        self._running = False
        try:
            self.socket.close(0)
        except Exception:
            pass



class VuerPreprocessor:
    def __init__(self, manus_receiver=None):
        self.manus_receiver = manus_receiver

        self.vuer_head_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1]]
        )
        self.vuer_right_wrist_mat = np.array(
            [[1, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

        self.vuer_left_wrist_mat = np.array(
            [[1, 0, 0, -0.5], [0, 1, 0, 1], [0, 0, 1, -0.5], [0, 0, 0, 1]]
        )

    def process(self, tv):
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        if self.manus_receiver is not None:
            manus_left, manus_right = self.manus_receiver.get_latest()
        else:
            manus_left, manus_right = None, None

        if manus_left is not None and manus_right is not None:
            # 使用 Manus 作为 finger point 源
            left_landmarks = manus_left    # shape (25,3)
            right_landmarks = manus_right  # shape (25,3)
            # print("Using manus as finger tracking")
            print(
                "[MANUS raw tips] thumb", manus_left[24],
                " index", manus_left[4],
                " middle", manus_left[9],
            )
        else:
            # 回退用 VP 原始 landmarks
            left_landmarks = tv.left_landmarks.copy()
            right_landmarks = tv.right_landmarks.copy()
        

        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        rel_left_wrist_mat = (
            fast_mat_inv(head_mat) @ left_wrist_mat @ hand2inspire_l_arm
        )
        # rel_left_wrist_mat[0:3, 3] = sensitivity * (
        #     rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]
        # )

        rel_right_wrist_mat = (
            fast_mat_inv(head_mat) @ right_wrist_mat @ hand2inspire_r_arm
        )  # wTr = wTh @ hTr
        # rel_right_wrist_mat[0:3, 3] = sensitivity * (
        #     rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
        # )

        # head_rmat_inv = fast_mat_inv(head_mat)
        # rel_right_wrist_mat[:3, :3] = (
        #     head_rmat_inv[:3, :3] @ rel_right_wrist_mat[:3, :3]
        # )
        # rel_left_wrist_mat[:3, :3] = head_rmat_inv[:3, :3] @ rel_left_wrist_mat[:3, :3]

        # homogeneous
        left_hand_vuer_mat = np.concatenate(
            [left_landmarks.copy().T, np.ones((1, left_landmarks.shape[0]))]
        )
        right_hand_vuer_mat = np.concatenate(
            [right_landmarks.copy().T, np.ones((1, right_landmarks.shape[0]))]
        )

        # print("LEFT landmarks shape:", left_landmarks.shape)
        # print("RIGHT landmarks shape:", right_landmarks.shape)

        # change of basis
        left_hand_mat = T_robot_openxr @ left_hand_vuer_mat
        right_hand_mat = T_robot_openxr @ right_hand_vuer_mat

        left_hand_mat_wb = fast_mat_inv(left_wrist_mat) @ left_hand_mat
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat

        unitree_left_hand = (T_to_unitree_hand @ left_hand_mat_wb)[0:3, :].T
        unitree_right_hand = (T_to_unitree_hand @ right_hand_mat_wb)[0:3, :].T
        # print("[After vr transforming] left hand:", unitree_left_hand.shape)

        # unitree_left_hand = manus_left.copy()
        # print("[No vr transforming] left hand:", unitree_left_hand.shape)

        unitree_tip_indices = [4, 9, 14]  # [thumb, index, middle] in OpenXR

        # unitree_tip_indices = [24, 4, 9]  # [thumb, index, middle] in MANUS


        # Check if hand data is initialized
        left_q_target, right_q_target = None, None
        hand_retargeting = HandRetargeting(
            HandType.UNITREE_DEX3
        )  # TODO: add if to distinguish hand
        # hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        if not np.all(left_hand_mat == 0.0):
            # Extract the relevant tip indices (assumed defined elsewhere)
            ref_left_value = unitree_left_hand[unitree_tip_indices].copy()
            ref_right_value = unitree_right_hand[unitree_tip_indices].copy()

            # Apply scaling factors to calibrate the values
            ref_left_value[0] *= 1.15
            ref_left_value[1] *= 1.05
            ref_left_value[2] *= 0.95

            ref_right_value[0] *= 1.15
            ref_right_value[1] *= 1.05
            ref_right_value[2] *= 0.95

            # Use the retargeting methods to convert reference values to qpos.
            left_q_target = hand_retargeting.left_retargeting.retarget(ref_left_value)[
                hand_retargeting.right_dex_retargeting_to_hardware
            ]
            right_q_target = hand_retargeting.right_retargeting.retarget(
                ref_right_value
            )[hand_retargeting.right_dex_retargeting_to_hardware]

        return (
            head_mat,
            rel_left_wrist_mat,
            rel_right_wrist_mat,
            left_q_target,
            right_q_target,
        )

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(
            self.vuer_right_wrist_mat, tv.right_hand.copy()
        )
        self.vuer_left_wrist_mat = mat_update(
            self.vuer_left_wrist_mat, tv.left_hand.copy()
        )

        # change of basis
        right_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )
        left_wrist_mat = (
            grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        )

        left_fingers = np.concatenate(
            [tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))]
        )
        right_fingers = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))]
        )

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire_l_finger.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire_r_finger.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers


class VuerTeleop:
    def __init__(self, config_file_path, img_shm_name):
        # self.resolution = (720,1280) #(480,640)
        self.resolution = (720, 640)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        
        if img_shm_name is None:
            self.shm = shared_memory.SharedMemory(
                create=True, size=np.prod(self.img_shape) * np.uint8().itemsize
            )
        else:
            self.shm = shared_memory.SharedMemory(name=img_shm_name)
            
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped, self.shm.name, image_queue, toggle_streaming
        )
        self.manus_receiver = ManusSkeletonReceiver(
            address="tcp://localhost:8000",
            left_glove_sn="85ab6e24",
            right_glove_sn="c152afa7",
        )

        # self.processor = VuerPreprocessor(manus_receiver=self.manus_receiver)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir("../assets")
        with Path(config_file_path).open("r") as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg["left"])
        right_retargeting_config = RetargetingConfig.from_dict(cfg["right"])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self, full_head=False):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = (
            self.processor.process(self.tv)
        )
        if full_head:
            head_rmat = head_mat
        else:
            head_rmat = head_mat[:3, :3]

        left_wrist_mat[2, 3] += 0.55
        right_wrist_mat[2, 3] += 0.55
        left_wrist_mat[0, 3] += 0.05
        right_wrist_mat[0, 3] += 0.05

        # left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[
        #     [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]
        # ]
        # right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[
        #     [4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]
        # ]

        return head_rmat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat

    def shutdown(self):
        # self.shm.close()
        # self.shm.unlink()
        self.tv.shutdown()
        if hasattr(self, "manus_receiver") and self.manus_receiver is not None:
            self.manus_receiver.stop()
