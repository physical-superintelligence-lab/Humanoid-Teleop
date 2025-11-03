# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is based on an initial draft generously provided by Zixuan Chen.
# -----------------------------------------------------------------------------

import types
import numpy as np
import mujoco, mujoco_viewer
import glfw
from collections import deque
import torch

from vr import VuerTeleop
import os
import sys

from robot_control.robot_arm import G1_29_ArmController, H1_2_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK, H1_2_ArmIK
from robot_control.robot_hand_inspire import Inspire_Controller
from robot_control.robot_hand_unitree import Dex3_1_Controller

from constants import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)



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

def _key_callback(self, window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_S:
        self.commands[0] -= 0.05
    elif key == glfw.KEY_W:
        self.commands[0] += 0.05
    elif key == glfw.KEY_A:
        self.commands[1] += 0.1
    elif key == glfw.KEY_D:
        self.commands[1] -= 0.1
    elif key == glfw.KEY_Q:
        self.commands[2] += 0.05
    elif key == glfw.KEY_E:
        self.commands[2] -= 0.05
    elif key == glfw.KEY_Z:
        self.commands[3] += 0.05
    elif key == glfw.KEY_X:
        self.commands[3] -= 0.05
    elif key == glfw.KEY_J:
        self.commands[4] += 0.1
    elif key == glfw.KEY_U:
        self.commands[4] -= 0.1
    elif key == glfw.KEY_K:
        self.commands[5] += 0.05
    elif key == glfw.KEY_I:
        self.commands[5] -= 0.05
    elif key == glfw.KEY_L:
        self.commands[6] += 0.05
    elif key == glfw.KEY_O:
        self.commands[6] -= 0.1
    elif key == glfw.KEY_T:
        self.commands[7] = not self.commands[7]
        if self.commands[7]:
            print("Toggled arm control ON")
        else:
            print("Toggled arm control OFF")
    elif key == glfw.KEY_ESCAPE:
        print("Pressed ESC")
        print("Quitting.")
        glfw.set_window_should_close(self.window, True)
        return
    print(
        f"vx: {self.commands[0]:<{8}.2f}"
        f"vy: {self.commands[2]:<{8}.2f}"
        f"yaw: {self.commands[1]:<{8}.2f}"
        f"height: {(0.75 + self.commands[3]):<{8}.2f}"
        f"torso yaw: {self.commands[4]:<{8}.2f}"
        f"torso pitch: {self.commands[5]:<{8}.2f}"
        f"torso roll: {self.commands[6]:<{8}.2f}"
    )

class HumanoidEnv:
    def __init__(self, policy_jit, robot_type="g1", device="cuda"):
        self.robot_type = robot_type
        self.device = device
        self.teleoperator = VuerTeleop("inspire_hand.yml", None)

        self.arm_ik = G1_29_ArmIK(Visualization=False)
        
        if robot_type == "g1":
            model_path = "g1_wrist.xml"
            self.stiffness = np.array([
                150, 150, 150, 300, 80, 20,
                150, 150, 150, 300, 80, 20,
                400, 400, 400,
                100, 100, 100, 100, 30, 30, 30,
                100, 100, 100, 100, 30, 30, 30,
            ])
            self.damping = np.array([
                2, 2, 2, 4, 2, 1,
                2, 2, 2, 4, 2, 1,
                15, 15, 15,
                7.5, 7.5, 7.5, 7.5, 6, 6, 6,
                7.5, 7.5, 7.5, 7.5, 6, 6, 6,
            ])
            self.num_actions = 15
            self.num_dofs = 29
            self.default_dof_pos = np.array([
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
                0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0,
            ])
            self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 20, 20, 20,
                25, 25, 25, 25, 20, 20, 20,
            ])
            self.arm_dof_lower_range = np.array([-0.4]*4+[-2.0]*3+[-0.4]*4+[-2.0]*3)
            self.arm_dof_upper_range = -self.arm_dof_lower_range
            self.dof_names = ["left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
                              "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
                              "waist_yaw", "waist_roll", "waist_pitch",
                              "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
                              "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"]

        else:
            raise ValueError(f"Robot type {robot_type} not supported!")
        
        self.obs_indices = np.arange(self.num_dofs)
        
        self.sim_duration = 100 * 20.0
        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.commands = np.zeros(8, dtype=np.float32)
        self.viewer.cam.distance = 2.5 # 5.0
        self.viewer.cam.elevation = 0.0
        self.viewer._key_callback = types.MethodType(_key_callback, self.viewer)
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)
        
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.action_scale = 0.25
        self.arm_action = self.default_dof_pos[15:] # 14 dim 
        self.prev_arm_action = self.default_dof_pos[15:] # 14 dim
        self.arm_blend = 0.0
        self.toggle_arm = False

        self.scales_ang_vel = 0.25
        self.scales_dof_vel = 0.05
        
        self.nj = 29
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15 # no wrist joint (model input)
        self.history_len = 10
        self.extra_history_len = 25
        self._n_demo_dof = 8 # 4+4 no wrist joint
        
        self.dof_pos = np.zeros(self.nj, dtype=np.float32)
        self.dof_vel = np.zeros(self.nj, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(self.nj)

        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3, ))
        self.demo_obs_template[:self._n_demo_dof] = self.default_dof_pos[np.r_[15:19, 22:26]]
        self.demo_obs_template[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75

        self.target_yaw = 0.0 

        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3

        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.extra_history_buf = deque(maxlen=self.extra_history_len)
        for i in range(self.history_len):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))
        for i in range(self.extra_history_len):
            self.extra_history_buf.append(np.zeros(self.n_proprio))
    
        self.policy_jit = policy_jit

        self.adapter = torch.jit.load("adapter_jit.pt", map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        norm_stats = torch.load("adapter_norm_stats.pt", weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)

        self.adapter_input = torch.zeros((1, 8 + 4), device=self.device, dtype=torch.float32)
        self.adapter_output = torch.zeros((1, 15), device=self.device, dtype=torch.float32)


        print("qpos dimension (nq):", self.model.nq)      # 总的自由度数量
        print("qvel dimension (nv):", self.model.nv)      # 对应速度
        print("ctrl dimension (nu):", self.model.nu)      # 可控输入
        for j in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            print(f"joint {j}: name={name}, qpos_addr={self.model.jnt_qposadr[j]}")

        print("Length of data.qpos:", len(self.data.qpos))
        print("Length of data.qvel:", len(self.data.qvel))
        print("Length of data.ctrl:", len(self.data.ctrl))


    def extract_data(self):
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        
    def get_observation(self):
        rpy = quatToEuler(self.quat)

        self.target_yaw = self.viewer.commands[1]
        dyaw = rpy[2] - self.target_yaw
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0

        obs_idx = np.r_[0:19, 22:26] 

        obs_dof_vel = self.dof_vel[obs_idx]
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0

        obs_dof_pos = self.dof_pos[obs_idx]
        obs_default_dof_pos = self.default_dof_pos[obs_idx]

        obs_last_action = self.last_action[obs_idx]

        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        self.adapter_input = np.concatenate([np.zeros(4), obs_dof_pos[15:]])

        self.adapter_input[0] = 0.75 + self.viewer.commands[3]
        self.adapter_input[1] = self.viewer.commands[4]
        self.adapter_input[2] = self.viewer.commands[5]
        self.adapter_input[3] = self.viewer.commands[6]

        self.adapter_input = torch.tensor(self.adapter_input).to(self.device, dtype=torch.float32).unsqueeze(0)
            
        self.adapter_input = (self.adapter_input - self.input_mean) / (self.input_std + 1e-8)
        self.adapter_output = self.adapter(self.adapter_input.view(1, -1))
        self.adapter_output = self.adapter_output * self.output_std + self.output_mean

        obs_prop = np.concatenate([
                    self.ang_vel * self.scales_ang_vel,
                    rpy[:2],
                    (np.sin(dyaw),
                    np.cos(dyaw)),
                    (obs_dof_pos - obs_default_dof_pos),
                    obs_dof_vel * self.scales_dof_vel,
                    obs_last_action,
                    gait_obs,
                    self.adapter_output.cpu().numpy().squeeze(),
        ])

        obs_priv = np.zeros((self.n_priv, ))
        obs_hist = np.array(self.proprio_history_buf).flatten()

        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self._n_demo_dof] = obs_dof_pos[15:]
        obs_demo[self._n_demo_dof] = self.viewer.commands[0]
        obs_demo[self._n_demo_dof+1] = self.viewer.commands[2]
        self._in_place_stand_flag = np.abs(self.viewer.commands[0]) < 0.1
        obs_demo[self._n_demo_dof+3] = self.viewer.commands[4]
        obs_demo[self._n_demo_dof+4] = self.viewer.commands[5]
        obs_demo[self._n_demo_dof+5] = self.viewer.commands[6]
        obs_demo[self._n_demo_dof+6:self._n_demo_dof+9] = 0.75 + self.viewer.commands[3]

        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))
        
    def run(self):        
        for i in range(int(self.sim_duration / self.sim_dt)):
            self.extract_data()
            
            if i % self.sim_decimation == 0:
                obs = self.get_observation()
                
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    extra_hist = torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float).view(1, -1).to(self.device)
                    raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                raw_action = np.clip(raw_action, -40., 40.)
                self.last_action = np.concatenate([raw_action.copy(), (self.dof_pos - self.default_dof_pos)[15:] / self.action_scale])
                scaled_actions = raw_action * self.action_scale
                
                if i % 100 == 0 and i > 0 and self.viewer.commands[7]:
                    print("Δarm_action =", np.linalg.norm(self.arm_action - self.prev_arm_action))
                    self.arm_blend = 0
                    self.prev_arm_action = self.dof_pos[15:].copy()
                    # self.arm_action = np.random.uniform(0, 1, 14) * (self.arm_dof_upper_range - self.arm_dof_lower_range) + self.arm_dof_lower_range
                    head_rmat, left_pose, right_pose, left_qpos, right_qpos = (
                        self.teleoperator.step()
                    )

                    print("head_rmat shape:", np.shape(head_rmat))
                    print("head_rmat:\n", head_rmat)

                    print("left_pose shape:", np.shape(left_pose))
                    print("left_pose:", left_pose)

                    print("right_pose shape:", np.shape(right_pose))
                    print("right_pose:", right_pose)
                    
                    current_lr_arm_q = self.dof_pos[15:]
                    sol_q, tau_ff = self.arm_ik.solve_ik(
                        left_pose, right_pose, current_lr_arm_q, None
                    )
                    self.arm_action = sol_q
                    self.toggle_arm = True

                    current_q = self.dof_pos[15:]
                    delta_q = self.arm_action - current_q
                    velocity_limit = 30.0 
                    max_step = velocity_limit * self.control_dt  # 每次控制周期最大可移动角度

                    motion_scale = np.max(np.abs(delta_q)) / max_step
                    if motion_scale > 1.0:
                        self.arm_action = current_q + delta_q / motion_scale

                elif not self.viewer.commands[7]:
                    if self.toggle_arm:
                        self.toggle_arm = False
                        self.arm_blend = 0
                        self.prev_arm_action = self.dof_pos[15:].copy()
                        self.arm_action = self.default_dof_pos[15:]
                pd_target = np.concatenate([scaled_actions, np.zeros(14)]) + self.default_dof_pos
                pd_target[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
                self.arm_blend = min(1.0, self.arm_blend + 0.1)
                

                self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
                if self._in_place_stand_flag and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.25])
                if (not self._in_place_stand_flag) and ((np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)):
                    self.gait_cycle = np.array([0.25, 0.75])
                
                # self.viewer.cam.azimuth += 0.1
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                self.viewer.render()
                
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()

if __name__ == "__main__":

    robot = "g1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    policy_pth = 'amo_jit.pt'
    
    policy_jit = torch.jit.load(policy_pth, map_location=device)
    
    env = HumanoidEnv(policy_jit=policy_jit, robot_type=robot, device=device)
    
    env.run()
        
        
