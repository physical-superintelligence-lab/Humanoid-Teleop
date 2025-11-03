import numpy as np
import torch
import math
from typing import Optional
import pink
import pinocchio as pin
from pink import solve_ik as pink_solve_ik
from pink.limits import VelocityLimit
from pink.tasks.task import Task

from robot_control.robot_arm_joints import G1_JOINT_NAME_MAP

from datetime import datetime
import os, sys, atexit




AMO_CONFIG_DEFAULTS = {
    # === Task Weights ===
    # Global position/orientation weights for each end-effector task.
    # Larger AMO_IK_POS_W → stronger positional correction.
    # Larger AMO_IK_ORI_W → more emphasis on orientation alignment.
    "AMO_IK_POS_W": "1.0",
    "AMO_IK_ORI_W": "0.5",

    # Relative weighting between multi-target tasks (head / left arm / right arm).
    # Higher values prioritize that target in the optimization.
    "AMO_IK_W_HEAD": "2.5",
    "AMO_IK_W_LEFT": "1.0",
    "AMO_IK_W_RIGHT": "1.0",

    # === Solver Regularization ===
    # Levenberg–Marquardt damping term for stability in underdetermined systems.
    "AMO_IK_LM": "1e-3",

    # QP damping used in Pink’s quadratic solver (helps numerical stability).
    "AMO_IK_QP_DAMP": "1e-4",

    # Regularization terms that penalize large joint deviations or torso movements:
    # - REG_Q: arm posture regularization (encourages minimal deviation from previous pose)
    # - REG_H: pelvis height regularization
    # - REG_RPY: torso orientation (yaw/pitch/roll) regularization
    "AMO_IK_REG_Q": "0.0005",
    "AMO_IK_REG_H": "0.0004",
    "AMO_IK_REG_RPY": "0.006",

    # === Virtual Joint Velocity Limits ===
    # Limit the maximum speed (rad/s or m/s) of the "virtual" joints controlling:
    # - pelvis height (pz)
    # - torso yaw, pitch, and roll
    # These prevent sudden unrealistic body motion while keeping motion smooth.
    "AMO_IK_VEL_PZ": "8.0",      # m/s  — allows fast vertical adjustment
    "AMO_IK_VEL_YAW": "0.30",    # rad/s
    "AMO_IK_VEL_PITCH": "0.30",  # rad/s
    "AMO_IK_VEL_ROLL": "0.30",   # rad/s

    # If set to "1", disables all Pink velocity constraints (use with caution).
    "AMO_IK_NO_LIMITS": "0",

    # === Solver & Debug ===
    # Pink QP solver backend — usually "quadprog" (can be changed to "proxqp" or others).
    "AMO_QP_SOLVER": "quadprog",

    # If set to "1", suppresses most console debug prints.
    "AMO_SILENT": "1",
}

# Apply all defaults to the environment (only if not already set externally)
for k, v in AMO_CONFIG_DEFAULTS.items():
    os.environ.setdefault(k, str(v))


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


def rotation_matrix_to_rpy(R):
    sy = -R[2, 0]
    pitch = math.asin(np.clip(sy, -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


class FrameFrobeniusTask(Task):
    """
    Task for aligning a frame's position + orientation using
    Frobenius norm on rotation error.

    Error vector:
        e = [ (p_des - p) , vec(R_des - R) ]
    Jacobian:
        J = [ -Jv ; JR ]
        where JR_i = vec(- [ω_i]^ @ R)
    """

    def __init__(self, frame_name: str, model: pin.Model, data: pin.Data,
                 pos_w: float = 1.0, ori_w: float = 0.5,
                 lm_damping: float = 1e-2, gain: float = 1.0):
        super().__init__(
            cost=np.hstack([
                np.sqrt(pos_w) * np.ones(3),
                np.sqrt(ori_w) * np.ones(9)
            ]),
            gain=gain,
            lm_damping=lm_damping,
        )
        self._frame_name = frame_name
        self._frame_id = model.getFrameId(frame_name)
        self._model = model
        self._data = data
        self._R_des = np.eye(3)
        self._p_des = np.zeros(3)

    def set_target(self, p_des: np.ndarray, R_des: np.ndarray):
        """Set desired translation and rotation."""
        self._p_des = np.asarray(p_des, dtype=np.float64).reshape(3)
        self._R_des = np.asarray(R_des, dtype=np.float64).reshape(3, 3)

    def compute_error(self, configuration):
        oMf = configuration.get_transform_frame_to_world(self._frame_name)
        p = oMf.translation
        R = oMf.rotation
        e_p = self._p_des - p
        e_R = (self._R_des - R).reshape(9)
        return np.hstack([e_p, e_R]).astype(np.float64)

    def compute_jacobian(self, configuration):
        """Compute Jacobian aligned with the error vector."""
        model = configuration.model
        data = configuration.data
        q = configuration.q
        fid = model.getFrameId(self._frame_name)

        # Compute joint Jacobians and frame placement
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        # Local-world-aligned frame Jacobian
        J6 = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jv = J6[:3, :]
        Jw = J6[3:, :]

        R = configuration.get_transform_frame_to_world(self._frame_name).rotation
        JR = np.zeros((9, Jw.shape[1]), dtype=np.float64)
        for i in range(Jw.shape[1]):
            wx, wy, wz = Jw[:, i]
            w_hat = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
            JR[:, i] = (-w_hat @ R).reshape(9)
        return np.vstack([-Jv, JR])

    def __repr__(self):
        try:
            pos_w = float(self.cost[0]**2)
            ori_w = float(self.cost[3]**2)
        except Exception:
            pos_w = ori_w = 0.0
        return f"FrameFrobeniusTask(frame={self._frame_name}, pos_w={pos_w:.3f}, ori_w={ori_w:.3f})"


class RPYHRegularizationTask(Task):
    """
    Regularization task for torso orientation (yaw, pitch, roll)
    and pelvis height (h). Keeps these values close to a reference.
    """

    def __init__(self, model: pin.Model, data: pin.Data,
                 torso_frame: str = "torso_link",
                 pelvis_frame: str = "pelvis",
                 w_height: float = 5e-2, w_rpy: float = 5e-2,
                 lm_damping: float = 1e-3, gain: float = 1.0,
                 jname_pz=None, jname_yaw=None, jname_pitch=None, jname_roll=None):
        super().__init__(
            cost=np.array([np.sqrt(w_height), np.sqrt(w_rpy), np.sqrt(w_rpy), np.sqrt(w_rpy)], dtype=np.float64),
            gain=gain,
            lm_damping=lm_damping,
        )
        self._model = model
        self._data = data
        self._fid_torso = model.getFrameId(torso_frame)
        self._fid_pelvis = model.getFrameId(pelvis_frame)
        self._h_ref = 0.75
        self._rpy_ref = np.zeros(3, dtype=np.float64)

        def _v_idx(jname):
            try:
                jid = model.getJointId(jname)
                return model.joints[jid].idx_v if jid > 0 else None
            except Exception:
                return None

        self._iv_pz    = _v_idx(jname_pz)
        self._iv_yaw   = _v_idx(jname_yaw)
        self._iv_pitch = _v_idx(jname_pitch)
        self._iv_roll  = _v_idx(jname_roll)

    def set_reference(self, h_ref: float, rpy_ref: np.ndarray):
        self._h_ref = float(h_ref)
        self._rpy_ref = np.asarray(rpy_ref, dtype=np.float64).reshape(3)

    def compute_error(self, configuration):
        oMt = configuration.get_transform_frame_to_world(self._model.frames[self._fid_torso].name)
        oMp = configuration.get_transform_frame_to_world(self._model.frames[self._fid_pelvis].name)
        p = oMp.translation
        R = oMt.rotation

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])

        h = p[2]
        e_h = self._h_ref - h
        e_rpy = self._rpy_ref - np.array([yaw, pitch, roll])
        return np.hstack([e_h, e_rpy]).astype(np.float64)

    def compute_jacobian(self, configuration):
        J = np.zeros((4, configuration.model.nv), dtype=np.float64)
        if self._iv_pz is not None: J[0, self._iv_pz] = 1.0
        if self._iv_yaw is not None: J[1, self._iv_yaw] = 1.0
        if self._iv_pitch is not None: J[2, self._iv_pitch] = 1.0
        if self._iv_roll is not None: J[3, self._iv_roll] = 1.0
        return J
    
    def __repr__(self):
        rpy_str = np.array2string(np.round(self._rpy_ref, 3), separator=',', precision=3)
        return f"RPYHRegularizationTask(h_ref={self._h_ref:.3f}, rpy_ref={rpy_str})"



class ArmPostureTask(Task):
    """
    joint posture regularization for upper limbs.
    Encourages the current configuration to stay close to reference q_ref.
    """

    def __init__(self, model: pin.Model, data: pin.Data,
                 ctrl_q_ids: np.ndarray, ctrl_v_ids: np.ndarray,
                 w_vec: np.ndarray):
        super().__init__(
            cost=np.sqrt(w_vec).astype(np.float64),
            gain=1.0, lm_damping=1e-3
        )
        self._model = model
        self._data = data
        self._q_ids = np.asarray(ctrl_q_ids, dtype=int)
        self._v_ids = np.asarray(ctrl_v_ids, dtype=int)
        self._q_ref = None

    def set_reference(self, q_ref: np.ndarray):
        self._q_ref = np.asarray(q_ref, dtype=np.float64).copy()

    def compute_error(self, configuration):
        q = configuration.q
        if self._q_ref is None:
            self._q_ref = q.copy()
        return (q[self._q_ids] - self._q_ref[self._q_ids]).astype(np.float64)

    def compute_jacobian(self, configuration):
        nv = configuration.model.nv
        J = np.zeros((len(self._q_ids), nv), dtype=np.float64)
        for i, vid in enumerate(self._v_ids):
            J[i, vid] = 1.0
        return J

    def __repr__(self):
        mean_w = np.mean(self.cost ** 2) if hasattr(self, "cost") else 0.0
        return f"ArmPostureTask(n_ctrl={len(self._q_ids)}, mean_w={mean_w:.2e})"






class PinkIKSolver:
    def __init__(self, urdf_path: str, frame_names: dict, controlled_joint_names: list):
        """Initialize Pink IK solver for multi-target humanoid upper-body IK."""
        if pink is None:
            raise RuntimeError("Pink is required. Install via `pip install pin-pink`.")
        if pin is None:
            raise RuntimeError("Pinocchio is required. Install via conda-forge or pip.")

        # === Load model ===
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.has_free_flyer = False

        # === Frame definitions ===
        self.frame_left_name  = frame_names["left"]
        self.frame_right_name = frame_names["right"]
        self.frame_head_name  = frame_names["head"]
        self.frame_left  = self._ensure_frame(self.frame_left_name)
        self.frame_right = self._ensure_frame(self.frame_right_name)
        self.frame_head  = self._ensure_frame(self.frame_head_name)
        self.frame_torso  = self._ensure_frame("torso_link")
        self.frame_pelvis = self._ensure_frame("pelvis")

        # Virtual joint names (used for torso orientation + pelvis height)
        self.jname_ik_pz    = "ik_pelvis_z_joint"
        self.jname_ik_yaw   = "ik_torso_yaw_joint"
        self.jname_ik_pitch = "ik_torso_pitch_joint"
        self.jname_ik_roll  = "ik_torso_roll_joint"

        # === Controlled joints ===
        self.ctrl_jids = [self._joint_id_from_name(n) for n in controlled_joint_names]
        self.ctrl_dof_ids = np.concatenate([
            np.arange(self.model.joints[jid].idx_v,
                    self.model.joints[jid].idx_v + self.model.joints[jid].nv, dtype=int)
            for jid in self.ctrl_jids
        ])
        self.ctrl_order = list(controlled_joint_names)
        self.ctrl_q_ids = np.array([self.model.joints[jid].idx_q for jid in self.ctrl_jids], dtype=int)
        self.nd = self.model.nv
        self.nd_ctrl = self.ctrl_dof_ids.shape[0]

        # === IK weights ===
        self.pos_w = float(os.environ.get("AMO_IK_POS_W", 1.0))
        self.ori_w = float(os.environ.get("AMO_IK_ORI_W", 0.5))
        self.target_weights = {
            "head":  float(os.environ.get("AMO_IK_W_HEAD",  2.5)),
            "left":  float(os.environ.get("AMO_IK_W_LEFT",  1.0)),
            "right": float(os.environ.get("AMO_IK_W_RIGHT", 1.0)),
        }

        # === Regularization & solver parameters ===
        self.lm_damping = float(os.environ.get("AMO_IK_LM", 1e-3))
        self.qp_damping = float(os.environ.get("AMO_IK_QP_DAMP", 1e-4))
        self.reg_q   = float(os.environ.get("AMO_IK_REG_Q", 5e-4))
        self.reg_h   = float(os.environ.get("AMO_IK_REG_H", 4e-4))
        self.reg_rpy = float(os.environ.get("AMO_IK_REG_RPY", 6e-3))
        self.dt = 0.02

        # === Initialize tasks ===
        self._FrameFrobeniusTask = FrameFrobeniusTask
        self._RPYHRegularizationTask = RPYHRegularizationTask
        self._ArmPostureTask = ArmPostureTask

        # Create multi-target tasks
        self._tasks = {
            "left":  FrameFrobeniusTask(self.frame_left_name,  self.model, self.data, self.pos_w, self.ori_w),
            "right": FrameFrobeniusTask(self.frame_right_name, self.model, self.data, self.pos_w, self.ori_w),
            "head":  FrameFrobeniusTask(self.frame_head_name,  self.model, self.data, self.pos_w, self.ori_w),
        }

        # === Velocity limits ===
        try:
            self.model.velocityLimit = np.zeros(self.nd, dtype=np.float64)
        except Exception:
            self.model.velocityLimit = np.zeros(self.nd).astype(np.float64)

        def set_v_limit(joint_name, cap):
            try:
                jid = self.model.getJointId(joint_name)
                if jid > 0:
                    iv = self.model.joints[jid].idx_v
                    self.model.velocityLimit[iv] = float(cap)
            except Exception:
                pass

        # Arm (slow) vs wrist (fast) limits
        upper_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint"
        ]
        wrist_joints = [
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        for jn in upper_joints: set_v_limit(jn, 8.0)
        for jn in wrist_joints: set_v_limit(jn, 12.0)

        # Virtual joints (pz, yaw, pitch, roll)
        set_v_limit(self.jname_ik_pz,    float(os.environ.get("AMO_IK_VEL_PZ",    8.0)))
        set_v_limit(self.jname_ik_yaw,   float(os.environ.get("AMO_IK_VEL_YAW",   0.3)))
        set_v_limit(self.jname_ik_pitch, float(os.environ.get("AMO_IK_VEL_PITCH", 0.3)))
        set_v_limit(self.jname_ik_roll,  float(os.environ.get("AMO_IK_VEL_ROLL",  0.3)))

        if VelocityLimit is None:
            raise RuntimeError("pink.limits.VelocityLimit not available.")
        self._vel_limit = VelocityLimit(self.model)

        # === Posture regularization (arms) ===
        w_upper = self.reg_q
        # w_wrist = 0.25 * w_upper
        w_wrist = w_upper
        w_vec = np.array([
            *([w_upper] * 8),
            *([w_wrist] * 6)
        ], dtype=np.float64)
        self._arm_posture_task = ArmPostureTask(
            self.model, self.data, self.ctrl_q_ids, self.ctrl_dof_ids, w_vec=w_vec
        )

        # === Torso & pelvis regularization (created once, reused each solve) ===
        self._rpyh_task = RPYHRegularizationTask(
            self.model, self.data,
            torso_frame="torso_link", pelvis_frame="pelvis",
            w_height=self.reg_h, w_rpy=self.reg_rpy,
            lm_damping=self.lm_damping, gain=1.0,
            jname_pz=self.jname_ik_pz,
            jname_yaw=self.jname_ik_yaw,
            jname_pitch=self.jname_ik_pitch,
            jname_roll=self.jname_ik_roll,
        )

        # === Cache all task objects 
        self._all_tasks = [
            self._tasks["left"],
            self._tasks["right"],
            self._tasks["head"],
            self._arm_posture_task,
            self._rpyh_task,
        ]


        # === Debug print ===
        self.debug_ik = True
        if self.debug_ik:
            print("[IK:init] frames =", self.frame_left_name, self.frame_right_name, self.frame_head_name)
            print("[IK:init] nd_ctrl =", self.nd_ctrl, "pos_w=", self.pos_w, "ori_w=", self.ori_w)
            print("[IK:init] reg_h=", self.reg_h, "reg_rpy=", self.reg_rpy, "dt=", self.dt)

        self._solve_call_count = 0
        

    def solve(self, q_init: np.ndarray, targets: dict,
            substeps: int = 8,
            h0: Optional[float] = None,
            rpy0: Optional[np.ndarray] = None) -> tuple:
        """
        Multi-target IK solver (head + left + right + torso regularization).
        """
        # --- Initialize configuration ---
        cfg = pink.Configuration(self.model, self.data, q_init.copy())

        # --- 1. Update main target tasks (head / left / right) ---
        active_tasks = []
        base_cost = np.hstack([
            np.sqrt(self.pos_w) * np.ones(3),
            np.sqrt(self.ori_w) * np.ones(9)
        ])

        for key, task in self._tasks.items():
            if key not in targets:
                continue
            p_des, R_des = targets[key]
            task.set_target(p_des, R_des)

            # Scale weight inplace (avoid new arrays)
            w = float(self.target_weights.get(key, 1.0))
            task.cost[:] = np.sqrt(w) * base_cost
            active_tasks.append(task)

        # --- 2. Arm posture regularization ---
        self._arm_posture_task.set_reference(cfg.q)
        active_tasks.append(self._arm_posture_task)

        # --- 3. Torso + pelvis regularization (reuse cached RPYH task) ---
        if h0 is None or rpy0 is None:
            oMt = cfg.get_transform_frame_to_world("torso_link")
            oMp = cfg.get_transform_frame_to_world("pelvis")
            roll, pitch, yaw = rotation_matrix_to_rpy(oMt.rotation)
            h_ref = float(oMp.translation[2])
            rpy_ref = np.array([yaw, pitch, roll])
        else:
            rpy0 = np.asarray(rpy0, dtype=np.float64).reshape(3)
            h_ref = float(h0)
            rpy_ref = np.array([rpy0[0], rpy0[1], rpy0[2]], dtype=np.float64)

        self._rpyh_task.set_reference(h_ref, rpy_ref)
        active_tasks.append(self._rpyh_task)

        # --- 4. Velocity constraints ---
        constraints = []
        if hasattr(self, "_vel_limit") and os.environ.get("AMO_IK_NO_LIMITS", "0") != "1":
            constraints = [self._vel_limit]
        elif self.debug_ik:
            print("[IKDBG] VelocityLimit disabled by env")

        # --- 5. Solve (multi-substep integration) ---
        dt_sub = self.dt / max(1, int(substeps))
        solver_name = os.environ.get("AMO_QP_SOLVER", "quadprog")

        v_norm_total = 0.0
        for _ in range(substeps):
            v = pink_solve_ik(
                cfg, active_tasks,
                dt=dt_sub,
                solver=solver_name,
                damping=self.qp_damping,
                limits=constraints,
                safety_break=False,
            )
            v_norm_total += np.linalg.norm(v)
            cfg.integrate_inplace(v, dt_sub)

        # --- 6. Extract results ---
        oMp = cfg.get_transform_frame_to_world("pelvis")
        oMt = cfg.get_transform_frame_to_world("torso_link")
        roll, pitch, yaw = rotation_matrix_to_rpy(oMt.rotation)
        h_new = float(oMp.translation[2])
        rpy_new = np.array([roll, pitch, yaw])

        if self.debug_ik and os.environ.get("AMO_SILENT", "1") != "1":
            print(f"[IK:solve] |v|~{v_norm_total/substeps:.3e} solver={solver_name}")

        return cfg.q.copy(), h_new, rpy_new



    def get_ctrl_joint_angles_in_order(self, q: np.ndarray) -> np.ndarray:
        vals = []
        for jid in self.ctrl_jids:
            j = self.model.joints[jid]
            if j.nq != 1:
                raise RuntimeError("Only 1-DoF joints supported in controlled set")
            vals.append(float(q[j.idx_q]))
        return np.array(vals, dtype=np.float64)
    

    def assemble_q(self, joint_pos, base_xyz=None, base_quat_wxyz=None):
        """
        Assemble a full URDF-format joint vector (q) from Real joint positions.

        Args:
            joint_pos: 29 dim joint pos array read from Humanoid
            base_xyz: pelvis world position (3,)
            base_quat_wxyz: pelvis world orientation quaternion (w,x,y,z)
        Returns:
            q (np.ndarray): URDF-format configuration vector (nq,)
        """
        q = np.zeros(self.model.nq, dtype=np.float64)

        qw, qx, qy, qz = base_quat_wxyz
        q_pin_base = np.array([base_xyz[0], base_xyz[1], base_xyz[2], qx, qy, qz, qw])
        name_to_pos = {"pelvis": q_pin_base}
        for idx, joint_name in G1_JOINT_NAME_MAP.items():
            name_to_pos[joint_name] = float(joint_pos[idx])

        # === Map Mujoco joints to URDF joints ===
        for jname, val in name_to_pos.items():
            jid = self.model.getJointId(jname)
            if jid <= 0 or jid >= len(self.model.joints):
                continue
            j = self.model.joints[jid]
            idx_q, nq = int(j.idx_q), int(j.nq)

            if nq == 1:
                q[idx_q] = float(val)
            elif nq == 7:
                q[idx_q:idx_q + 7] = np.asarray(val, dtype=np.float64).reshape(7,)

        # === Floating base (pelvis) pose ===
        if base_xyz is not None and base_quat_wxyz is not None:
            try:
                # Convert (w,x,y,z) → (x,y,z,w)
                w, x, y, z = base_quat_wxyz
                q_base = np.array([*base_xyz, x, y, z, w], dtype=np.float64)

                # Find floating base joint or first nq==7 joint
                jid_fb = self.model.getJointId("floating_base_joint")

                # If invalid or out of range → search for first nq == 7 joint
                if jid_fb <= 0 or jid_fb >= len(self.model.joints):
                    jid_fb = next(
                        (jid for jid in range(1, len(self.model.joints))
                        if getattr(self.model.joints[jid], "nq", 0) == 7),
                        None
                    )

                # Write safely if found and valid
                if jid_fb is not None and 0 < jid_fb < len(self.model.joints):
                    j0 = self.model.joints[jid_fb]
                    i0 = int(getattr(j0, "idx_q", -1))
                    if 0 <= i0 and i0 + 7 <= self.model.nq:
                        q[i0:i0 + 7] = q_base
            except Exception as e:
                print(f"[IKDBG] write floating_base_joint failed: {e}")

        # === Virtual joint initialization (pelvis_z + torso_rpy) ===
        try:
            # Pelvis height
            if base_xyz is not None:
                jid_h = self.model.getJointId(getattr(self, "jname_ik_pz", "ik_pelvis_z_joint"))
                if 0 < jid_h < len(self.model.joints):
                    iq = int(self.model.joints[jid_h].idx_q)
                    if 0 <= iq < self.model.nq:
                        q[iq] = float(base_xyz[2])

            # Torso RPY
            if base_quat_wxyz is not None:
                qw, qx, qy, qz = map(float, base_quat_wxyz)
                roll, pitch, yaw = quatToEuler(np.array([qw, qx, qy, qz], dtype=np.float64))
                for jn, val in [
                    (getattr(self, "jname_ik_yaw", "ik_torso_yaw_joint"), yaw),
                    (getattr(self, "jname_ik_pitch", "ik_torso_pitch_joint"), pitch),
                    (getattr(self, "jname_ik_roll", "ik_torso_roll_joint"), roll),
                ]:
                    jid = self.model.getJointId(jn)
                    if 0 < jid < len(self.model.joints):
                        iq = int(self.model.joints[jid].idx_q)
                        if 0 <= iq < self.model.nq:
                            q[iq] = val
        except Exception as e:
            print(f"[IKDBG] virtual joint init failed: {e}")

        return q
    


    def _ensure_frame(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid == len(self.model.frames):
            raise RuntimeError(f"URDF frame not found: {name}")
        return fid

    def _ensure_any_frame(self, names):
        for n in names:
            try:
                return self._ensure_frame(n)
            except Exception:
                continue
        raise RuntimeError(f"None of frames found from candidates: {names}")

    def _joint_id_from_name(self, jname: str) -> int:
        jid = self.model.getJointId(jname)
        if jid == 0:
            raise RuntimeError(f"URDF joint not found: {jname}")
        return jid
    


    # def _build_virtual_ik_urdf(self, urdf_path: str) -> str:
    #     """
    #     基于原始 URDF 生成一个仅用于 IK 的"虚拟"URDF：
    #     - 在 world 与 pelvis 之间插入 1 个竖直 prismatic 关节（h）
    #     - 在 waist_pitch_joint 与 torso_link 之间插入 3 个虚拟旋转关节（yaw/pitch/roll）
    #     返回：新 URDF 路径
    #     """
    #     import xml.etree.ElementTree as ET, os
    #     tree = ET.parse(urdf_path)
    #     robot = tree.getroot()
    #     # 虚拟关节命名（若实例未提前赋值，则使用默认）
    #     jname_ik_pz    = getattr(self, 'jname_ik_pz', 'ik_pelvis_z_joint')
    #     jname_ik_yaw   = getattr(self, 'jname_ik_yaw', 'ik_torso_yaw_joint')
    #     jname_ik_pitch = getattr(self, 'jname_ik_pitch', 'ik_torso_pitch_joint')
    #     jname_ik_roll  = getattr(self, 'jname_ik_roll', 'ik_torso_roll_joint')
    #     # world link
    #     has_world = any((e.tag == 'link' and e.attrib.get('name') == 'world') for e in robot)
    #     if not has_world:
    #         robot.insert(0, ET.Element('link', { 'name': 'world' }))
    #     # prismatic z between world and pelvis
    #     if not any((e.tag == 'joint' and e.attrib.get('name') == jname_ik_pz) for e in robot):
    #         j = ET.Element('joint', { 'name': jname_ik_pz, 'type': 'prismatic' })
    #         ET.SubElement(j, 'origin', { 'xyz': '0 0 0', 'rpy': '0 0 0' })
    #         ET.SubElement(j, 'parent', { 'link': 'world' })
    #         ET.SubElement(j, 'child', { 'link': 'pelvis' })
    #         ET.SubElement(j, 'axis', { 'xyz': '0 0 1' })
    #         ET.SubElement(j, 'limit', { 'lower': '0.20', 'upper': '1.20', 'effort': '1.0', 'velocity': '10.0' })
    #         robot.insert(1, j)
    #     # find waist_pitch_joint（在其与 torso_link 之间插入 3R，枢轴=torso_link 原点）
    #     waist_pitch = None
    #     for e in robot:
    #         if e.tag == 'joint' and e.attrib.get('name') == 'waist_pitch_joint':
    #             waist_pitch = e; break
    #     if waist_pitch is None:
    #         out = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), 'g1_body29_hand14_virtual.urdf')
    #         tree.write(out)
    #         return out
    #     # 将 waist_pitch 的 child 改为 ik_torso_yaw_link
    #     for c in waist_pitch:
    #         if c.tag == 'child':
    #             c.set('link', 'ik_torso_yaw_link'); break
    #     # 确保中间 links 存在
    #     have = set([e.attrib.get('name') for e in robot if e.tag == 'link'])
    #     for lname in ['ik_torso_yaw_link','ik_torso_pitch_link','ik_torso_roll_link']:
    #         if lname not in have:
    #             robot.append(ET.Element('link', { 'name': lname }))
    #     # 添加 yaw/pitch/roll 三关节：yaw(z) -> pitch(y) -> roll(x) -> torso_link
    #     def add_rev(name, parent, child, axis):
    #         if any((e.tag == 'joint' and e.attrib.get('name') == name) for e in robot):
    #             return
    #         j = ET.Element('joint', { 'name': name, 'type': 'revolute' })
    #         ET.SubElement(j, 'origin', { 'xyz': '0 0 0', 'rpy': '0 0 0' })
    #         ET.SubElement(j, 'parent', { 'link': parent })
    #         ET.SubElement(j, 'child', { 'link': child })
    #         ET.SubElement(j, 'axis', { 'xyz': axis })
    #         ET.SubElement(j, 'limit', { 'lower': '-3.1416', 'upper': '3.1416', 'effort': '1.0', 'velocity': '10.0' })
    #         robot.append(j)
    #     add_rev(jname_ik_yaw,   'ik_torso_yaw_link',   'ik_torso_pitch_link', '0 0 1')
    #     add_rev(jname_ik_pitch, 'ik_torso_pitch_link', 'ik_torso_roll_link',  '0 1 0')
    #     add_rev(jname_ik_roll,  'ik_torso_roll_link',  'torso_link',          '1 0 0')

    #     out = os.path.join(os.path.dirname(os.path.abspath(urdf_path)), 'g1_body29_hand14_virtual.urdf')
    #     tree.write(out)
    #     return out
