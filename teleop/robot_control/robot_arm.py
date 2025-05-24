import threading
import time

import numpy as np
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_._SportModeState_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from robot_control.robot_arm_joints import (
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    H1_2_JointArmIndex,
    H1_2_JointIndex,
)
from utils.logger import logger

kTopicLowCommand = "rt/arm_sdk"
# kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
TOPIC_SPORT_STATE = "rt/odommodestate"
G1_29_Num_Motors = 35
H1_2_Num_Motors = 35


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None


class LowState:
    def __init__(self, num_motors):
        self.motor_state = [MotorState() for _ in range(num_motors)]


class G1_29_LowState(LowState):
    def __init__(self):
        super().__init__(G1_29_Num_Motors)


class H1_2_LowState(LowState):
    def __init__(self):
        super().__init__(H1_2_Num_Motors)


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class BaseArmController:
    """Base class for arm controllers with common functionality."""

    def __init__(self, robot_type, num_motors, joint_index_enum, joint_arm_index_enum):
        self.robot_type = robot_type
        self.num_motors = num_motors
        self.JointIndex = joint_index_enum
        self.JointArmIndex = joint_arm_index_enum
        self.stop_event = threading.Event()

        logger.info(f"Initialize {self.robot_type}_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        # Control parameters
        self.kp_high = 200.0
        self.kd_high = 5.0
        self.kp_low = 100.0
        self.kd_low = 7.5
        self.kp_wrist = 30.0
        self.kd_wrist = 6.0

        self.all_motor_q = None
        self.arm_velocity_limit = 30.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        # Initialize DDS communication
        ChannelFactoryInitialize(0)
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand, LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()
        # Subscribe odometry information
        self.odom_buffer = DataBuffer()
        self.odom_subscriber = ChannelSubscriber(TOPIC_SPORT_STATE, SportModeState_)
        self.odom_subscriber.Init(self._odom_callback, 1)

        # Initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            logger.info(
                f"[{self.robot_type}_ArmController] Waiting to subscribe dds..."
            )

        # Initialize command message
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        # print(f"Current all body motor state q:\n{self.all_motor_q} \n")
        # print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger.info("Lock all joints except two arms...\n")

        # Set motor control parameters
        self.ctrl_lock = threading.Lock()
        self._setup_motor_params()
        logger.info("Lock OK!\n")

        # Initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger.info(f"Initialize {self.robot_type}_ArmController OK!\n")

    def _odom_callback(self, msg):
        self.odom_buffer.SetData(msg)

    def get_odom_data(self):
        state = self.odom_buffer.GetData()
        if not state:
            return None

        return {
            "position": np.array(state.position),
            "velocity": np.array(state.velocity),
            "orientation_rpy": np.array(state.imu_state.rpy),
            "orientation_quaternion": np.array(state.imu_state.quaternion),
        }

    def _setup_motor_params(self):
        """Set up motor parameters based on motor type."""
        arm_indices = set(member.value for member in self.JointArmIndex)
        for id in self.JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]

    def _subscribe_motor_state(self):
        """Thread to subscribe to motor state updates."""
        while not self.stop_event.is_set():
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = (
                    G1_29_LowState() if self.robot_type == "G1_29" else H1_2_LowState()
                )
                for id in range(self.num_motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        """Limit movement speed for safety."""
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / ((velocity_limit * self.control_dt))
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        """Thread to send control commands to motors."""
        while not self.stop_event.is_set():
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            cliped_arm_q_target = self.clip_arm_q_target(
                arm_q_target, velocity_limit=self.arm_velocity_limit
            )

            for idx, id in enumerate(self.JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target, is_first=False):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array(
            [self.lowstate_buffer.GetData().motor_state[id].q for id in self.JointIndex]
        )

    def get_imu_data(self):
        """Get IMU data from the robot."""
        return self.lowstate_subscriber.Read().imu_state

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array(
            [
                self.lowstate_buffer.GetData().motor_state[id].q
                for id in self.JointArmIndex
            ]
        )

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array(
            [
                self.lowstate_buffer.GetData().motor_state[id].dq
                for id in self.JointArmIndex
            ]
        )

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position."""
        logger.info(f"[{self.robot_type}_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles
        while not self.stop_event.is_set():
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                logger.info(
                    f"[{self.robot_type}_ArmController] both arms have reached the home position."
                )
                break
            time.sleep(0.1)

    def speed_gradual_max(self, t=7.0):
        """Gradually increase arm velocity to maximum over time."""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """Set arms velocity to the maximum value immediately."""
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def _Is_wrist_motor(self, motor_index):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def set_weight_to_1(self):
        with self.ctrl_lock:
            self.msg.motor_cmd[29].q = 1
            self.msg.motor_cmd[30].q = 1
            self.msg.motor_cmd[31].q = 1
            self.msg.motor_cmd[32].q = 1
            self.msg.motor_cmd[33].q = 1
            self.msg.motor_cmd[34].q = 1

    def gradually_set_weight_to_0(self):
        for i in range(500, 0, -1):
            self.msg.motor_cmd[29].q = i / 500
            self.msg.motor_cmd[30].q = i / 500
            self.msg.motor_cmd[31].q = i / 500
            self.msg.motor_cmd[32].q = i / 500
            self.msg.motor_cmd[33].q = i / 500
            self.msg.motor_cmd[34].q = i / 500
            time.sleep(0.01)

    def shutdown(self):
        # self.gradually_set_weight_to_0()
        """Shutdown controller and clean up threads."""
        logger.info("controller: shutting down threads")
        self.stop_event.set()
        self.publish_thread.join(timeout=1)
        self.subscribe_thread.join(timeout=1)
        logger.info("controller: shut down")

    def reset(self):
        """Reset controller to initial state."""
        logger.info("controller: resetting")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self.lowstate_buffer = DataBuffer()
        self.stop_event.clear()

        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            logger.info(
                f"[{self.robot_type}_ArmController] Waiting to subscribe dds..."
            )

        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        logger.info("controller: reset")


class G1_29_ArmController(BaseArmController):
    def __init__(self):
        super().__init__(
            "G1_29", G1_29_Num_Motors, G1_29_JointIndex, G1_29_JointArmIndex
        )

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors


class H1_2_ArmController(BaseArmController):
    def __init__(self):
        super().__init__("H1_2", H1_2_Num_Motors, H1_2_JointIndex, H1_2_JointArmIndex)

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            H1_2_JointIndex.kLeftAnkle.value,
            H1_2_JointIndex.kRightAnkle.value,
            # Left arm
            H1_2_JointIndex.kLeftShoulderPitch.value,
            H1_2_JointIndex.kLeftShoulderRoll.value,
            H1_2_JointIndex.kLeftShoulderYaw.value,
            H1_2_JointIndex.kLeftElbowPitch.value,
            # Right arm
            H1_2_JointIndex.kRightShoulderPitch.value,
            H1_2_JointIndex.kRightShoulderRoll.value,
            H1_2_JointIndex.kRightShoulderYaw.value,
            H1_2_JointIndex.kRightElbowPitch.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            H1_2_JointIndex.kLeftElbowRoll.value,
            H1_2_JointIndex.kLeftWristPitch.value,
            H1_2_JointIndex.kLeftWristyaw.value,
            H1_2_JointIndex.kRightElbowRoll.value,
            H1_2_JointIndex.kRightWristPitch.value,
            H1_2_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors


if __name__ == "__main__":
    import pinocchio as pin
    from robot_arm_ik import G1_29_ArmIK, H1_2_ArmIK

    # Choose robot type
    robot_type = "H1_2"  # Change to "G1_29" to use the G1_29 robot

    if robot_type == "G1_29":
        arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
        arm = G1_29_ArmController()
    else:
        arm_ik = H1_2_ArmIK(Unit_Test=True, Visualization=False)
        arm = H1_2_ArmController()

    # Initial position
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.1]),
    )

    rotation_speed = 0.005  # Rotation speed in radians per iteration
    q_target = np.zeros(35)
    tauff_target = np.zeros(35)

    user_input = input(
        "Please enter the start signal (enter 's' to start the subsequent program): \n"
    )
    if user_input.lower() == "s":
        step = 0
        arm.speed_gradual_max()
        while True:
            if step <= 120:
                angle = rotation_speed * step
                L_quat = pin.Quaternion(
                    np.cos(angle / 2), 0, np.sin(angle / 2), 0
                )  # y axis
                R_quat = pin.Quaternion(
                    np.cos(angle / 2), 0, 0, np.sin(angle / 2)
                )  # z axis

                L_tf_target.translation += np.array([0.001, 0.001, 0.001])
                R_tf_target.translation += np.array([0.001, -0.001, 0.001])
            else:
                angle = rotation_speed * (240 - step)
                L_quat = pin.Quaternion(
                    np.cos(angle / 2), 0, np.sin(angle / 2), 0
                )  # y axis
                R_quat = pin.Quaternion(
                    np.cos(angle / 2), 0, 0, np.sin(angle / 2)
                )  # z axis

                L_tf_target.translation -= np.array([0.001, 0.001, 0.001])
                R_tf_target.translation -= np.array([0.001, -0.001, 0.001])

            L_tf_target.rotation = L_quat.toRotationMatrix()
            R_tf_target.rotation = R_quat.toRotationMatrix()

            current_lr_arm_q = arm.get_current_dual_arm_q()
            current_lr_arm_dq = arm.get_current_dual_arm_dq()

            sol_q, sol_tauff = arm_ik.solve_ik(
                L_tf_target.homogeneous,
                R_tf_target.homogeneous,
                current_lr_arm_q,
                current_lr_arm_dq,
            )

            arm.ctrl_dual_arm(sol_q, sol_tauff)

            step += 1
            if step > 240:
                step = 0
            time.sleep(0.01)
