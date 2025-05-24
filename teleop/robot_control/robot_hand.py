import threading
import time
from enum import Enum
from pathlib import Path

import numpy as np
import yaml
from unitree_dds_wrapper.idl import unitree_go
from unitree_dds_wrapper.publisher import Publisher
from unitree_dds_wrapper.subscription import Subscription

from utils.logger import logger

from .dex_retargeting.retargeting_config import RetargetingConfig


class HandType(Enum):
    INSPIRE_HAND = "../assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "../assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "../../assets/unitree_hand/unitree_dex3.yml"


class HandRetargeting:
    def __init__(self, hand_type: HandType):
        # Set the default URDF directory based on hand type.
        if hand_type == HandType.UNITREE_DEX3:
            RetargetingConfig.set_default_urdf_dir("../assets")
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            RetargetingConfig.set_default_urdf_dir("../../assets")
        elif hand_type == HandType.INSPIRE_HAND:
            RetargetingConfig.set_default_urdf_dir("../assets")

        config_file_path = Path(hand_type.value)

        try:
            with config_file_path.open("r") as f:
                self.cfg = yaml.safe_load(f)

            if "left" not in self.cfg or "right" not in self.cfg:
                raise ValueError(
                    "Configuration file must contain 'left' and 'right' keys."
                )

            # For Inspire hand the YAML sets target_joint_names to null.
            # Here we fix that by explicitly specifying the target joint names.
            expected_target_joints_left = [
                "L_index_proximal_joint",
                "L_index_intermediate_joint",
                "L_middle_proximal_joint",
                "L_middle_intermediate_joint",
                "L_pinky_proximal_joint",
                "L_pinky_intermediate_joint",
                "L_ring_proximal_joint",
                "L_ring_intermediate_joint",
                "L_thumb_proximal_yaw_joint",
                "L_thumb_proximal_pitch_joint",
                "L_thumb_intermediate_joint",
                "L_thumb_distal_joint",
            ]
            if not self.cfg["left"].get("target_joint_names"):
                logger.warning(
                    "Left target_joint_names not specified. Using default: {}".format(
                        expected_target_joints_left
                    )
                )
                self.cfg["left"]["target_joint_names"] = expected_target_joints_left

            expected_target_joints_right = [
                name.replace("L_", "R_") for name in expected_target_joints_left
            ]
            if not self.cfg["right"].get("target_joint_names"):
                logger.warning(
                    "Right target_joint_names not specified. Using default: {}".format(
                        expected_target_joints_right
                    )
                )
                self.cfg["right"]["target_joint_names"] = expected_target_joints_right

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg["left"])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg["right"])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names

            # For Inspire hand, use the target_joint_names from the config as the API joint names.
            if hand_type == HandType.INSPIRE_HAND:
                self.left_dex3_api_joint_names = self.cfg["left"]["target_joint_names"]
                self.right_dex3_api_joint_names = self.cfg["right"][
                    "target_joint_names"
                ]
            else:
                # Legacy dex3 API joint names for other hand types.
                self.left_dex3_api_joint_names = [
                    "left_hand_thumb_0_joint",
                    "left_hand_thumb_1_joint",
                    "left_hand_thumb_2_joint",
                    "left_hand_middle_0_joint",
                    "left_hand_middle_1_joint",
                    "left_hand_index_0_joint",
                    "left_hand_index_1_joint",
                ]
                self.right_dex3_api_joint_names = [
                    "right_hand_thumb_0_joint",
                    "right_hand_thumb_1_joint",
                    "right_hand_thumb_2_joint",
                    "right_hand_middle_0_joint",
                    "right_hand_middle_1_joint",
                    "right_hand_index_0_joint",
                    "right_hand_index_1_joint",
                ]

            # Build the mapping from dex-retargeting API joints to hardware (target) joints.
            self.left_dex_retargeting_to_hardware = [
                self.left_retargeting_joint_names.index(name)
                for name in self.left_dex3_api_joint_names
            ]
            self.right_dex_retargeting_to_hardware = [
                self.right_retargeting_joint_names.index(name)
                for name in self.right_dex3_api_joint_names
            ]

            # Archive: This is the joint order of the dex-retargeting library version 0.1.1.
            # For example:
            # print([joint.get_name() for joint in self.left_retargeting.optimizer.robot.get_active_joints()])
            # ['left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
            #  'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
            #  'left_hand_index_0_joint', 'left_hand_index_1_joint']
            # Similarly for the right hand.

        except FileNotFoundError:
            print(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            print(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise


inspire_tip_indices = [4, 9, 14, 19, 24]


class InspireController:
    def __init__(self):
        self.cmd = unitree_go.msg.dds_.MotorCmds_()
        self.state = unitree_go.msg.dds_.MotorStates_()
        self.lock = threading.Lock()
        self.handcmd = Publisher(unitree_go.msg.dds_.MotorCmds_, "rt/inspire/cmd")
        self.handstate = Subscription(
            unitree_go.msg.dds_.MotorStates_, "rt/inspire/state"
        )
        self.cmd.cmds = [unitree_go.msg.dds_.MotorCmd_() for _ in range(12)]
        self.state.states = [unitree_go.msg.dds_.MotorState_() for _ in range(12)]

        self.stop_event = threading.Event()
        self.subscribe_state_thread = threading.Thread(target=self.subscribe_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)

    def subscribe_state(self):
        while not self.stop_event.is_set():
            if self.handstate.msg:
                self.state = self.handstate.msg
            time.sleep(0.01)

    def ctrl(self, left_angles, right_angles):
        for i in range(6):
            self.cmd.cmds[i].q = left_angles[i]
            self.cmd.cmds[i + 6].q = right_angles[i]
        self.handcmd.msg.cmds = self.cmd.cmds
        self.handcmd.write()

    def get_current_dual_hand_q(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(12)])
            return q

    def get_right_q(self):
        with self.lock:
            q = np.array([self.state.states[i].q for i in range(6)])
            return q

    def get_left_q(self):
        with self.lock:
            q = np.array([self.state.states[i + 6].q for i in range(6)])
            return q

    def shutdown(self):
        self.stop_event.set()
        self.subscribe_state_thread.join()

    def reset(self):
        if self.stop_event.is_set():
            self.stop_event.clear()
        self.subscribe_state_thread = threading.Thread(target=self.subscribe_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()
        print("H1HandController has been reset.")
