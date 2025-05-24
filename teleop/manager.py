import logging
import os
import re
import shutil
import sys
from multiprocessing import Event, Lock, Manager, Process, Queue, shared_memory
from pathlib import Path

import numpy as np

from constants import *
from master import RobotTaskmaster
from progress import ProgressTracker
from utils.logger import logger
from worker import RobotDataWorker

FREQ = 30
DELAY = 1 / FREQ

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class TeleopManager:
    def __init__(self, task_name="default_task", robot="h1", debug=False):
        self.task_name = task_name
        logger.info(f"#### (Task: {self.task_name}):")
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        self.manager = Manager()
        self.shared_dict = self.manager.dict()

        self.shared_dict["kill_event"] = self.manager.Event()
        self.shared_dict["session_start_event"] = self.manager.Event()
        self.shared_dict["failure_event"] = self.manager.Event()
        self.shared_dict["end_event"] = self.manager.Event()  # TODO: redundent
        self.progress_tracker = ProgressTracker()

        if robot == "h1":
            totalsize = (
                H1_sizes.LEG_STATE_SIZE
                + H1_sizes.ARM_STATE_SIZE
                + H1_sizes.HAND_STATE_SIZE
                + H1_sizes.IMU_QUATERNION_SIZE
                + H1_sizes.IMU_ACCELEROMETER_SIZE
                + H1_sizes.IMU_GYROSCOPE_SIZE
                + H1_sizes.IMU_RPY_SIZE
                # + H1_sizes.ODOM_POSITION_SIZE
                # + H1_sizes.ODOM_VELOCITY_SIZE
                # + H1_sizes.ODOM_RPY_SIZE
                # + H1_sizes.ODOM_QUATERNION_SIZE
            )
        elif robot == "g1":
            totalsize = (
                G1_sizes.LEG_STATE_SIZE
                + G1_sizes.ARM_STATE_SIZE
                + G1_sizes.HAND_STATE_SIZE
                + G1_sizes.IMU_QUATERNION_SIZE
                + G1_sizes.IMU_ACCELEROMETER_SIZE
                + G1_sizes.IMU_GYROSCOPE_SIZE
                + G1_sizes.IMU_RPY_SIZE
                + G1_sizes.ODOM_POSITION_SIZE
                + G1_sizes.ODOM_VELOCITY_SIZE
                + G1_sizes.ODOM_RPY_SIZE
                + G1_sizes.ODOM_QUATERNION_SIZE
                + G1_sizes.HAND_PRESS_SIZE
            )

        self.robot_data_shm = shared_memory.SharedMemory(
            create=True, size=totalsize * np.dtype(np.float64).itemsize
        )
        self.robot_shm_array = np.ndarray(
            (totalsize,), dtype=np.float64, buffer=self.robot_data_shm.buf
        )

        self.teleop_shm = shared_memory.SharedMemory(
            create=True, size=55 * np.dtype(np.float64).itemsize
        )
        self.teleop_shm_array = np.ndarray(
            (55,), dtype=np.float64, buffer=self.teleop_shm.buf
        )

        def run_taskmaster():
            taskmaster = RobotTaskmaster(
                self.task_name,
                self.shared_dict,
                self.robot_shm_array,
                self.teleop_shm_array,
                robot,
            )
            taskmaster.start()

        def run_dataworker():
            taskworker = RobotDataWorker(
                self.shared_dict, self.robot_shm_array, self.teleop_shm_array, robot
            )
            taskworker.start()

        self.taskmaster_proc = Process(target=run_taskmaster)
        self.dataworker_proc = Process(target=run_dataworker)

    def start_processes(self):
        logger.info("Starting taskmaster and dataworker processes.")
        self.taskmaster_proc.start()
        self.dataworker_proc.start()

    def update_directory(self):
        self.shared_dict["dirname"] = self.progress_tracker.get_next()
        os.makedirs(self.shared_dict["dirname"], exist_ok=True)
        os.makedirs(os.path.join(self.shared_dict["dirname"], "color"), exist_ok=True)
        os.makedirs(os.path.join(self.shared_dict["dirname"], "depth"), exist_ok=True)
        # dirname = self.shared_dict["dirname"]

        # logger.info(f"Data directory set to: {dirname}")

    def start_session(self):
        self.update_directory()
        self.shared_dict["failure_event"].clear()
        self.shared_dict["kill_event"].clear()
        self.shared_dict["session_start_event"].set()
        logger.info("Session started.")

    def stop_session(self):
        self.shared_dict["kill_event"].set()
        self.shared_dict["session_start_event"].clear()
        logger.info("Session stopped.")

    def cleanup(self):
        logger.info("Cleaning up processes and shared resources...")
        self.shared_dict["end_event"].set()
        self.shared_dict["kill_event"].set()
        self.shared_dict["session_start_event"].set()
        self.taskmaster_proc.join(timeout=10)
        self.dataworker_proc.join(timeout=10)

        if self.taskmaster_proc.is_alive():
            logger.warning("Forcing termination of taskmaster process.")
            self.taskmaster_proc.terminate()
            self.taskmaster_proc.join(timeout=2)

        if self.dataworker_proc.is_alive():
            logger.warning("Forcing termination of dataworker process.")
            self.dataworker_proc.terminate()
            self.dataworker_proc.join(timeout=2)

        self.manager.shutdown()

        self.robot_data_shm.close()
        self.robot_data_shm.unlink()
        self.teleop_shm.close()
        self.teleop_shm.unlink()
        logger.info("Cleanup complete.")

    def run_command_loop(self):
        last_cmd = None
        logger.info(
            "Press 's' to start, 'q' to stop/merge, 'd' for a failure case, 'exit' to quit."
        )
        try:
            while True:
                user_input = input("> ").lower()
                if user_input == "s" and last_cmd != "s":
                    self.start_session()
                    dirname = self.shared_dict["dirname"]
                    logger.info(f"Current task: {dirname}")
                    last_cmd = "s"
                elif user_input == "q":
                    self.stop_session()
                    last_cmd = "q"
                elif user_input == "d":
                    self.shared_dict["failure_event"].set()
                    self.stop_session()
                    last_cmd = "d"
                elif user_input == "exit":
                    self.cleanup()
                    sys.exit(0)
                else:
                    logger.info(
                        "Invalid command. Use 's' to start, 'q' to stop/merge, 'exit' to quit."
                    )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected. Exiting...")
            self.cleanup()
            sys.exit(0)
