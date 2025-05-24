import os
import signal
import subprocess
import sys

from utils.logger import logger

FREQ = 30
DELAY = 1 / FREQ

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class LidarProcess:
    def __init__(self, dirname) -> None:
        self.program_cmd = [
            "./point_cloud_recorder",
            "./mid360_config.json",
            dirname + "/lidar",
        ]
        self.proc = None

    def run(self):
        self.proc = subprocess.Popen(
            self.program_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info("LidarProcess started.")

    def cleanup(self):
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:  # if the process is still running
                logger.info("Sending SIGINT to the lidar process...")
                self.proc.send_signal(signal.SIGINT)
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()  # force kill after timeout
                    logger.info("Lidar process killed after timeout.")
        except Exception as e:
            logger.error(f"Error cleaning up lidar process: {e}")
