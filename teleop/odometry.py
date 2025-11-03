import time
from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                         ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.unitree_go.msg.dds.SportModeState_ import SportModeState_

TOPIC_SPORT_STATE = "rt/odommodestate"       # High frequency
TOPIC_SPORT_LF_STATE = "rt/lf/odommodestate" # Low frequency

import threading
import numpy as np

# Dummy logger for this example
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def SetData(self, msg):
        with self.lock:
            self.data = msg

    def GetData(self):
        with self.lock:
            return self.data

class BaseOdometryReader:
    def __init__(self):
        self.stop_event = threading.Event()
        self.odom_buffer = DataBuffer()

        logger.info("Initializing Odometry Reader...")

        # DDS Init
        ChannelFactoryInitialize(0)

        # Subscribe to SportModeState
        self.odom_subscriber = ChannelSubscriber(TOPIC_SPORT_STATE, SportModeState_)
        self.odom_subscriber.Init(self._odom_callback, 1)

        self.subscribe_thread = threading.Thread(target=self._wait_for_data)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

    def _odom_callback(self, msg):
        self.odom_buffer.SetData(msg)

    def _wait_for_data(self):
        while not self.stop_event.is_set():
            if self.odom_buffer.GetData() is None:
                logger.info("Waiting for odometry data...")
                time.sleep(0.1)
            else:
                break

    def get_position(self):
        state = self.odom_buffer.GetData()
        return np.array(state.position) if state else None

    def get_velocity(self):
        state = self.odom_buffer.GetData()
        return np.array(state.velocity) if state else None

    def get_orientation_rpy(self):
        state = self.odom_buffer.GetData()
        return np.array(state.imu_state.rpy) if state else None

    def get_orientation_quaternion(self):
        state = self.odom_buffer.GetData()
        return np.array(state.imu_state.quaternion) if state else None

    def get_yaw_speed(self):
        state = self.odom_buffer.GetData()
        return state.yaw_speed if state else None

    def print_state(self):
        state = self.odom_buffer.GetData()
        if state is None:
            print("No odometry data available.")
            return

        print("Position:", state.position)
        print("Velocity:", state.velocity)
        print("Euler RPY:", state.imu_state.rpy)
        print("Quaternion:", state.imu_state.quaternion)
        print("Yaw speed:", state.yaw_speed)
        print("=" * 40)

    def shutdown(self):
        logger.info("Shutting down odometry reader...")
        self.stop_event.set()
        self.subscribe_thread.join(timeout=1)

# Example usage
if __name__ == "__main__":
    odom = BaseOdometryReader()

    try:
        while True:
            odom.print_state()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        odom.shutdown()
