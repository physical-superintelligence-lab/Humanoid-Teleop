import json
import os
import queue
import sys
import threading

import cv2

FREQ = 30
DELAY = 1 / FREQ

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class AsyncImageWriter:
    def __init__(self):
        self.queue = queue.Queue()
        self.kill_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.kill_event.is_set() or not self.queue.empty():
            try:
                filename, image_array = self.queue.get(timeout=0.5)
                rgb_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                # print("rgb_image shape is", rgb_image.shape)
                # img = np.frombuffer(image_array, dtype=np.uint8).reshape((480, 640, 3))
                cv2.imwrite(filename, rgb_image)
            except queue.Empty:
                continue

    def write_image(self, filename, image_array):
        self.queue.put((filename, image_array))

    def close(self):
        self.kill_event.set()
        self.thread.join()


class AsyncWriter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.queue = queue.Queue()
        self.kill_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with open(self.filepath, "a") as f:
            while not self.kill_event.is_set() or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.5)
                    # logger.debug(f"async writer: writing elements {item}")
                    f.write(item + "\n")
                    # f.flush()
                except queue.Empty:
                    continue

    def write(self, item):
        self.queue.put(item)

    def close(self):
        self.kill_event.set()
        self.thread.join()


class IKDataWriter:
    def __init__(self, dirname):
        self.buffer = []
        self.async_writer = AsyncWriter(os.path.join(dirname, "ik_data.jsonl"))

    def write_data(
        self,
        right_angles,
        left_angles,
        arm_time,
        ik_time,
        sol_q,
        tau_ff,
        head_rmat,
        left_pose,
        right_pose,
    ):
        entry = {
            "right_angles": right_angles.tolist(),
            "left_angles": left_angles.tolist(),
            "armtime": arm_time,
            "iktime": ik_time,
            "sol_q": sol_q.tolist(),
            "tau_ff": tau_ff.tolist(),
            "head_rmat": head_rmat.tolist(),
            "left_pose": left_pose.tolist(),
            "right_pose": right_pose.tolist(),
        }
        self.async_writer.write(json.dumps(entry))

    def close(self):
        self.async_writer.close()
