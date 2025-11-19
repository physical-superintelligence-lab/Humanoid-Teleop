import logging
import numpy as np
import time
import requests
from helpers import RequestMessage, ResponseMessage

logger = logging.getLogger(__name__)

class GrootRemotePolicy:
    def __init__(self, host, port):
        self.url = f"{host}:{port}"

    def pred_action(self, image, state, instruction):
        # Dummy values for fields not used
        gt_action = None
        dataset_paths = []

        request = RequestMessage(
            image=image,
            instruction=instruction,
            history={},
            state=state,
            condition={},
            gt_action=np.array([]) if gt_action is None else gt_action,
            dataset_name=dataset_paths[0] if dataset_paths else "test",
            timestamp="sample_-1",
        )

        print("\n4. Sending request to server...")
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.url}/act",
                json=request.serialize(),
                timeout=60.0,
            )
            elapsed = time.time() - start_time

            response.raise_for_status()
            response_data = response.json()
            response_msg = ResponseMessage.deserialize(response_data)
            pred_action = np.array(response_msg.action)

            return pred_action

        except Exception as e:
            print(f"Error sending request: {e}")
            return None


# --------------------------------------------------------
# Instantiate remote policy
policy = GrootRemotePolicy("http://0.0.0.0", 8001)

# Task instruction
TASK_INSTRUCTION = "fullbody/pick_up_dumpling_toy_and_squat_to_put_on_chair"

# ------------------ FIX image type ----------------------
# Server REQUIRES int type image; 0–255 range
obs_img = np.random.randint(0, 256, size=(1, 480, 640, 3), dtype=np.uint8)

# --------------- Random arm/hand joints -----------------
# 7 DOF per arm, 7 DOF per hand (example)
arm_joints = np.random.randn(14)
hand_joints = np.random.randn(14)

# ------------------- FIX state syntax -------------------
state = {
    "left_arm": arm_joints[0:7],
    "right_arm": arm_joints[7:14],
    "left_hand": hand_joints[0:7],
    "right_hand": hand_joints[7:14],
}

# Call policy
t0 = time.time()
for i in range(10):
    result = policy.pred_action(
        image=obs_img,
        state=state,
        instruction=TASK_INSTRUCTION,
    )
t1 = time.time()


print("Predicted action:", result)
print("total time is", t1-t0)
