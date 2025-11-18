import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
merged_file_path = "data/g1_1001/Basic/pick_up_dumpling_toy_and_squat_to_put_on_chair/episode_10/data.json"
with open(merged_file_path, "r") as f:
    data_list = json.load(f)

# Extract curves from t=0
h_odometry = [frame["states"]["odometry"]["position"][2] for frame in data_list]
h_action = [frame["actions"]["torso_height"] for frame in data_list]

# Plot
plt.figure()
plt.plot(h_odometry, label="Odometry Height (state)")
plt.plot(h_action, label="Torso Height (action)")
plt.xlabel("Timestep")
plt.ylabel("Height (m)")
plt.legend()
plt.title("Height vs Timestep (Odometry vs Action)")
plt.grid(True)
plt.show()
