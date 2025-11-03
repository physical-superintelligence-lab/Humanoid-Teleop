In each timestep:
14 dof arm: data["states"]["arm_state"] 7 dof left arm + 7 dof right arm
14 dof hand: data["states"]["hand_state"] 7 left fingers + 7 right fingers
3 dof torso rpy: data["states"]["imu"]["rpy"]
1 dof torso height: data["states"]["odometry"]["position"][2]

15 dof leg: data["states"]["leg_state"] 6 dof left leg + 6 dof right leg + 3 dof waist


Recommend using 14 dof arm + 14 dof hand + 3 dof rpy + 1 dof height for training inputs and outputs.

Can get legstate through amo policy inference.