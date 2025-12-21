class Modality:
    def __init__(self, robot_type: str, order_array=None) -> None:
        self.robot_type = robot_type.lower()
        if self.robot_type == "g1":
            self.modality_sizes_dict = self._build_g1_sizes()
        elif self.robot_type == "h1":
            self.modality_sizes_dict = self._build_h1_sizes()
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        self.order_array = order_array or self._default_order()
        self.modality_start_indices = self._compute_start_indices()

    def _default_order(self):
        base_order = [
            "leg",
            "arm",
            "hand",
            "imu_quaternion",
            "imu_accelerometer",
            "imu_gyroscope",
            "imu_rpy",
            "odom_position",
            "odom_velocity",
            "odom_rpy",
            "odom_quaternion",
        ]
        if self.robot_type == "g1":
            base_order.append("hand_press")
        return base_order

    def _build_g1_sizes(self):
        return {
            "leg": G1_sizes.LEG_STATE_SIZE,
            "arm": G1_sizes.ARM_STATE_SIZE,
            "hand": G1_sizes.HAND_STATE_SIZE,
            "imu_quaternion": G1_sizes.IMU_QUATERNION_SIZE,
            "imu_accelerometer": G1_sizes.IMU_ACCELEROMETER_SIZE,
            "imu_gyroscope": G1_sizes.IMU_GYROSCOPE_SIZE,
            "imu_rpy": G1_sizes.IMU_RPY_SIZE,
            "odom_position": G1_sizes.ODOM_POSITION_SIZE,
            "odom_velocity": G1_sizes.ODOM_VELOCITY_SIZE,
            "odom_rpy": G1_sizes.ODOM_RPY_SIZE,
            "odom_quaternion": G1_sizes.ODOM_QUATERNION_SIZE,
            "hand_press": G1_sizes.HAND_PRESS_SIZE,
        }

    def _build_h1_sizes(self):
        return {
            "leg": H1_sizes.LEG_STATE_SIZE,
            "arm": H1_sizes.ARM_STATE_SIZE,
            "hand": H1_sizes.HAND_STATE_SIZE,
            "imu_quaternion": H1_sizes.IMU_QUATERNION_SIZE,
            "imu_accelerometer": H1_sizes.IMU_ACCELEROMETER_SIZE,
            "imu_gyroscope": H1_sizes.IMU_GYROSCOPE_SIZE,
            "imu_rpy": H1_sizes.IMU_RPY_SIZE,
            # "odom_position": H1_sizes.ODOM_POSITION_SIZE,
            # "odom_velocity": H1_sizes.ODOM_VELOCITY_SIZE,
            # "odom_rpy": H1_sizes.ODOM_RPY_SIZE,
            # "odom_quaternion": H1_sizes.ODOM_QUATERNION_SIZE,
        }

    def _compute_start_indices(self) -> dict:
        start_indices = {}
        current_index = 0
        for modality in self.order_array:
            start_indices[modality] = current_index
            current_index += self.modality_sizes_dict[modality]
        return start_indices

    def get_start_index(self, modality_name: str) -> int:
        assert (
            modality_name in self.modality_start_indices
        ), f"Modality {modality_name} not found."
        start_idx = self.modality_start_indices[modality_name]
        return start_idx

    def get_end_index(self, modality_name: str) -> int:
        start_index = self.get_start_index(modality_name)
        return start_index + self.modality_sizes_dict[modality_name]

    def has_modality(self, modality_name: str) -> bool:
        return modality_name in self.modality_sizes_dict


class G1_sizes:
    LEG_STATE_SIZE = 15
    ARM_STATE_SIZE = 14
    HAND_STATE_SIZE = 14
    IMU_QUATERNION_SIZE = 4
    IMU_ACCELEROMETER_SIZE = 3
    IMU_GYROSCOPE_SIZE = 3
    IMU_RPY_SIZE = 3
    HAND_PRESS_SIZE = 216
    ODOM_POSITION_SIZE = 3
    ODOM_VELOCITY_SIZE = 3
    ODOM_RPY_SIZE = 3
    ODOM_QUATERNION_SIZE = 4


class H1_sizes:
    LEG_STATE_SIZE = 13
    ARM_STATE_SIZE = 14
    HAND_STATE_SIZE = 12
    IMU_QUATERNION_SIZE = 4
    IMU_ACCELEROMETER_SIZE = 3
    IMU_GYROSCOPE_SIZE = 3
    IMU_RPY_SIZE = 3
    # ODOM_POSITION_SIZE = 3
    # ODOM_VELOCITY_SIZE = 3
    # ODOM_RPY_SIZE = 3
    # ODOM_QUATERNION_SIZE = 4
