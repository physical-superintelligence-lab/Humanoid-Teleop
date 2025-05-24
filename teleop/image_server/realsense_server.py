import datetime
import threading
import time  # for sleep control

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# Shared variables for the latest frames
latest_rgb_bytes = None
latest_ir_bytes = None
latest_depth_bytes = None
frame_lock = threading.Lock()


def frame_capture_thread():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
    pipeline.start(config)

    global latest_rgb_bytes, latest_ir_bytes, latest_depth_bytes

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        ir_left_frame = frames.get_infrared_frame(1)
        ir_right_frame = frames.get_infrared_frame(2)

        if not (depth_frame and color_frame and ir_left_frame and ir_right_frame):
            continue

        # Convert frames to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_array = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        ir_right_image = np.asanyarray(ir_right_frame.get_data())

        # Convert single-channel IR images to 3-channel images for consistency
        ir_left_image = cv2.cvtColor(ir_left_image, cv2.COLOR_GRAY2BGR)
        ir_right_image = cv2.cvtColor(ir_right_image, cv2.COLOR_GRAY2BGR)

        # Combine the two IR images horizontally
        ir_combined = np.hstack((ir_left_image, ir_right_image))

        ret_rgb, encoded_rgb = cv2.imencode(".jpg", color_image)
        ret_ir, encoded_ir = cv2.imencode(".jpg", ir_combined)
        if ret_rgb and ret_ir:
            rgb_bytes = encoded_rgb.tobytes()
            ir_bytes = encoded_ir.tobytes()
            depth_bytes = depth_array.tobytes()

            with frame_lock:
                latest_rgb_bytes = rgb_bytes
                latest_ir_bytes = ir_bytes
                latest_depth_bytes = depth_bytes


def start_server():
    # Start the frame capture thread
    capture_thread = threading.Thread(target=frame_capture_thread, daemon=True)
    capture_thread.start()

    context = zmq.Context()
    # Create a REP socket for request-response
    socket = context.socket(zmq.REP)
    socket.bind("tcp://192.168.123.162:5556")
    print("Server started, waiting for client requests...")

    try:
        while True:
            # Wait for a client request
            request = socket.recv()  # blocks until a request is received
            print(
                f"Received request: {request.decode('utf-8')} at {datetime.datetime.now()}"
            )

            with frame_lock:
                rgb_bytes = latest_rgb_bytes
                ir_bytes = latest_ir_bytes
                depth_bytes = latest_depth_bytes

            if rgb_bytes is None or ir_bytes is None or depth_bytes is None:
                print("No frames available yet, sending empty response.")
                socket.send(b"")  # or send an error message if desired
            else:
                # Respond with the latest frames as a multipart message
                socket.send_multipart([rgb_bytes, ir_bytes, depth_bytes])
                print(f"Sent frame at {datetime.datetime.now()}")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    start_server()
