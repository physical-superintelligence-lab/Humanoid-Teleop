import datetime
import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

# Shared variables
latest_rgb_bytes = None
latest_ir_bytes = None
# Pre-generate a fake depth buffer (zeros) to maintain pipeline compatibility
# 640x480 uint16 = 614,400 bytes
FAKE_DEPTH_BYTES = np.zeros((480, 640), dtype=np.uint16).tobytes()
frame_lock = threading.Lock()

def frame_capture_thread():
    global latest_rgb_bytes, latest_ir_bytes
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable ONLY RGB and IR (to fit USB 2.0 bandwidth)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

    try:
        pipeline.start(config)
        print("RealSense: RGB + IR active. Depth is MOCKED (zeros) for USB 2.0 compatibility.")
    except Exception as e:
        print(f"Failed to start RealSense: {e}")
        return

    while True:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            ir_left_frame = frames.get_infrared_frame(1)
            ir_right_frame = frames.get_infrared_frame(2)

            if not (color_frame and ir_left_frame and ir_right_frame):
                continue

            # Process Color
            color_image = np.asanyarray(color_frame.get_data())
            # Use JPEG quality 80 for USB 2.0 stability
            _, encoded_rgb = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            # Process IR
            ir_l = np.asanyarray(ir_left_frame.get_data())
            ir_r = np.asanyarray(ir_right_frame.get_data())
            # Convert to BGR to match your original processing logic
            ir_l_bgr = cv2.cvtColor(ir_l, cv2.COLOR_GRAY2BGR)
            ir_r_bgr = cv2.cvtColor(ir_r, cv2.COLOR_GRAY2BGR)
            ir_combined = np.hstack((ir_l_bgr, ir_r_bgr))
            _, encoded_ir = cv2.imencode(".jpg", ir_combined, [cv2.IMWRITE_JPEG_QUALITY, 60])

            with frame_lock:
                latest_rgb_bytes = encoded_rgb.tobytes()
                latest_ir_bytes = encoded_ir.tobytes()

        except Exception as e:
            print(f"Capture error: {e}")

def start_server():
    threading.Thread(target=frame_capture_thread, daemon=True).start()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://192.168.123.164:5556")
    print("Server started, waiting for client requests...")

    try:
        while True:
            cur = time.time()
            request = socket.recv()
            print(f"req time: {time.time() - cur}")
            cur = time.time()
            with frame_lock:
                rgb = latest_rgb_bytes
                ir = latest_ir_bytes

            if rgb is None or ir is None:
                socket.send(b"")
            else:
                # Send RGB, IR, and the FAKE depth zeros
                socket.send_multipart([rgb, ir, FAKE_DEPTH_BYTES])
                print(f"send time: {time.time() - cur}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    start_server()
