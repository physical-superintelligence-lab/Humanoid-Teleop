import cv2
import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.123.162:5556")

print("Client connected")

while True:
    socket.send(b"get")

    reply = socket.recv_multipart()
    print("Received parts:", len(reply))

    rgb_bytes = reply[0]
    print("RGB bytes:", len(rgb_bytes))

    img = cv2.imdecode(
        np.frombuffer(rgb_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    print("Decode failed?", img is None)

    if img is not None:
        cv2.imshow("RGB", img)
        cv2.waitKey(1)
