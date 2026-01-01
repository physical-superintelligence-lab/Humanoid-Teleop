import cv2
import zmq
import numpy as np

def start_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    
    # 注意：请确保此处的 IP 地址与服务端 IP (192.168.123.164) 一致
    server_ip = "192.168.123.164" 
    socket.connect(f"tcp://{server_ip}:5556")

    print(f"Connected to server at {server_ip}")
    print("Press 'q' to exit.")

    try:
        while True:
            # 发送请求
            socket.send(b"get")

            # 接收多部分消息 [RGB, IR, DEPTH]
            reply = socket.recv_multipart()
            
            if len(reply) < 3:
                continue

            rgb_bytes = reply[0]
            ir_bytes = reply[1]
            # depth_bytes = reply[2] # 暂不处理 mock 的深度数据

            # 1. 解码 RGB 图像
            rgb_img = cv2.imdecode(
                np.frombuffer(rgb_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            # 2. 解码 IR 图像 (服务端将左右红外合并成了一张 JPEG)
            ir_img = cv2.imdecode(
                np.frombuffer(ir_bytes, np.uint8),
                cv2.IMREAD_COLOR
            )

            # 检查解码是否成功
            if rgb_img is not None:
                cv2.imshow("RealSense RGB", rgb_img)
            
            if ir_img is not None:
                cv2.imshow("RealSense IR (Left | Right)", ir_img)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Client error: {e}")
    finally:
        cv2.destroyAllWindows()
        socket.close()
        context.term()

if __name__ == "__main__":
    start_client()