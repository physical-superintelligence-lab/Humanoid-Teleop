# Introduction

This repository contains USC GVL's custom Humanoid data collection pipeline, along with the re-engineering of Unitree's teleoperation of the G1 humanoid robots. For teleoperating and collecting data using the H1, please check out the H1 branch.

# Environment Requirements

We use micromamba instead of conda for its lightweight.

## Inverse Kinematics Libraries (Host only)

```bash
micromamba create -n tv python=3.8
micromamba activate tv
micromamba install pinocchio -c conda-forge
pip install meshcat
pip install casadi
```

## Install Unitree SDK2 Python (Host only)

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

> [!NOTE]
> You need to fix the dds issue by manually removing any code relevant to \_\_domain\_\_

## TeleVision and Apple Vision Pro configuration (Host only)

```bash
pip install -r requirements.txt
```

## Apple Vision Pro (AVP) Setup Local streaming (Host and AVP)

**Apple** does not allow WebXR on non-https connections. To test the application locally, we need to create a self-signed certificate and install it on the client. You need a ubuntu machine and a router. Connect the VisionPro and the ubuntu machine to the same router.

1. install mkcert: https://github.com/FiloSottile/mkcert
2. check local ip address:

```bash
ifconfig | grep inet
```

Suppose the local ip address of the ubuntu machine is `192.168.123.2`

3. create certificate:

```bash
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.123.2 localhost 127.0.0.1
```

ps. place the generated cert.pem and key.pem files in `teleop`.

```bash
cp cert.pem key.pem teleop/
```

4. open firewall on server:

```bash
sudo ufw allow 8012
```

5. install ca-certificates on VisionPro:

```
mkcert -CAROOT
```

Copy the `rootCA.pem` via AirDrop to VisionPro and install it.

Settings > General > About > Certificate Trust Settings. Under "Enable full trust for root certificates", turn on trust for the certificate.

settings > Apps > Safari > Advanced > Feature Flags > Enable WebXR Related Features

6. open the browser on Safari on VisionPro and go to https://192.168.123.2:8012?ws=wss://192.168.123.2:8012

7. Click `Enter VR` and `Allow` to start the VR session.

## Image Server environment (Robot only)

On G1's PC, install the image server's requirements

```bash
micromamba create -n vision python=3.8
micromamba activate vision
pip install pyrealsense2 opencv-python zmq numpy
```

## Lidar (Host only)

Compile and lidar's C++ point cloud recorder program

```bash
mkdir Livox-SDK2/build && cd Livox-SDK2/build
cmake ..
make -j
mv point_cloud_recordedr/point_cloud_recorder ../../avp_teleop/teleop
```

# Start teleoperating and data collection

## Image Server

Copy `realsense_server.py` in the `teleop/image_server` directory to the PC of Unitree G_1, and execute the following command **in the PC**:

```bash
micromamba activate vision
python realsense_server.py
```

## Data collection metadata preparation

1. Write down the task information in a json file inside `teleop/task_defs` directory
2. In the `teleop/` directory, run `python taskcreator.py` to generate the task metadata. Take a look at `teleop/task_defs/example.json` for an example task's json format.

## Teleoperating Data Collection Instruction

> Warning : All persons must maintain an adequate safety distance from the robot to avoid danger!

1. Connect your host computer to G1. Then, set up your local IP address to 192.168.123.123 with netmask 255.255.255.0 on G1's network interface.
2. Connect both your computer and AVP to the same local router wifi in which you set up your cert with.
3. Open robot and set to sports mode (using remote control by pressing L1 + A, then L1 + UP, and lastly R1 + X when the G1 is gently touching the ground).
4. On G1 PC: start the image server as above.
5. On host computer: run

```bash
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><NetworkInterfaceAddress>192.168.123.123</NetworkInterfaceAddress></General></Domain></CycloneDDS>"
```

6. On host computer: run `python main.py --robot g1` under the `teleop/` directory and wait until the robot is in ready state. The terminal should signal both "master" and "worker" processes are waiting for starting signal.
7. On AVP, connect to robot using https://<host_ip_address_on_your_local_router>:8012/?ws=wss://<host_ip_address_on_your_local_router>:8012. Then, press `Enter VR` and then `Allow` to enter the web interface for teleoperating the G1.
8. Back on host computer, enter `s` to start recording an episode.
9. Type `q` and enter if the episode is successful, otherwise `d` and enter to discard the last session.
10. Repeat by pressing `s` to start recording the next episode. Record 40 episodes for each task.
11. Type `exit` to shut down the program.

# Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/unitreerobotics/avp_teleoperate
2. https://github.com/OpenTeleVision/TeleVision
3. https://github.com/dexsuite/dex-retargeting
4. https://github.com/vuer-ai/vuer
5. https://github.com/stack-of-tasks/pinocchio
6. https://github.com/casadi/casadi
7. https://github.com/meshcat-dev/meshcat-python
8. https://github.com/zeromq/pyzmq
9. https://github.com/unitreerobotics/unitree_dds_wrapper
10. https://github.com/tonyzhaozh/act
11. https://github.com/facebookresearch/detr
