启动机器人
等待30s 手指重置
等待30s 机器人变软
遥控器先一起按l1+a; 然后 l2+r2
connect to wifi: tp-link-e5d4-5g
open 1st terminal: connect to robot
ssh unitree@192.168.123.164  password:123
如果机器人端卡死或者visionpro里面没画面： ctrl c; killall python; python realsense_server.py
cd ~/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop
conda activate robopolicy

open 2nd terminal:
cd ~/hongyi/Unitree_Robotics/Humanoid-Teleop/teleop
conda activate robopolicy
机器人脚刚好接触地面
python main.py --robot g1

vision pro 进一个网页：
https://192.168.0.110:8012?ws=wss://192.168.0.110:8012

稍微长按visionpro右上角按钮 让页面居中
enter vr

websocket is connected. id:9548aacf-2249-43e1-b26e-5fca8d4a3da6
default socket worker is up, adding clientEvents 
Uplink task running. id:9548aacf-2249-43e1-b26e-5fca8d4a3da6

s: 开始teleop录制
q: 结束并保存当前episode，记得要等一下等到终端说waiting for start再开始因为文件要merge写入(几秒就可以)
d: 结束并丢弃当前episode，适用于失败数据
exit：直接退出



