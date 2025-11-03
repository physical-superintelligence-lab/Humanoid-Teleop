import numpy as np
import threading
import mujoco
import mujoco_viewer

from master_whole_body import RobotTaskmaster

def main():
    model = mujoco.MjModel.from_xml_path("../assets/g1/g1_body29_hand14.xml")
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer.cam.distance = 2.5

    for j in range(model.njnt):
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        print(f"{j:02d} | qpos adr={adr:<2} | name={jn}")
    
    for i in range(model.nu):
        j_id = model.actuator_trnid[i, 0]
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        print(f"ctrl[{i}] → {jn}")

    from multiprocessing import Event, Array
    shared_data = {
        "kill_event": Event(),
        "session_start_event": Event(),
        "failure_event": Event(),
        "end_event": Event(),
        "dirname": "./debug",
    }

    robot_shm_array = Array("d", 1024)
    teleop_shm_array = Array("d", 1024)

    taskmaster = RobotTaskmaster(
        "debug",
        shared_data,
        robot_shm_array,
        teleop_shm_array,
        robot="g1",
        sim_model=model,
        sim_data=data,
    )

    # 允许 standing 线程运行
    shared_data["kill_event"].set()

    # 开启 maintain_standing（它会从真机读状态 → MuJoCo控制）
    t = threading.Thread(target=taskmaster.maintain_standing, daemon=True)
    t.start()

    while viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()

    shared_data["end_event"].set()

if __name__ == "__main__":
    main()
