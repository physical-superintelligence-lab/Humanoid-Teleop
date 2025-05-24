import argparse

from manager import TeleopManager

# TODO: make it client server
# create a tv.step() thread and request image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Teleoperation Data Collector")
    parser.add_argument(
        "--task_name", type=str, default="default_task", help="Name of the task"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--robot", default="g1", help="Use h1/g1 controllers")
    args = parser.parse_args()

    manager = TeleopManager(
        task_name=args.task_name, robot=args.robot, debug=args.debug
    )
    manager.start_processes()
    # TODO: run in two separate terminals for debuggnig
    manager.run_command_loop()
