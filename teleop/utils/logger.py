import logging


# --------------------- Debug Logger Setup ---------------------
class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[37m",  # white
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname
        if original_levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[original_levelname]}{original_levelname}{self.RESET}"
            )
        formatted = super().format(record)
        record.levelname = original_levelname
        return formatted


logger = logging.getLogger("robot_teleop")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = ColorFormatter("[%(asctime)s] [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# os.makedirs('logs', exist_ok=True)
# fh = logging.FileHandler(f"logs/robot_teleop_{time.strftime('%Y%m%d_%H%M%S')}.log")
# fh.setFormatter(formatter)
# logger.addHandler(fh)
# --------------------------------------------------------------
