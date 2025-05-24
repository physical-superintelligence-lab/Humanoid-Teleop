import time

from .logger import logger


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed*1000:.2f}ms")
        return result

    return wrapper


def run_with_retries(func, default_return=None, max_retries=3):
    """Execute a function with retry logic"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            logger.warning(
                f"Error in {func.__name__}: {e} (attempt {attempt+1}/{max_retries})"
            )
            time.sleep(0.1 * (2**attempt))  # Exponential backoff

    logger.error(f"Function {func.__name__} failed after {max_retries} attempts")
    return default_return


def monitor_resources():
    import psutil

    process = psutil.Process()
    memory = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    logger.debug(f"Memory: {memory.rss/1024/1024:.1f}MB, CPU: {cpu_percent}%")
