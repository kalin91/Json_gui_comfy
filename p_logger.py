"""Logger setup for parent process in JSON GUI module."""

import logging
import re
import sys
from multiprocessing.context import SpawnContext
from queue import Empty
import app.logger as applog
from comfy.cli_args import args
from torch import multiprocessing as mlp

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

PATTERN = r'(^\s*File\s+)"(.*)", line (\d+), in (.*)$'


class PrettyTracebackFormatter(logging.Formatter):
    """Custom formatter to prettify exception tracebacks."""

    def formatException(self, ei) -> str:
        """Formats exception tracebacks for better readability."""
        tb = super().formatException(ei)
        lines = tb.splitlines()
        out = []

        for line in lines:
            match = re.match(PATTERN, line)
            if match:
                pre_text = match.group(1)
                filename = match.group(2)
                lineno = match.group(3)
                funcname = match.group(4)
                out.append(f'{pre_text}"{filename}:{lineno}" in {funcname}')
            else:
                out.append(line)

        return "\n".join(out)


class ErrorFilter(logging.Filter):
    """Allows only ERROR and CRITICAL levels."""

    def filter(self, record) -> bool:
        if "Traceback (most recent call last):" in record.getMessage():

            def split_traceback_lines(msg: str) -> str:
                """Splits traceback lines to improve readability."""
                msg_lines = msg.splitlines()
                message_lines = []
                for line in msg_lines:
                    match = re.match(PATTERN, line, re.MULTILINE)
                    if match:
                        pre_text = match.group(1)
                        filename = match.group(2)
                        lineno = match.group(3)
                        funcname = match.group(4)
                        message_lines.append(f'{pre_text}"{filename}:{lineno}" in {funcname}')
                    else:
                        message_lines.append(line)
                return "\n".join(message_lines)

            try:
                record.msg = split_traceback_lines(record.getMessage())
                if hasattr(record, "message"):
                    record.message = split_traceback_lines(record.message)
            except Exception:
                pass
        return record.levelno >= logging.ERROR


class DefaultFilter(logging.Filter):
    """Allows only DEBUG, INFO, WARNING levels."""

    def filter(self, record) -> bool:
        return record.levelno < logging.ERROR


def setup_logger(
    level: int = logging.INFO,
    capacity: int = 300,
    use_stdout: bool = False,
) -> None:
    """
    Sets up the global logger for the parent process.

    Args:
        level (int, optional): Logging level. Defaults to logging.INFO.
        capacity (int, optional): Maximum number of log records to keep. Defaults to 300.
        use_stdout (bool, optional): Whether to use stdout for non-error logs. Defaults to False.
    """
    if applog.get_logs():
        return

    applog.logs = applog.deque(maxlen=capacity)

    # Intercept stdout / stderr
    applog.stdout_interceptor = sys.stdout = applog.LogInterceptor(sys.stdout)
    applog.stderr_interceptor = sys.stderr = applog.LogInterceptor(sys.stderr)

    logger.setLevel(level)

    error_format = "%(asctime)25s  [%(levelname)-8s] %(filename)25s:%(funcName)-30s:%(lineno)-5d  %(message)s"

    default_format = "%(asctime)25s  [%(levelname)-8s] %(filename)25s:%(funcName)-30s:%(lineno)-5d  %(message)s"

    formatter = logging.Formatter(default_format)
    error_formatter = PrettyTracebackFormatter(error_format)

    # Always log errors
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(error_formatter)
    error_handler.addFilter(ErrorFilter())
    logger.addHandler(error_handler)

    if use_stdout:
        # Non-errors go to stdout
        default_handler = logging.StreamHandler(sys.stdout)
        default_handler.setLevel(logging.DEBUG)
        default_handler.setFormatter(formatter)
        default_handler.addFilter(DefaultFilter())
        logger.addHandler(default_handler)
    else:
        # Everything goes to stderr
        fallback_handler = logging.StreamHandler(sys.stderr)
        fallback_handler.setLevel(logging.DEBUG)
        fallback_handler.setFormatter(formatter)
        logger.addHandler(fallback_handler)


setup_logger(level=args.verbose, use_stdout=args.log_stdout)
logging.info("Logger initialized for parent process.")
# Use spawn context to avoid CUDA fork issues
MP_CONTEXT: SpawnContext = mlp.get_context("spawn")

# Global queue for child process logging (must use spawn context)
LOG_QUEUE: mlp.Queue = MP_CONTEXT.Queue()


def poll_log_queue() -> int:
    """Poll the log queue and process any pending records.

    Returns:
        Number of records processed.
    """
    count = 0
    while True:
        try:
            record = LOG_QUEUE.get_nowait()
            if record is None:
                break
            logger.handle(record)
            count += 1
        except Empty:
            break
    return count


def get_log_queue() -> mlp.Queue:
    """Get the global log queue for child processes."""
    return LOG_QUEUE


def get_mp_context() -> SpawnContext:
    """Get the spawn multiprocessing context."""
    return MP_CONTEXT
