"""Logger setup for child processes."""

import logging
from logging import handlers
import sys
from typing import Callable, Optional
from torch import multiprocessing as mlp


def setup_child_logger(queue: mlp.Queue) -> None:
    """Sets up the logger for child processes."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Limpia handlers heredados (MUY importante)
    root.handlers.clear()

    queue_handler = handlers.QueueHandler(queue)
    root.addHandler(queue_handler)


def worker_wrapper(
    func: Callable[[mlp.Queue], list[str]], log_queue: mlp.Queue, flow_queue: mlp.Queue
) -> Optional[list[str]]:
    """Wrapper to setup child logger and execute the function."""
    try:
        setup_child_logger(log_queue)
        logger = logging.getLogger("STDOUT")
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.info("Child process logger initialized.")
        return func(flow_queue)
    except Exception as e:
        logging.exception("Error in worker wrapper")
        raise e


class StreamToLogger:
    """Redirects writes to a logger instance."""

    # Attributes that tqdm and other libraries check for terminal capabilities
    encoding = "utf-8"

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def isatty(self) -> bool:
        """Report as a TTY so tqdm uses Unicode characters."""
        return True

    def write(self, message) -> None:
        """Write message to logger, handling progress bars correctly."""
        if not message:
            return

        self._buffer += message

        # Process complete lines (ending with \n)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            # Handle \r: only keep the last segment (progress bar update)
            if "\r" in line:
                line = line.split("\r")[-1]
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

        # For progress bars: \r without \n means "update current line"
        if "\r" in self._buffer:
            # Get the last segment after \r
            progress_msg = self._buffer.split("\r")[-1].strip()
            if progress_msg:
                # Create a log record with a special attribute
                record = self.logger.makeRecord(
                    self.logger.name,
                    self.level,
                    "(progress)",
                    0,
                    progress_msg,
                    (),
                    None,
                )
                record.is_progress = True  # Mark as progress update
                self.logger.handle(record)
            self._buffer = ""  # Clear buffer after sending progress

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self._buffer.strip():
            self.logger.log(self.level, self._buffer.strip())
            self._buffer = ""
