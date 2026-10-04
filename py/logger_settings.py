import logging
import sys

class DebugFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)

        if record.levelno == logging.DEBUG:
            return f"\033[95m{message}\033[0m"

        if record.levelno == logging.WARNING:
            return f"\033[93m{message}\033[0m"

        return message


def configure_logging(verbose: bool, debug: bool):

    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        DebugFormatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            "%H:%M:%S"
        )
    )

    logging.basicConfig(
        level=level,
        handlers=[
            handler
        ]
    )

#place duct tape on numba's mouth
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('numba.core.ssa').setLevel(logging.WARNING)
