import sys
import os


def _is_local_main_process():
    from toolkit.accelerator import get_accelerator
    return get_accelerator().is_local_main_process


def print_acc(*args, **kwargs):
    if _is_local_main_process():
        print(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def setup_log_to_file(filename, is_local_main_process=None):
    if is_local_main_process is None:
        is_local_main_process = _is_local_main_process()
    if is_local_main_process:
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
    # Capture the real streams before replacing them — wrapping the
    # already-replaced sys.stdout as the stderr Logger's "terminal" would
    # double-write every stderr message to the file. Both wrappers share a
    # single file handle.
    log_file = open(filename, 'a')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)
