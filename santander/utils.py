from contextlib import contextmanager
from logging import StreamHandler, Formatter, FileHandler
from pathlib import Path
import requests
import time


class Timer:
    def __init__(self, out=print, init=True):
        self.out = out
        self.start_time = time.time()
        self.time = None
        self.msg = None

        if init:
            self.out('Start.\n')

    @staticmethod
    def _calc_time(start_time):
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            return f'{elapsed_time:.1f} sec'
        elif elapsed_time < 3600:
            elapsed_time = round(elapsed_time)
            return f'{elapsed_time // 60} min {elapsed_time % 60} sec'
        else:
            elapsed_time = round(elapsed_time // 60)  # min
            return f'{elapsed_time // 60} hour {elapsed_time % 60} min'

    def _step_out(self):
        if self.msg:
            self.out(f'[{self.msg}] done in {self._calc_time(self.time)}.\n')

    def step(self, msg):
        self._step_out()
        
        self.time = time.time()
        self.msg = msg
        self.out(f'[{self.msg}] start.')

    def step_fin(self):
        assert self.msg

        self._step_out()
        self.msg = None
        self.time = None

    def finish(self):
        self._step_out()
        elapsed_time = self._calc_time(self.start_time)
        self.out(f'All processes done in {elapsed_time}.')
        return elapsed_time


@contextmanager
def step_timer(msg, out=print, init=True):

    def _calc_time(start_time):
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            return f'{elapsed_time:.1f} sec'
        elif elapsed_time < 3600:
            elapsed_time = round(elapsed_time)
            return f'{elapsed_time // 60} min {elapsed_time % 60} sec'
        else:
            elapsed_time = round(elapsed_time // 60)  # min
            return f'{elapsed_time // 60} hour {elapsed_time % 60} min'

    t0 = time.time()
    if init:
        out(f'[{msg}] start.')
    yield
    out(f'[{msg}] done in {_calc_time(t0)}.')


def setup_logger(logger, log_file_path, clear_log_file=True):
    Path(log_file_path).parent.mkdir(exist_ok=True)

    if clear_log_file:
        with open(log_file_path, 'w'):
            pass

    fmt = '%(asctime)s %(name)s%(lineno)d [%(levelname)s][%(funcName)s] %(message)s'
    log_fmt = Formatter(fmt=fmt)
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler(log_file_path, 'a')
    handler.setLevel('DEBUG')
    handler.setFormatter(log_fmt)
    logger.setLevel('DEBUG')
    logger.addHandler(handler)


def send_line_notification(message):
    line_token = 'aW0gilFfKYQ1ivcRFBcWnBsJ8Wn4jxoI4vq1Q4EGmme'
    endpoint = 'https://notify-api.line.me/api/notify'
    message = f'\n{message}'
    payload = {'message': message}
    headers = {'Authorization': f'Bearer {line_token}'}
    requests.post(endpoint, data=payload, headers=headers)


@contextmanager
def send_error_to_line(message):
    try:
        yield
    except Exception as e:
        error_message = f'''{message}
            error: {e}'''
        send_line_notification(error_message)
        raise Exception(e)
