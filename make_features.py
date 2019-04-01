import os
from logging import getLogger
import time

from santander.features.features import Main, DAE
from santander.utils import (
    send_line_notification, send_error_to_line,
    setup_logger, calc_time
)

DATA_DIR = './input/'
logger = getLogger(__name__)
setup_logger(logger, './log/make_features.log')


def make_features():
    start_time = time.time()

    Main(data_dir=DATA_DIR, out=logger.info).run()
    DAE(data_dir=DATA_DIR, device='cuda:0', out=logger.info, prefix='dae').run()

    elapsed_time = calc_time(start_time)
    message = f'make_features done in {elapsed_time}.'
    send_line_notification(message)


if __name__ == '__main__':
    with send_error_to_line('function make_features failed.'):
        make_features()
