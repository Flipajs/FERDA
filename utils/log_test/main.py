from __future__ import absolute_import
import logging, time
from utils import get_logger, init_logging
from . import module_one as one
from . import module_two as two

if __name__ == "__main__":
    init_logging()
    logger = get_logger(__name__)
    # print logger.getEffectiveLevel()

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')
    logger.critical('This is CRITICAL!')

    # logging directly without the "logger" will use the default "root" logger
    # the root logger only logs in the file, not the console (no matter what level)
    logging.info("This is a info message from \"root\"")
    logging.critical("This is a critical message from \"root\"")

    one.method("FOOO", 16)
    two.method()

    logger.error('Another error!')
    logger.info('Finished.')
