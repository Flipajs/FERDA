from logging import getLogger

logger = getLogger(__name__)


def method():
    logger.info("Running method in module 2")

    my_dict = {1: "one", 2: "two"}
    try:
        my_dict[3]
    except Exception as e:
        logger.exception(e)
