from logging import getLogger

logger = getLogger(__name__)


def method(param1, param2):
    # using parameters
    logger.info("Running method in module 1 with parameters %s and %d", param1, param2)
    logger.error("Something went wrong in method in module 1")
