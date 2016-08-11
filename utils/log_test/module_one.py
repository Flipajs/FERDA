import logging

def method(param1, param2):
    logger = logging.getLogger(__name__)
    # using parameters
    logger.info("Running method in module 1 with parameters %s and %d", param1, param2)
    logging.info("Where am I called?")
