import logging
from utils import get_logger
# import module_two as two

def method(param1, param2):
    logger = get_logger(__name__)
    # using parameters
    logger.info("Running method in module 1 with parameters %s and %d", param1, param2)
    logger.error("Something went wrong in method in module 1")
