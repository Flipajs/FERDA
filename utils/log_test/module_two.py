from __future__ import unicode_literals
import logging
from utils import get_logger

def method():
    logger = get_logger(__name__)
    logger.info("Running method in module 2")

    my_dict = {1: "one", 2: "two"}
    try:
        number_three = my_dict[3]
    except Exception as e:
        logger.exception(e)
