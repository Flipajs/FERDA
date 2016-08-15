import logging
from utils import get_logger

def method():
    logger = get_logger(__name__)
    logger.info("Running method in module 2")
