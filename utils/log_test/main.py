import logging
import module_one as one
import module_two as two

if __name__ == "__main__":
    # FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
    # logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S')

    loglevel = "debug"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
    logger = logging.getLogger(__name__)
    logger.debug('This message should appear on the console')
    logger.info('So should this')
    logger.warning('And this, too')

    one.method("a", "b")
    two.method("c", "d")
