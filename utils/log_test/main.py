import logging, time
import module_one as one
import module_two as two

if __name__ == "__main__":
    # FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
    # logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S')

    # parsing log level from string parameter
    loglevel = "debug"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    wd = "."
    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = wd + "/" + timestr + ".log"
    
    # configuring logging - this must be done before any log messages, otherwise it doesn't work
    
    logging.basicConfig(filename=filename, level=numeric_level)

    # each module should create their own "logger" to track message source
    logger = logging.getLogger(__name__)
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')

    # logging directly without the "logger" will replace module name with "root"
    logging.info("test")

    one.method("FOOO", 16)
    two.method()
