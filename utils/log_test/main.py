import logging, time
from utils import get_logger
# import module_one as one
# import module_two as two

if __name__ == "__main__":
    # FORMAT = '%(asctime)-15s %(levelname)s %(message)s'
    # logging.basicConfig(format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S')
    # logging.basicConfig()
    # basicConfig shouldn't be called if multiple handlers are to be set manually

    # parsing log level from string parameter
    # each module should create their own "logger" to track message source

    wd = "."
    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = wd + "/" + timestr + ".log"

    logger = get_logger(__name__, filename)
    # print logger.getEffectiveLevel()

    for h in logger.handlers:
        print " ---- Found handler %s" % type(h)

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')

    # logging directly without the "logger" will replace module name with "root"
    # logging.info("test")

    # one.method("FOOO", 16)
    # two.method()
