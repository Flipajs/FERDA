import logging.config
import module_one as one
import module_two as two
import yaml
import os


log_cfg = """
---
version: 1
disable_existing_loggers: False
formatters:
    simple:
        format: "%(asctime)s - %(name)11s - %(levelname)7s - %(message)s"

handlers:
    console:
        class: logging.StreamHandler
        level: INFO
        formatter: simple
        stream: ext://sys.stdout

    file_handler:
        class: logging.handlers.RotatingFileHandler
        level: DEBUG
        formatter: simple
        filename: [project_path]/crawler.log
        maxBytes: 10485760 # 10MB
        backupCount: 20
        encoding: utf8

root:
    level: DEBUG
    handlers: [console, file_handler]
...
"""


def init_logging(project_directory):
    """
    Initialise logging. This method must be called before retrieving any logger and
    before logging any message. If config file is not found, default config settings are used.
    :param project_directory path to the workspace (might not be created yet)"""

    cfg_file = project_directory + "/log_cfg.yaml"

    # use custom config for already existing workspaces
    if os.path.exists(cfg_file):
        with open(cfg_file, 'rt') as f:
            logging_config = yaml.safe_load(f.read().replace("[project_path]", project_directory))

    # load default config from templates, if workspaces wasn't yet generated
    else:
        logging_config = yaml.safe_load(log_cfg.replace("[project_path]", project_directory))

        if not os.path.exists(project_directory):
            os.makedirs(project_directory)

    # apply logging settings
    logging.config.dictConfig(logging_config)


if __name__ == "__main__":
    # print logger.getEffectiveLevel()
    init_logging(".")
    logger = logging.getLogger(__name__)

    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')
    logger.critical('This is CRITICAL!')

    logging.info("This is a info message from \"root\"")
    logging.critical("This is a critical message from \"root\"")

    one.method("FOOO", 16)
    two.method()

    logger.error('Another error!')
    logger.info('Finished.')

