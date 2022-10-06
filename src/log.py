import logging
import logging.config

from os import path
log_file_path = path.join(path.dirname(path.abspath(__file__)), 'log.config')

logging.config.fileConfig(log_file_path, disable_existing_loggers=False)

app_logger = logging.getLogger(__name__)