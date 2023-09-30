import os
import logging


def get_logger(name, level):
  logger = logging.getLogger(name=name)
  logger.setLevel(level=level)

  ch = logging.StreamHandler()
  ch.setLevel(level=level)
  fh = logging.FileHandler(os.path.join(os.path.abspath(''), 'log'))
  fh.setLevel(level=level)

  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  fh.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)
  return logger
