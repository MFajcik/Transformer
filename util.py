import socket
import sys

__author__ = "Martin Fajčík"

import datetime
import logging
import logging.config
import os

import yaml


def setup_logging(
        module,
        default_level=logging.INFO,
        env_key='LOG_CFG',
        logpath=os.getcwd(),
        extra_name="",
        config_path=None
):
    """
        Setup logging configuration\n
        Logging configuration should be available in `YAML` file described by `env_key` environment variable

        :param module:     name of the module
        :param logpath:    path to logging folder [default: script's working directory]
        :param config_path: configuration file, has more priority than configuration file obtained via `env_key`
        :param env_key:    evironment variable containing path to configuration file
        :param default_level: default logging level, (in case of no local configuration is found)
    """

    if not os.path.exists(os.path.dirname(logpath)):
        os.makedirs(os.path.dirname(logpath))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    stamp = timestamp + "_" + socket.gethostname() + "_" + extra_name

    path = config_path if config_path is not None else os.getenv(env_key, None)
    if path is not None and os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            for h in config['handlers'].values():
                if h['class'] == 'logging.FileHandler':
                    h['filename'] = os.path.join(logpath, module, stamp, h['filename'])
                    touch(h['filename'])
            for f in config['filters'].values():
                if '()' in f:
                    f['()'] = globals()[f['()']]
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level, filename=os.path.join(logpath, stamp))


def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')


class LevelOnly(object):
    levels = {
        "CRITICAL": 50,
        "ERROR": 40,
        "WARNING": 30,
        "INFO": 20,
        "DEBUG": 10,
        "NOTSET": 0,
    }

    def __init__(self, level):
        self.__level = self.levels[level]

    def filter(self, logRecord):
        return logRecord.levelno <= self.__level


def touch(f):
    """
    Create empty file at given location f
    :param f: path to file
    """
    basedir = os.path.dirname(f)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    open(f, 'a').close()

import functools, traceback
def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = sys.exc_info()
            traceback.clear_frames(tb)
            raise type(val).with_traceback(tb) from None
    return wrapper
