# -*- coding: utf-8 -*-
""" A package of logging for information handling, which is compatible with DDP. 
    Learn from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/logger.py
"""

import os
import logging
from datetime import datetime
from collections import defaultdict
import torch
import torch.distributed as dist

# ================================================================
# API
# ================================================================


# value function
def logkv(key, val):
    """ Log a value of some diagnostic. 
        Call this once for each diagnostic quantity, each iteration. 
        If called many times, last value will be used.
    """
    get_current().logkv(key, val)


def logkv_mean(key, val):
    """ The same as logkv(), but if called many times, values averaged.
    """
    get_current().logkv_mean(key, val)


def logkv_max(key, val): 
    """ Log max value when called many times. 
        Value must be a positive number. 
    """
    get_current().logkv_max(key, val)


def logkvs(d):
    """ Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """ Write all of the diagnostics from the current iteration
    """
    return get_current().dumpkvs()


def getkvs():
    return get_current().name2val


# info function
def debug(*args, **kwargs): 
    get_current().log(logging.DEBUG, *args, **kwargs)


def info(*args, **kwargs): 
    get_current().log(logging.INFO, *args, **kwargs)


def warn(*args, **kwargs): 
    get_current().log(logging.WARNING, *args, **kwargs)


def error(*args, **kwargs): 
    get_current().log(logging.ERROR, *args, **kwargs)


""" There is no critical info. 
    Set console_level = logging.CRITICAL to disable terminal output. 
"""


def set_level(level):
    """ Set logging threshold on current logger. 
        Any info with lower level will be ignored. 
    """
    get_current().set_level(level)


# ================================================================
# Backend implementation
# ================================================================


# we use torchrun for running DDP model
def get_rank(): 
    if dist.is_initialized(): 
        return dist.get_rank()
    else: 
        return -1


def get_world_size(): 
    if dist.is_initialized(): 
        return dist.get_world_size()
    else: 
        return 1


def all_reduce_mean(val): 
    # get the average val from all GPUs
    assert Logger.DEVICE is not None, 'Device of GPU is not found. Please specify the device when setting up logger. '
    
    x_reduce = torch.tensor(val, device=Logger.DEVICE)
    dist.all_reduce(x_reduce)
    x_reduce /= get_world_size()
    return x_reduce.item()


def broadcast(val, rank): 
    # broadcast val from rank to all GPUs
    assert Logger.DEVICE is not None, 'Device of GPU is not found. Please specify the device when setting up logger. '

    if isinstance(val, torch.Tensor): 
        val_tensor = val.to(Logger.DEVICE)
        dist.broadcast(val_tensor, rank)
        val = val_tensor.cpu()
    elif isinstance(val, list): 
        val_tensor = torch.tensor(val, device=Logger.DEVICE)
        dist.broadcast(val_tensor, rank)
        val = val_tensor.cpu().tolist()
    elif isinstance(val, (int, float)): 
        val_tensor = torch.tensor(val, device=Logger.DEVICE)
        dist.broadcast(val_tensor, rank)
        val = val_tensor.item()
    else: 
        raise ValueError('Unsupport input type of val. ')
    
    return val


# ================================================================
# Module
# ================================================================


class Logger(object):
    """ A simple logger based on logging. 
    """
    CURRENT = None # as a global variable
    DEVICE = None # must specified first if using DDP

    def __init__(
        self, 
        log_file, 
        formatter, 
        level=logging.DEBUG, 
        console_level=logging.INFO, 
        disable_branch=True         # disable ouput if not on main process
    ): 
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.disable_branch = disable_branch

        # setup logging
        self.logger = logging.Logger('mylogger125')
        self.logger.setLevel(logging.DEBUG)
        logging.Formatter.converter = lambda *args: datetime.now().timetuple()

        if log_file is not None: 
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    # Logging API
    # ----------------------------------------
    def logkv(self, key, val): 
        val = val.item() if isinstance(val, torch.Tensor) else val
        self.name2val[key] = val

    def logkv_mean(self, key, val): 
        val = val.item() if isinstance(val, torch.Tensor) else val
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
        self.name2cnt[key] = cnt + 1
    
    def logkv_max(self, key, val): 
        # make sure value is always > 0
        val = val.item() if isinstance(val, torch.Tensor) else val
        oldval = self.name2val[key]
        if val > oldval: 
            self.name2val[key] = val
    
    def dumpkvs(self):
        if get_world_size() > 1: 
            for (name, val) in self.name2val.items(): 
                # Caution! Only float value would be calculated
                if isinstance(val, float): 
                    self.name2val[name] = all_reduce_mean(val)
        
        out = self.name2val.copy()

        # output as debug info
        # a very simple scheme, which will be improved further
        self.log(logging.DEBUG, dict(self.name2val), stacklevel=4)
        
        # clear
        self.name2val.clear()
        self.name2cnt.clear()

        return out

    def log(self, level, *args, **kwargs): 
        if get_rank() > 0 and level <= logging.INFO and self.disable_branch: 
            return
        
        if 'stacklevel' not in kwargs: 
            kwargs['stacklevel'] = 3 # output filename and line correctly
        
        if level == logging.DEBUG: 
            self.logger.debug(*args, **kwargs)
        elif level == logging.INFO: 
            self.logger.info(*args, **kwargs)
        elif level == logging.WARNING: 
            self.logger.warning(*args, **kwargs)
        elif level == logging.ERROR: 
            self.logger.error(*args, **kwargs)
        else: 
            raise ValueError(f"Unknown logger level: {level}")
    
    # hardly used
    def set_level(self, level):
        self.logger.setLevel(level)


def get_current(): 
    if Logger.CURRENT is None: # that means you only want to show message with level >= logging.INFO on terminal
        setup()

    return Logger.CURRENT


def setup(
    log_dir=None, 
    file_level=logging.DEBUG, 
    console_level=logging.INFO, 
    formatter=None, 
    disable_branch=True, 
    device=torch.device("cuda") if torch.cuda.is_available() and get_world_size() > 1 else None
): 
    """ Initialize logger configuration.

    Args:
        log_dir (str, optional): 
            Directory to save log files. If None, no log file will be saved. Defaults to None. 
        file_level (int, optional): 
            Logging level for file output. Defaults to DEBUG level. 
        console_level (int, optional): 
            Logging level for console output. Defaults to INFO level. 
        formatter (logging.Formatter, optional): 
            Custom log formatter if needed. Defaults to:
            '%Y-%m-%d %H:%M:%S filename[line:lineno] levelname message'. 
        disable_branch (bool, optional): 
            Whether to disable non-main process logging in multiprocessing. 
        device (torch.device, optional): 
            Device for logging in current process. Automatically set when starts with 'torchrun'. 
    """
    
    if formatter is None: 
        formatter = logging.Formatter(
            fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    Logger.DEVICE = device
    if log_dir is None: 
        log_file = None
    else: 
        log_dir = os.path.join(log_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # All processes use same filename. 
        # The file will be rewritten if using multi-host with shared path. 
        if get_world_size() > 1: 
            t = torch.tensor([int(n) for n in now.split('_')])
            t = broadcast(t, 0)
            now = '_'.join(str(int(n)).zfill(2) for n in t)
        
        log_file = os.path.join(log_dir, now + f'_log.txt')
    
    Logger.CURRENT = Logger(
        log_file=log_file, 
        formatter=formatter, 
        level=file_level, 
        console_level=console_level, 
        disable_branch=disable_branch, 
    )
    
    info('Logger setup done. ')

