import datetime
import logging
import time

from .dist_util import get_dist_info, master_only

initialized_logger = {}

class AvgTimer():
    def __init__(self, window=200):
        self.window = window
        self.current_time = 0
        self.total_time = 0
        self.count = 0
        self.avg_time = 0
        self.start()

    def start(self):
        self.start_time = self.tic = time.time()

    def record(self):
        self.count += 1
        self.toc = time.time()
        self.current_time = self.toc - self.tic
        self.total_time += self.current_time
        self.avg_time = self.total_time / self.count
        if self.count > self.window:
            self.count = 0
            self.total_time = 0
        self.tic = time.time()

    def get_current_time(self):
        return self.current_time

    def get_avg_time(self):
        return self.avg_time

class MessageLogger():
    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt.get('name', 'inference')
        self.logger = get_root_logger()
        self.start_time = time.time()

    def reset_start_time(self):
        self.start_time = time.time()

    @master_only
    def __call__(self, log_vars):
        # 推理阶段通常不调用此处的 __call__，但为了兼容性保留
        message = f'[{self.exp_name}] Processing...'
        for k, v in log_vars.items():
            if isinstance(v, float):
                message += f' {k}: {v:.4f}'
        self.logger.info(message)

def init_tb_logger(log_dir):
    return None

def init_wandb_logger(opt):
    return None

def get_root_logger(logger_name='team01_HAT', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    
    # 兼容单卡推理逻辑
    rank, _ = get_dist_info()
    if rank != 0 and rank is not None:
        logger.setLevel('ERROR')
    else:
        logger.setLevel(log_level)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, 'w')
            file_handler.setFormatter(logging.Formatter(format_str))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            
    initialized_logger[logger_name] = True
    return logger

def get_env_info():
    import torch
    import torchvision
    msg = "\nVersion Information: "
    msg += f'\n\tPyTorch: {torch.__version__}'
    msg += f'\n\tTorchVision: {torchvision.__version__}'
    return msg