import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from contextlib import contextmanager
import re

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)

def create_logger(cfg, cfg_path, phase='train', rank=-1, attack_type=None, epsilon=None, channel=None, step_decay=None, num_pixels=None, defense_type=None, experiment_number=0):
    # set up logger dir
    dataset = cfg.DATASET.DATASET
    dataset = dataset.replace(':', '_')
    model = cfg.MODEL.NAME
    cfg_path = os.path.basename(cfg_path).split('.')[0]
    attack_status = attack_type if attack_type else 'NoAttack'
    defense_status = defense_type if defense_type else 'NoDefense'
    
    if rank in [-1, 0]:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        run_dir = Path(cfg.LOG_DIR) / dataset / model / \
            (cfg_path + '_' + time_str + '_' + attack_status + '_' + defense_status)

        if not run_dir.exists():
            print(f'=> creating {run_dir}')
            run_dir.mkdir(parents=True, exist_ok=True)

        log_file = run_dir / f'{cfg_path}_{time_str}_{phase}_{attack_status}_{defense_status}.log'
        head = '%(asctime)-15s %(message)s'
        
        # Ensure to reset logging config to avoid conflicts
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(filename=str(log_file), format=head)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        console = logging.StreamHandler()
        logging.getLogger('').addHandler(console)

        return logger, str(run_dir), str(run_dir)
    else:
        return None, None, None

def create_experiment_logger(base_dir, experiment_number, attack_type, epsilon=None, channel=None, step_decay=None, num_pixels=None):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    exp_dir = Path(base_dir) / f'exp_{experiment_number}_epsilon_{epsilon}'
    
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    log_file = exp_dir / f'exp_{experiment_number}_{time_str}_{attack_type}.log'
    head = '%(asctime)-15s %(message)s'
    
    # Ensure to reset logging config to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger(f'exp_{experiment_number}')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(exp_dir)

def select_device(logger=None, device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            if logger:
                logger.info("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        if logger:
            logger.info(f'Using torch {torch.__version__} CPU')

    if logger:
        logger.info('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.TRAIN.LR0,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=cfg.TRAIN.LR0,
            betas=(cfg.TRAIN.MOMENTUM, 0.999)
        )

    return optimizer


def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()
