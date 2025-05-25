""" Functions for devices preparing. 
"""

import os
import random
import numpy as np
import torch
import torch.distributed as dist
import dill


# initialize ddp environment
def setup_dist(args): 
    """ Prepare device for training or testing using DDP, while each process only runs on one GPU. 
        
        GPU ids should be continuous, like 0, 1, 2, ...
        Use os.environ['CUDA_VISIBLE_DEVICES'] to make it specific. 
    """

    # get args from environment
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.master_addr = os.environ['MASTER_ADDR'] # could set by rzdv_endpoint
    args.master_port = os.environ['MASTER_PORT']

    # init device
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device('cuda', args.local_rank)

    if args.local_rank == 0: 
        print('Initial process group with tcp://{}:{}. '.format(args.master_addr, args.master_port))
    
    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://{}:{}'.format(args.master_addr, args.master_port), 
        world_size=args.world_size, 
        rank=args.rank
    )
    # torch.distributed.barrier()

    # set seed
    if args.seed is not None:
        seed = args.seed + args.rank
        set_seed(seed)


def set_seed(seed): 
    """ Fix the seed for reproducibility. 
    """
    
    # python & numpy
    random.seed(seed)
    np.random.seed(seed)

    # pytorch
    """ Use torch.manual_seed() to seed the RNG for all device (both CPU and CUDA). 
    """
    torch.manual_seed(seed)

    # for convluational layers
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# model type, add more if needed
TRANSFORM_TYPE = (
    torch.nn.Module, 
    torch.optim.Optimizer, 
    torch.cuda.amp.GradScaler, 
    torch.optim.lr_scheduler.LRScheduler,
    torch.optim.lr_scheduler._LRScheduler,
) # transform types that need to be changed with 'state_dict' before saving


def save_model(output_dir, model_name, **kwargs): # not beautiful, update further
    """ Save model and other utils, which should be executed on master process. 
        Implement 'step' in kwargs. 
    """

    os.makedirs(output_dir, exist_ok=True)

    epoch = kwargs.pop("epoch", None)
    step =  kwargs.pop("step", None)

    if epoch is not None:
        count = epoch
        key_word = "epoch"
    elif step is not None: 
        count = step
        key_word = "step"
    else: 
        count = len(os.listdir(output_dir))
        key_word = "no"
    
    
    # filename and path
    save_filename = '%s_%s[%s].pth' % (model_name, key_word, count)
    save_path = os.path.join(output_dir, save_filename)
    
    # save data
    data_to_save = {
        key_word: count
    }
    
    for k, v in kwargs.items(): 
        if isinstance(v, TRANSFORM_TYPE): 
            data_to_save[k] = v.state_dict()
        else: 
            data_to_save[k] = v
    
    torch.save(data_to_save, save_path, pickle_module=dill) # to pickle scheduler with lambda function


def load_model(filename, **kwargs): 
    """ Load the parameters of the model and optimizer. 
        To avoid mistake, always loading model without DDP on cpu. 
        Immutable objects will be assigned with saved values. 
    """
    
    checkpoint = torch.load(filename, map_location='cpu') #always cpu is well

    for k, v in kwargs.items(): 
        assert k in checkpoint, 'Key "%s" has not been found in checkpoint %s' % (k, filename)
        if isinstance(v, TRANSFORM_TYPE): 
            v.load_state_dict(checkpoint[k])
        else: 
            kwargs[k] = checkpoint[k]
    
    return kwargs

