# -*- coding: utf-8 -*-
""" Parser used by this program. 
"""

import argparse

class ModelParser():
    """ Setup base parser. 
    """

    def __init__(self): 
        self.parser = argparse.ArgumentParser()

        """ base options """
        self.parser.add_argument('--model', type=str, default='resnet50', help='Name of model. ')
        self.parser.add_argument('--output_dir', type=str, default='./output_dir', help='The directory for saving checkpoints and logs. ')
        
        """ dataloader options """
        self.parser.add_argument('--data_root', type=str, default='./data/GenImage2', help='Root of dataset. ')
        self.parser.add_argument('--num_workers', type=int, default=8,  help='Number of workers in dataloader. ')

        self.parser.add_argument('--seed', type=int, default=None, help='Random seed for the main processes. ')
    
    @property
    def args(self): 
        return self.parser.parse_args()
    
    @staticmethod
    def _str2bool(value): 
        if value.lower() in ('yes', 'true', 't', 'y', '1'): 
            return True
        elif value.lower() in ('no', 'false', 't', 'y', '0'): 
            return False
        else: 
            raise argparse.ArgumentTypeError('Unsupported value encountered. ')


class TrainParser(ModelParser): 
    def __init__(self): 
        super().__init__()
        """ optimizer options """
        self.parser.add_argument('--use_fp16', type=self._str2bool, default=True, help='Use fp16 when training. ')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Base learning rate for training. ')
        self.parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5, help='Gamma in StepLR. ')
        self.parser.add_argument('--lr_scheduler_step', type=int, default=80000, help='Scheduler step in StepLR. ')

        """ training options """
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size of tasks. ')
        self.parser.add_argument('--num_class_train', type=int, default=3, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_train', type=int, default=5, help='Number of samples in support set of each class during training. ')
        self.parser.add_argument('--num_query_train', type=int, default=5, help='Number of samples in query set of each class during training. ')

        self.parser.add_argument('--num_class_val', type=int, default=2, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_val', type=int, default=5, help='Number of samples in support set of each class during validation. ')
        self.parser.add_argument('--num_query_val', type=int, default=15, help='Number of samples in query set of each class during validation. ')

        self.parser.add_argument('--exclude_class', type=str, default="ADM", help='Image folder to be tested. ')
        
        self.parser.add_argument('--total_training_steps', type=int, default=200000, help='Total steps for training. ')
        self.parser.add_argument('--accumulation_steps', type=int, default=1, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints). ')
        self.parser.add_argument('--save_interval', type=int, default=10000, help='Interval between saving model weights. ')
        self.parser.add_argument('--log_interval', type=int, default=1000, help='Interval between logs. ')
        self.parser.add_argument('--eval_interval', type=int, default=10000, help='Interval between logs. ')

        # add other options if you want


class TestParser(ModelParser): 
    def __init__(self): 
        super().__init__()
        self.parser.add_argument('--use_fp16', type=self._str2bool, default=True, help='Use fp16 when training. ')
        self.parser.add_argument('--test_class', type=str, default="ADM", help='Image folder to be tested. ')
        self.parser.add_argument('--ckpt_path', type=str, default="/home/wushiyu/fewshotdetect/output_dir/ckpt/resnet50_100000.pth", help='Image folder to be tested. ')


        """ testing options """
        self.parser.add_argument('--num_class_test', type=int, default=2, help='Number of classes per iteration during training. ')
        self.parser.add_argument('--num_support_test', type=int, default=5, help='Number of samples in support set of each class during training. ')
        self.parser.add_argument('--num_query_test', type=int, default=15, help='Number of samples in query set of each class during training. ')
        # add other options if you want