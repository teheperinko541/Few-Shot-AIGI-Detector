# -*- coding: utf-8 -*-
""" Train ResNet50 for classification. 
"""

import os
import random
import torch
from tqdm import tqdm

from torch.amp import autocast, GradScaler
import timm
from einops import rearrange
from torchmetrics.classification import Accuracy, AveragePrecision

from model.prototypical_utils import compute_prototypical_loss
from datasets import setup_infinity_train_dataloader, setup_val_dataloader
from util.parser import TrainParser
from util.utils import save_model, setup_dist
import util.logger as logger


def main(): 
    #################### prepare ####################
    args = TrainParser().args
    setup_dist(args) # ddp setup

    # terminal writer and file writer
    logger.setup(log_dir=args.output_dir, device=args.device)
    #################################################

    
    ########## setup dataset and dataloader #########
    logger.info("Creating training data loader...")

    # data we use in GenImage, real is nature from SDv14 & SDv15
    IMAGE_FOLDERS = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
    IMAGE_FOLDERS.remove(args.exclude_class)
    logger.info(f"Exclude class: {args.exclude_class}")

    train_iters = {
        folder: setup_infinity_train_dataloader(
            folder_path=os.path.join(args.data_root, folder, "train"), 
            batch_size=(args.num_support_train + args.num_query_train) * args.batch_size, # batch_size * task_size
            num_workers=args.num_workers
        ) for folder in IMAGE_FOLDERS
    }

    VAL_FOLDERS = IMAGE_FOLDERS + [args.exclude_class] # put at last
    val_dataloaders = {
        folder: setup_val_dataloader(
            folder_path=os.path.join(args.data_root, folder, "val"), 
            batch_size=args.num_support_val + args.num_query_val, 
            num_workers=args.num_workers, 
        ) for folder in VAL_FOLDERS
    }
    #################################################

    
    ################## create model #################
    logger.info("Creating model 'resnet50'... ")
    model = timm.create_model("resnet50", pretrained=True, num_classes=1024)
    print(model)

    model = model.to(args.device)
    #################################################
    

    ######### create optimizer and criterion ########
    logger.info("Creating optimizer and scheduler... ")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(optimizer)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        gamma=args.lr_scheduler_gamma,
        step_size=args.lr_scheduler_step
    )
    print(scheduler)
    #################################################


    #################### training ###################
    logger.info("Start training for %d steps. " % args.total_training_steps)

    scaler = GradScaler(enabled=args.use_fp16)

    effective_step = 0
    # starts looping
    for step in range(1, args.total_training_steps + 1): 
        model.train()

        optimizer.zero_grad()

        # select classes for single prototypical task
        selected_classes = random.sample(IMAGE_FOLDERS, args.num_class_train)

        # get data
        batch_data = torch.stack([next(train_iters[c])[0] for c in selected_classes], dim=0) # (num_class, batch * task_size, c, h, w)
        batch_data = batch_data.to(args.device)

        # make labels
        labels = torch.arange(0, args.num_class_train, device=args.device).repeat(args.batch_size * args.num_query_train)

        batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
        with autocast(enabled=args.use_fp16, device_type="cuda"):
            outputs = model(batch_data)
        outputs = rearrange(outputs, '(n b t) l -> b t n l', n=args.num_class_train, b=args.batch_size) # we change the subscript sequence

        loss, _ = compute_prototypical_loss(outputs, labels, args.num_support_train)

        logger.logkv_mean("loss", loss.item())
        scaler.scale(loss / args.accumulation_steps).backward()
        
        # accumulate
        if step % args.accumulation_steps == 0:
            effective_step += 1

            scaler.unscale_(optimizer)
            # other options if you want

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None: 
                scheduler.step() # per effective iter
        
        # logger info
        if step % args.log_interval == 0:
            logger.logkv("step", step)
            logger.logkv("effective_step", effective_step)
            logger.logkv("lr", scheduler.get_last_lr()[0] if scheduler is not None else args.lr)
            logger.dumpkvs()
        
        # save checkpoint
        if step % args.save_interval == 0: 
            logger.info('Save checkpoint at step: %d', step)
            
            kwargs = {
                'step': step, 
                'effective_step': effective_step, 
                'model': model, 
                'optimizer': optimizer, 
                'scheduler': scheduler, 
                'scaler': scaler, 
                'args': args
            }

            save_model(os.path.join(args.output_dir, "ckpt"), args.model, **kwargs)
            torch.cuda.empty_cache()
        
        
        ##### evaluation #####
        if step % args.eval_interval == 0: 
            logger.info('Evaluating at step: %d', step)
            model.eval()

            acc_calculator = Accuracy(task="multiclass", num_classes=2)
            ap_calculator = AveragePrecision(task="multiclass", num_classes=2, thresholds=10)

            # build dataset
            with torch.no_grad(): 
                for i in range(1, len(VAL_FOLDERS)): 
                    prob_list = []
                    label_list = []
                    for (real_batch, _), (fake_batch, _) in tqdm(zip(val_dataloaders["real"], val_dataloaders[VAL_FOLDERS[i]])): 
                        batch_data = torch.stack([real_batch, fake_batch], dim=0) # (2, task_size, c, h, w)
                        batch_data = batch_data.to(args.device)

                        batch_data = rearrange(batch_data, 'n b c h w -> (n b) c h w')
                        labels = torch.arange(0, 2, device=args.device).repeat(args.num_query_val)

                        with autocast(enabled=args.use_fp16, device_type="cuda"):
                            outputs = model(batch_data)
                        outputs = rearrange(outputs, '(n b) l -> 1 b n l', n=2) # we change the subscript sequence

                        _, scores = compute_prototypical_loss(outputs, labels, args.num_support_val)
                        
                        prob = scores.softmax(dim=-1).cpu()
                        labels = labels.cpu()

                        prob_list.append(prob)
                        label_list.append(labels)
                    
                    total_prob = torch.cat(prob_list, dim=0)
                    total_label = torch.cat(label_list, dim=0)
                    acc = acc_calculator(total_prob, total_label)
                    ap = ap_calculator(total_prob, total_label)

                    logger.info(f'Evaluation on {VAL_FOLDERS[i]} done. evaluating num: {len(total_prob)}, accuracy: {acc}, average precision: {ap}. ')
        ##### evaluation done #####
    
    #################################################


if __name__ == '__main__': 
    main()

