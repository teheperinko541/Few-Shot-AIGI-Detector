from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.distributed as dist

def setup_infinity_train_dataloader(
    folder_path, 
    batch_size=20, 
    num_workers=16, 
    pin_memory=True, 
    drop_last=True
):
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.RandomCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(folder_path, transform=transform) # As GenImage dataset is very large, so it's ok for different size of each class
    sampler = DistributedSampler(dataset) if dist.is_initialized() else None

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last
    )
    
    # adding epoch control
    epoch = 0

    while True: 
        if sampler is not None: 
            sampler.set_epoch(epoch)
        epoch += 1
        yield from loader


def setup_val_dataloader(
    folder_path, 
    batch_size=20, 
    num_workers=16, 
    pin_memory=True, 
    drop_last=True
): 
    transform = transforms.Compose([ 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
    ])

    return DataLoader(
        ImageFolder(folder_path, transform=transform), 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=drop_last, 
    )

