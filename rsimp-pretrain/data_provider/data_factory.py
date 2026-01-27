import torch
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_EuroSAT_MS, Dataset_EuroSAT_RGB, Dataset_EuroSAT_NIR, Dataset_EuroSAT_RGBN

# Mapping from string to Dataset class
dataset_dict = {
    'eurosat_ms': Dataset_EuroSAT_MS,
    'eurosat_rgb': Dataset_EuroSAT_RGB,
    'eurosat_nir': Dataset_EuroSAT_NIR,
    'eurosat_rgbn': Dataset_EuroSAT_RGBN
}

def data_provider(args, flag='train'):
    """
    Factory function to return Dataset and DataLoader based on args.data
    flag: 'train', 'val', or 'test'
    """
    assert args.data in dataset_dict, f"Unknown dataset {args.data}"

    Data = dataset_dict[args.data]

    # Determine shuffling and batch size based on train/test
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size  # evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    # Instantiate Dataset
    dataset = Data(
        configs=args,
        image_size=(args.image_size, args.image_size),
        artificially_missing_rate=args.mask_rate,
        flag=flag
        )

    print(f"[INFO] {flag.upper()} set: {len(dataset)} samples, batch_size={batch_size}")

    # Build DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    return dataset, data_loader
