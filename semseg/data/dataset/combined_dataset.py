from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    """
    Dataset that combines real and synthetic datasets into one.
    """

    def __init__(self, real_dataset, synthetic_dataset):
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.real_size = len(real_dataset)
        self.synthetic_size = len(synthetic_dataset)

    def __len__(self):
        return self.real_size + self.synthetic_size

    def __getitem__(self, idx):
        if idx < self.real_size:
            # Get item from real dataset
            return self.real_dataset[idx]
        else:
            # Get item from synthetic dataset
            synth_idx = idx - self.real_size
            return self.synthetic_dataset[synth_idx]
