import torch
import pytorch_lightning as pl
import numpy as np
import random
import json
import os
from datetime import datetime
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torch.utils.data import DataLoader

from data.dataset.kitti_dataset import KittiDataset
from model import DeeplabV3Resnet101


def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    pl.seed_everything(seed)  # PyTorch Lightning's seed function


def save_results(results, output_dir, filename=None):
    """
    Save test results to a JSON file.
    
    Args:
        results (dict): Test results to save
        output_dir (str): Directory to save results
        filename (str, optional): Custom filename for results. If None, generates a timestamp-based name
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{timestamp}.json"
    
    # Ensure filename has .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Full path to save file
    filepath = os.path.join(output_dir, filename)
    
    # Add timestamp to results
    results_with_meta = {
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Save results to file
    with open(filepath, 'w') as f:
        json.dump(results_with_meta, f, indent=4)
    
    print(f"Test results saved to: {filepath}")
    return filepath


def test_model_from_checkpoint(
    ckpt_path, test_dataloader=None, log_dir="lightning_logs", 
    results_dir="test_results", results_filename=None, seed=42
):
    """
    Load a PyTorch Lightning model from checkpoint and run the test phase with TensorBoard logging.

    Args:
        ckpt_path (str): Path to the checkpoint file
        test_dataloader (DataLoader, optional): Test data loader. If None, will use the one defined in the model.
        log_dir (str): Directory to save TensorBoard logs
        results_dir (str): Directory to save test results as JSON
        results_filename (str, optional): Custom filename for results. If None, generates a timestamp-based name
        seed (int): Random seed for reproducibility

    Returns:
        dict: Test results
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Load model from checkpoint
    model = DeeplabV3Resnet101.load_from_checkpoint(ckpt_path)

    # Put model in evaluation mode
    model.eval()

    # Set up TensorBoard logger
    logger = TensorBoardLogger(save_dir=log_dir, name="test_results")

    # Create trainer with logger
    trainer = Trainer(accelerator="auto", logger=logger)  # Uses GPU if available

    # Run test phase
    results = trainer.test(model, dataloaders=test_dataloader)

    # Save results to file
    save_results(results, results_dir, results_filename)
    
    print(f"TensorBoard logs saved to: {logger.log_dir}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test a PyTorch Lightning model from checkpoint"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="./data",
        help="Path to the test data image directory",
    )
    parser.add_argument(
        "--lbl_dir",
        type=str,
        default="./data",
        help="Path to the test data label directory",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="lightning_logs",
        help="Directory to save TensorBoard logs",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="test_results",
        help="Directory to save test results as JSON",
    )
    parser.add_argument(
        "--results_filename",
        type=str,
        default=None,
        help="Custom filename for results (default: timestamp-based)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = KittiDataset(
        args.img_dir,
        args.lbl_dir,
        image_transform=test_transform,
    )

    # Alternatively, if your model handles creating the dataloader:
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=3,
        persistent_workers=True,
    )

    # Run test with TensorBoard logging
    results = test_model_from_checkpoint(
        args.ckpt_path, test_dataloader, args.log_dir, 
        args.results_dir, args.results_filename, args.seed
    )
    print("Test results:", results)