import os
import random
import glob
import yaml
from tqdm import tqdm
import wandb
import torch
import importlib
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.data.utils import decollate_batch
from monai.data import DataLoader, CacheDataset
from monai.transforms import Compose, AsDiscrete
import logging
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt

from monai.metrics import (
    DiceMetric, 
    MeanIoU, 
    HausdorffDistanceMetric, 
    ConfusionMatrixMetric
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Dynamically load the model based on config prefix
def load_model_from_config(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_class_path = list(config['net'].keys())[0]
    model_kwargs = config['net'][model_class_path].get('kwargs', {})

    if model_class_path.startswith("monai."):
        model_module_path, model_class_name = model_class_path.rsplit(".", 1)
    elif model_class_path.startswith("CustomModels."):
        model_module_path, model_class_name = model_class_path.replace("CustomModels.", "").rsplit(".", 1)
    else:
        model_module_path, model_class_name = model_class_path.rsplit(".", 1)

    # Import the module and class dynamically
    model_module = importlib.import_module(model_module_path)
    model_class = getattr(model_module, model_class_name)
    return model_class(**model_kwargs).to(device)

 # Utility function to dynamically load a transformation class
def load_transform(class_path, args, kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    transform_class = getattr(module, class_name)
    return transform_class(*args, **kwargs)

def main():
    # Load configuration
    with open("C:/Users/gabridal/Documents/CT_kidney_segmentation/configs/config_template.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Ensure W&B API key is set and login if needed
    wandb.login()

    # Generate timestamp for unique experiment naming
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{config['wandb']['experiment_name']}_{timestamp}"

    # Initialize W&B run with settings from config
    run = wandb.init(
        project=config['wandb']['project_name'],
        entity=config['wandb']['entity_name'],
        name=experiment_name
    )

    # Define custom metrics to track
    run.define_metric("train/loss", summary="min")
    run.define_metric("val/dice_metric", summary="max")
    run.define_metric("val/sensitivity", summary="max")

    # Log the configuration dictionary to W&B for experiment tracking
    wandb.config.update(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically load the model based on config prefix
    model = load_model_from_config(config)
    model.to(device)

    # Load loss function from config
    loss_class_path, loss_params = next(iter(config['loss'].items()))
    loss_args = loss_params.get("args", [])
    loss_kwargs = loss_params.get("kwargs", {})

    # Convert weight to a torch.Tensor if it exists
    if 'weight' in loss_kwargs and isinstance(loss_kwargs['weight'], list):
        loss_kwargs['weight'] = torch.tensor(loss_kwargs['weight'], dtype=torch.float32).to(device)
    
    # Dynamically import and initialize the loss class
    module_path, class_name = loss_class_path.rsplit(".", 1)
    loss_module = importlib.import_module(module_path)
    loss_class = getattr(loss_module, class_name)
    loss_function = loss_class(*loss_args, **loss_kwargs)

    # Prepare optimizer
    learning_rate = config['train']['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config['train']['max_epochs'], eta_min=1e-9)

    # Initialize metrics and post-processing
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")
    confusion_matrix_metric = ConfusionMatrixMetric(
        include_background=False, 
        metric_name=["accuracy", "precision", "sensitivity", "specificity", "f1_score"],
        reduction="mean"
    )
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    # post_label = AsDiscrete(to_onehot=2)

    # Prepare data from configuration
    # Create the data list by pairing image and label files
    image_files = sorted(glob.glob(os.path.join(config['data']['image'], "*.nii")))  # Adjust extension if needed
    label_files = sorted(glob.glob(os.path.join(config['data']['label'], "*.nii")))

    assert len(image_files) == len(label_files), "Mismatch between number of images and labels."

    # Prepare the dataset as a list of dictionaries
    data = [{"image": img, "label": lbl} for img, lbl in zip(image_files, label_files)]
    assert len(data) > 0, "Data list is empty. Check image and label paths."

    from sklearn.model_selection import train_test_split

    # Split the data with an 80% train / 20% validation ratio
    train_files, val_files = train_test_split(data, test_size=0.2, random_state=42)

    # Print the number of samples
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")


    # Parse and apply transformations from config for training and validation
    train_transforms = Compose([
        load_transform(aug_class, aug_details.get('args', []), aug_details.get('kwargs', {}))
        for aug in config['train']['augmentation']
        for aug_class, aug_details in aug.items()
    ])

    val_transforms = Compose([
        load_transform(aug_class, aug_details.get('args', []), aug_details.get('kwargs', {}))
        for aug in config['valid']['augmentation']
        for aug_class, aug_details in aug.items()
    ])

    # Initialize CacheDataset with training and validation data lists
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=config['data']['cache_rate'], 
        num_workers=config['data']['num_workers']
    )
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=0.8, 
        num_workers=config['data']['num_workers']
    )

    # Create DataLoader for training and validation
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['train']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers']
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=config['train']['val_batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers']
    )

    # Training loop
    best_metric = -1
    
    for epoch in range(config['train']['max_epochs']):
        print(f"Epoch {epoch + 1}/{config['train']['max_epochs']}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            wandb.log({"train/loss_step": loss.item()})
        
        # Average epoch loss and log
        epoch_loss /= step
        wandb.log({"train/loss_epoch": epoch_loss})

        # Validation
        if (epoch + 1) % config['train']['val_interval'] == 0:
            model.eval()
            val_loss = 0
            val_step = 0
            with torch.no_grad():
                for val_data in tqdm(val_loader):
                    val_step += 1
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)

                    # # Sliding window inference for larger inputs
                    raw_outputs = sliding_window_inference(val_inputs, 
                                                           config['valid']['roi_size'], 
                                                           config['valid']['sw_batch_size'], 
                                                           model)
                    
                    print(f"Shape of val_outputs: {raw_outputs.shape}")
                    print(f"Shape of val_labels: {val_labels.shape}")

                    # Calculate loss
                    val_loss += loss_function(raw_outputs, val_labels).item()

                    # Post-process predictions and labels
                    val_outputs = [post_pred(i) for i in decollate_batch(raw_outputs)]
                    val_labels =  decollate_batch(val_labels)

                    # Convert val_outputs to class indices for Hausdorff Distance
                    val_outputs_hd = torch.argmax(raw_outputs, dim=1, keepdim=True)  # Shape: [B, 1, D, H, W]

                    # Compute metrics
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    iou_metric(y_pred=val_outputs, y=val_labels)
                    hd95_metric(y_pred=val_outputs_hd, y=val_labels)
                    confusion_matrix_metric(y_pred=val_outputs, y=val_labels)

                # Aggregate metrics
                val_loss /= val_step
                dice = dice_metric.aggregate().item()
                iou = iou_metric.aggregate().item()
                hd95 = hd95_metric.aggregate().item()
                cm = confusion_matrix_metric.aggregate()

                # Log metrics
                logging.info(
                    f"Validation Results - Epoch {epoch + 1}:\n"
                    f"\tLoss: {val_loss:.4f}\n"
                    f"\tDice: {dice:.4f}\n"
                    f"\tIoU: {iou:.4f}\n"
                    f"\t95% HD: {hd95:.4f}\n"
                    f"\tAccuracy: {(acc := cm[0].item()):.4f}\n"
                    f"\tPrecision: {(pre := cm[1].item()):.4f}\n"
                    f"\tSensitivity: {(sen := cm[2].item()):.4f}\n"
                    f"\tSpecificity: {(spe := cm[3].item()):.4f}\n"
                    f"\tF1 score: {(f1 := cm[4].item()):.4f}"
                )

                # Log metrics to Weights & Biases
                wandb.log({
                    "val/loss": val_loss,
                    "val/dice": dice,
                    "val/iou": iou,
                    "val/95hd": hd95,
                    "val/accuracy": acc,
                    "val/precision": pre,
                    "val/sensitivity": sen,
                    "val/specificity": spe,
                    "val/f1_score": f1,
                })
            
                # Reset metrics for the next epoch
                dice_metric.reset()
                iou_metric.reset()
                hd95_metric.reset()
                confusion_matrix_metric.reset()
            
            # Visualize valid samples
            random.seed(config["train"]["seed"])  # Ensure reproducibility
            sample_indices = random.sample(range(len(val_ds)), config["valid"]["num_sample_images"])

            
            # Load random samples
            inputs = torch.stack([val_ds[i]["image"] for i in sample_indices]).to(device)  # Shape: [B, C, D, H, W]
            labels = torch.stack([val_ds[i]["label"] for i in sample_indices]).to(device)  # Shape: [B, 1, D, H, W]

            # Perform inference
            outputs = sliding_window_inference(
                inputs,
                config["valid"]["roi_size"],
                config["valid"]["sw_batch_size"],
                model,
            )

            # Post-process predictions (reduce channel dimension with argmax)
            outputs = torch.argmax(outputs, dim=1, keepdim=True)  # Shape: [B, 1, D, H, W]
            print(f"Shape of outputs after argmax: {outputs.shape}")  # Should be [1, 1, 160, 160, 160]


            # Visualize results
            num_samples = config["valid"]["num_sample_images"]
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

            for i in range(num_samples):
                slice_idx = inputs.shape[2] // 2  # Middle slice of the 3D volume
                axes[i, 0].imshow(inputs[i, 0, slice_idx, :, :].cpu(), cmap="gray")  # Input
                axes[i, 0].set_title("Input")
                axes[i, 1].imshow(labels[i, 0, slice_idx, :, :].cpu(), cmap="gray")  # Ground Truth
                axes[i, 1].set_title("Ground Truth")
                axes[i, 2].imshow(outputs[i, 0, slice_idx, :, :].cpu(), cmap="gray")  # Prediction (foreground class)
                axes[i, 2].set_title("Prediction")

            # Remove axes for a cleaner look
            for ax in axes.flatten():
                ax.axis("off")

            # Log the figure to W&B
            try:
                wandb.log({"validation_examples": wandb.Image(fig)})
            except Exception as e:
                logging.warning(f"Failed to log validation examples to W&B: {e}")

            # Close the plot to free up memory
            plt.close(fig)

            # Save the best model
            if dice > best_metric:
                best_metric = dice
                os.makedirs(config['log']['save_dir'], exist_ok=True)
                torch.save(model.state_dict(), os.path.join(config['log']['save_dir'], "best_metric_model.pth"))
                logging.info("Saved new best metric model")

            logging.info(f"Validation Results - Epoch {epoch + 1}: Loss={val_loss:.4f}, Dice={dice:.4f}, IoU={iou:.4f}, HD95={hd95:.4f}")


        

        # # Visualize random validation examples
        # random.seed(config['train']["seed"])
        # sample_indices = random.sample(range(len(val_ds)), config['valid']['num_sample_images'])

        # inputs = torch.stack([val_ds[i]["image"] for i in sample_indices])
        # labels = torch.stack([val_ds[i]["label"] for i in sample_indices])
        # with torch.no_grad():
        #     outputs = sliding_window_inference(
        #         inputs.to(device), 
        #         config['valid']['roi_size'], 
        #         config['valid']['sw_batch_size'], 
        #         model
        #     )
        # print(f"Shape of outputs before decollation: {outputs.shape}")

        # outputs = [post_pred(i) for i in decollate_batch(outputs)]
        # fig, axes = plt.subplots(config['valid']['num_sample_images'], 3, figsize=(9, 3 * config['valid']['num_sample_images']))
        # for i, idx in enumerate(sample_indices):
        #     axes[i, 0].imshow(inputs[i, 0, :, :].cpu(), cmap="gray")
        #     axes[i, 1].imshow(labels[i].squeeze().cpu(), cmap="gray")
        #     axes[i, 2].imshow(outputs[i].squeeze().cpu(), cmap="gray", vmin=0, vmax=1)

        # # Log visualization to Weights & Biases
        # wandb.log({"val/examples": wandb.Image(fig)})

        # plt.close(fig)       

# Check for the main module
if __name__ == "__main__":
    main()
