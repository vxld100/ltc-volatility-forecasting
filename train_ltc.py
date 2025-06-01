import os
from joblib.logger import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import torch
import json
import time
import hashlib
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import mean_squared_error
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from scipy import stats
from decimal import Decimal, getcontext

def load_volatility_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')

    # Calculate log returns
    df['log_returns'] = np.log(df['close_price'] / df['close_price'].shift(1))

    # Drop the first row which will have NaN for returns
    df = df.dropna(subset=['log_returns'])
    
    # Basic info about the data
    print(f"Loaded data with {len(df)} records")
    print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
    
    return df

def scale_volatility(df, vol_vol: str):
    # Scale volatility data for the neural network
    min_vol = df[vol_vol].min()
    max_vol = df[vol_vol].max()
    df[vol_vol] = minmax_scale(df[vol_vol], feature_range=(0, 1), axis=0)

    print(f"Original Volatility Range: min={min_vol}, max={max_vol}")
    df[vol_vol] = minmax_scale(df[vol_vol], feature_range=(0, 1), axis=0)

    mse_loss_scaling_factor = (Decimal(str(max_vol)) - Decimal(str(min_vol)))**2
    print(f"MSE Loss Scaling Factor (for unscaling loss): {mse_loss_scaling_factor}")
    
    return df, min_vol, max_vol

def print_examples(dataset, num_examples=3, show_all_rows=True):
    """
    Print examples from the dataset
    
    Args:
        dataset: The dataset instance or subset
        num_examples: Number of examples to print
        show_all_rows: Whether to show all rows in the window
    """
    import pandas as pd
    
    print("===== Dataset Examples =====")
    
    # Check if it's a subset
    is_subset = isinstance(dataset, torch.utils.data.Subset)
    
    # Get the original dataset if it's a subset
    original_dataset = dataset.dataset if is_subset else dataset
    
    # Select indices evenly spaced through the dataset
    example_indices = np.linspace(0, len(dataset)-1, num_examples, dtype=int)
    
    for i, idx in enumerate(example_indices):
        # Get the data sample
        if is_subset:
            # For a subset, get the original index first
            original_idx = dataset.indices[idx]
            x, y = dataset[idx]
            # Get dates from the original dataset using the original index
            input_dates, target_date = original_dataset.get_dates(original_idx)
        else:
            x, y = dataset[idx]
            input_dates, target_date = dataset.get_dates(idx)
        
        print(f"\nExample {i+1}:")
        print(f"Input window dates: {input_dates[0]} to {input_dates[-1]}")
        print(f"Target date: {target_date}")
        
        # Print the input features
        print("\nInput features and time deltas:")
        
        # Print all rows or just a sample based on show_all_rows
        row_indices = range(x.shape[0]) if show_all_rows else list(range(3)) + ([] if x.shape[0] <= 4 else [x.shape[0]-1])
        
        print(f"{'Date':<12} | {'Features':<30} | {'Time Delta':<10}")
        print("-" * 60)
        
        for j in row_indices:
            # Format date as readable string
            try:
                date_str = pd.Timestamp(input_dates[j]).strftime('%Y-%m-%d')
            except:
                date_str = f"Date {j}"
                
            # Get features excluding the time delta (which is the last column)
            features = x[j, :-1].numpy()
            feature_str = ", ".join([f"{val:.4f}" for val in features])
            
            # Get time delta (last column)
            time_delta = x[j, -1].item()
            
            print(f"{date_str:<12} | {feature_str:<30} | {time_delta:.4f}")
            
            if not show_all_rows and j == 2 and x.shape[0] > 4:
                print("..." + " " * 57)
        
        print("-" * 60)
        print(f"Target value: {y.item():.6f}")
        
        # Explanation of time deltas
        print("\nTime delta explanation:")
        print("- Each time delta represents days between the current row and the next row")
        print("- The last time delta is a placeholder (usually 1.0)")
    
    print("\n===== End of Examples =====")

class VolatilityDatasetTimedelta(Dataset):
    def __init__(
            self, 
            dataframe, 
            window_size=20, 
            target_col='daily_volatility', 
            feature_cols=['daily_volatility', 'log_returns'],
            include_time_deltas: bool = True,
            time_delta_scale : int = 100
            ):
        """
        Args:
            dataframe: Pandas DataFrame with DATE and volatility columns
            window_size: Number of days to use as input
            target_col: Column name for the target volatility metric
            feature_cols: List of column names to use as features
        """
        self.dates = dataframe['DATE'].values
        self.target = dataframe[target_col].values
        
        if include_time_deltas:
            # Calculate time deltas in days between adjacent samples
            #date_diffs = np.diff(dataframe['DATE'].astype(np.int64))

            # Convert nanoseconds to days
            #time_deltas = date_diffs / (86400 * 1e9)  # Convert nanoseconds to days
            # Add a placeholder for the first element
            #time_deltas = np.insert(time_deltas, 0, 1.0)

            time_deltas = dataframe['DATE'].diff().dt.days
            time_deltas.fillna(1.0, inplace=True)

            time_deltas = np.log(time_deltas * time_delta_scale)
        else:
            time_deltas = np.ones(len(self.dates))
            time_deltas = time_deltas * time_delta_scale
        
        # Store original features
        self.features = dataframe[feature_cols].values
        self.time_deltas = time_deltas
        self.window_size = window_size
        
        # For 1-step ahead prediction
        self.total_samples = len(self.target) - window_size
        
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Input window: previous window_size days with features
        feature_window = self.features[idx:idx + self.window_size]
        
        # Time deltas for the window (exclude the first one which is just a placeholder)
        time_window = self.time_deltas[idx:idx + self.window_size]
        #time_window = self.time_deltas[idx+1:idx + self.window_size+1]
        time_window = np.expand_dims(time_window, axis=1)  # Add feature dimension
        
        # Combine features and time deltas
        x = np.concatenate([feature_window, time_window], axis=1)
        
        # Target: next day after the window
        y = self.target[idx + self.window_size]
        
        # Convert to tensors
        x = torch.FloatTensor(x)
        y = torch.FloatTensor([y])
        
        return x, y
    
    def get_dates(self, idx):
        """Return dates for the input window and target day"""
        input_dates = self.dates[idx:idx + self.window_size]
        target_date = self.dates[idx + self.window_size]
        return input_dates, target_date
    
    def get_target_date(self, idx):
        """Return just the target date (for splitting)"""
        return self.dates[idx + self.window_size]

# Create train and validation datasets by first creating all pairs, then splitting
def create_splits(
        vol_dataset, 
        val_ratio=0.2, 
        test_ratio=0.0, 
        ):
    """
    Create train and validation datasets with a proper chronological split
    
    Args:
        file_path: Path to CSV file
        window_size: Size of the window for prediction
        val_ratio: Proportion of data to use for validation (e.g., 0.2 = 20%)
        test_ratio: proportion of data to use for test (e.g., 0.2 = 20%)
        target_col: Column to predict
        feature_cols: List of columns to use as input features
        
    Returns:
        train_dataset, val_dataset, test_dataset: The split datasets
    """
    # Determine the split point based on target dates
    train_split = int(len(vol_dataset) * (1 - (val_ratio + test_ratio)))
    val_split = int(len(vol_dataset) * (1 - test_ratio))
    
    # Create indices for train and validation
    train_indices = list(range(train_split))
    val_indices = list(range(train_split, val_split))
    test_indices = list(range(val_split, len(vol_dataset)))
    
    # Create subsets for train and validation
    train_dataset = Subset(vol_dataset, train_indices)
    val_dataset = Subset(vol_dataset, val_indices)
    test_dataset = Subset(vol_dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

# Create DataLoaders
def create_dataloaders(dataset, batch_size=32, shuffle=False):
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,  # Shuffle training data
        num_workers=7,
    )
    
    return data_loader

def one_step_ahead_dm_test(actual, forecast1, forecast2, alternative='less', precision=28):
    """
    Compute the one-sided Diebold-Mariano test statistic for comparing one-step-ahead forecasts
    using Decimal for high precision arithmetic with very small values.
    
    Parameters:
    -----------
    actual : array-like
        The actual values (ground truth)
    forecast1 : array-like
        First forecast values
    forecast2 : array-like
        Second forecast values
    alternative : str, optional
        Alternative hypothesis, 'less' means forecast1 is better than forecast2
        'greater' means forecast2 is better than forecast1
    precision : int, optional
        Decimal precision to use (default: 28)
    
    Returns:
    --------
    dm_stat : float
        The Diebold-Mariano test statistic
    p_value : float
        The p-value for the test
    """
    # Set decimal precision
    getcontext().prec = precision
    
    # Convert all inputs to numpy arrays
    actual = np.asarray(actual)
    forecast1 = np.asarray(forecast1)
    forecast2 = np.asarray(forecast2)
    
    # Create empty lists to store Decimal values
    d = []
    
    # Compute the squared errors using Decimal
    for i in range(len(actual)):
        # Convert values to Decimal for precise calculation
        act = Decimal(str(actual[i]))
        f1 = Decimal(str(forecast1[i]))
        f2 = Decimal(str(forecast2[i]))
        
        # Calculate MSE for each forecast
        err1 = (act - f1) ** 2
        err2 = (act - f2) ** 2
        
        d.append(err1 - err2)
    
    # Number of observations
    n = len(d)
    
    # Calculate the mean of the loss differential
    mean_d = sum(d) / Decimal(n)
    
    # Calculate variance using Decimal
    squared_diffs = [((x - mean_d) ** 2) for x in d]
    variance = sum(squared_diffs) / Decimal(n - 1)  # Using unbiased estimator
    
    # Calculate the standard error of the mean
    std_error = (variance / Decimal(n)).sqrt()
    
    # Diebold-Mariano test statistic
    dm_stat = float(mean_d / std_error)
    
    # Compute p-value based on alternative hypothesis
    if alternative == 'less':
        # Test if forecast1 is better than forecast2
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == 'greater':
        # Test if forecast2 is better than forecast1
        p_value = 1 - stats.norm.cdf(dm_stat)
    else:
        raise ValueError("alternative must be 'less' or 'greater'")
    
    return dm_stat, p_value


class VolatilityTrainer(pl.LightningModule):
    def __init__(self, model, configuration: dict):
        super().__init__()
        self.model = model
        # called it configuration to differentiate it from model parameters and avoid confusion
        self.configuration = configuration
        self.loss_fn = nn.MSELoss()
        
        # Initialize metrics for distributed validation
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        # Split the input: Last column is time delta, rest are features
        features = x[:, :, :-1]
        time_deltas = x[:, :, -1:]  
        
        # Make sure time_deltas has shape [batch_size, seq_len, 1]
        if time_deltas.dim() == 2:
            time_deltas = time_deltas.unsqueeze(-1)
        
        # Pass to model using the built-in handling
        y_hat, _ = self.model(features, timespans=time_deltas)
        
        # Get only the last time step prediction
        y_hat = y_hat[:, -1, :]
        
        return y_hat
    
    def _shared_step(self, batch, batch_idx):
        """Shared step used by training, validation, and test steps"""
        x, y = batch
        # Use the forward method which handles batch processing and timespan splitting
        y_hat = self.forward(x) # y_hat shape should be [batch_size, output_dim]
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y
    
    def training_step(self, batch, batch_idx):
        loss, _,  _ = self._shared_step(batch, batch_idx)
        # Use sync_dist=True to synchronize metrics across GPUs
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss,  _, _ = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # Store for end of epoch processing
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}
    
    def on_validation_epoch_end(self):
        # Optional: Manual aggregation if needed for specific purposes
        if self.validation_step_outputs: # Check if list is not empty
             avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
             # self.log("val_loss_manual", avg_loss, prog_bar=True, sync_dist=True) # Example: log manually calculated too
             self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        if self.configuration["optimizer"]["name"] == "Adam":
            optimizer = torch.optim.Adam(
                    self.model.parameters(), 
                    lr=self.configuration["optimizer"]["learning_rate"],
                    weight_decay=self.configuration["optimizer"]["weight_decay"]
                )
        elif self.configuration["optimizer"]["name"] == "AdamW":
            optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=self.configuration["optimizer"]["learning_rate"], 
                    weight_decay=self.configuration["optimizer"]["weight_decay"]
                )
        
        # Add a learning rate scheduler
        if self.configuration["scheduler"]["name"] == "ReduceLROnPlateau":
            sched_conf = self.configuration["scheduler"]
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_conf.get("mode"),          # Reduce LR when val_loss stops decreasing
                factor=sched_conf.get("factor"),          # Multiply LR by this factor (0.5 = half the LR)
                patience=sched_conf.get("patience"),         # Wait this many epochs with no improvement before reducing
                min_lr=sched_conf.get("min_lr"),          # Don't reduce LR below this value
                verbose=True
            )
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": sched_conf.get("monitor"),
                "interval": sched_conf.get("interval"),
                "frequency": sched_conf.get("frequency")
            }

        elif self.configuration["scheduler"]["name"] == "CosineAnnealing":
            sched_conf = self.configuration["scheduler"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sched_conf.get("T_max"),  # Number of epochs/iterations for a complete cycle
                eta_min=sched_conf.get("min_lr"),  # Minimum learning rate
                verbose=True
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": sched_conf.get("interval"),  # "epoch" or "step"
                "frequency": sched_conf.get("frequency")
            }
        
        elif self.configuration["scheduler"]["name"] == "CosineAnnealingWarmRestarts":
            sched_conf = self.configuration["scheduler"]
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sched_conf.get("T_0"),  # Number of iterations for the first restart
                T_mult=sched_conf.get("T_mult"),  # Multiplicative factor for T_i after a restart
                eta_min=sched_conf.get("min_lr"),  # Minimum learning rate
                verbose=True
            )
            scheduler_config = {
                "scheduler": scheduler,
                "interval": sched_conf.get("interval"),  # "epoch" or "step"
                "frequency": sched_conf.get("frequency")
            }

        elif self.configuration["scheduler"]["name"] == "OneCycleLR":
            sched_conf = self.configuration["scheduler"]
            
            # You need to provide either total_steps OR (epochs AND steps_per_epoch)
            total_steps = sched_conf.get("total_steps", None)
            epochs = sched_conf.get("epochs", None)
            steps_per_epoch = sched_conf.get("steps_per_epoch", None)
            
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.configuration["optimizer"]["learning_rate"] * sched_conf.get("max_lr_mult"),  # Often 10x base LR
                total_steps=sched_conf.get("total_steps"),
                epochs=sched_conf.get("epochs"),
                steps_per_epoch=sched_conf.get("steps_per_epoch"),
                pct_start=sched_conf.get("pct_start"),  # 30% of training in warmup
                div_factor=sched_conf.get("div_factor"),  # Initial LR = max_lr/25
                final_div_factor=sched_conf.get("final_div_factor"),  # Final LR = initial_lr/10000
                anneal_strategy=sched_conf.get("anneal_strategy")  # 'cos' is usually better than 'linear'
            )
            
            scheduler_config = {
                "scheduler": scheduler,
                "interval": sched_conf.get("interval"),  # OneCycleLR must be updated every batch
                "frequency": sched_conf.get("frequency")
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


def get_predictions(model, dataloader):
    """
    Get predictions and true values from a dataloader (runs on CPU).

    Args:
        model: The trained PyTorch Lightning model (VolatilityTrainer instance).
        dataloader: The DataLoader to get predictions from (e.g., test_loader).

    Returns:
        tuple: (numpy array of predictions, numpy array of true values)
    """
    model.eval()  # Set model to evaluation mode
    # Ensure model is on CPU (should be if trainer used CPU)
    model.cpu()
    all_preds = []
    all_true = []

    with torch.no_grad():  # Disable gradient calculations
        for batch in dataloader:
            # Data is expected to be on CPU from the dataloader
            x, y = batch
            y_hat = model(x)
            all_preds.append(y_hat.numpy())
            all_true.append(y.numpy())

    # Concatenate predictions and true values from all batches
    predictions = np.concatenate(all_preds, axis=0).squeeze() # Remove single dimension if present
    true_values = np.concatenate(all_true, axis=0).squeeze() # Remove single dimension if present

    # Ensure they are 1D arrays
    if predictions.ndim > 1:
         predictions = predictions.flatten()
    if true_values.ndim > 1:
         true_values = true_values.flatten()

    return predictions, true_values

def plot_training_loss(metrics_path: str, plot_path: str):
    """
    Reads metrics from a CSV file logged by PyTorch Lightning's CSVLogger
    and plots the training and validation loss curves.

    Args:
        metrics_path: Path to the metrics.csv file.
        plot_path: Path where the plot image should be saved.
    """
    if not os.path.exists(metrics_path):
        print(f"Warning: Could not find metrics file at {metrics_path}")
        return # Exit if file doesn't exist

    try:
        metrics_df = pd.read_csv(metrics_path)

        # --- Prepare Training Loss Data ---
        # Filter for steps where train_loss was recorded
        train_loss_data = metrics_df[['step', 'train_loss']].dropna(subset=['train_loss'])
        if train_loss_data.empty:
            print("Warning: No 'train_loss' values found in metrics file.")
            # Decide if you want to plot only validation or exit
            # return # Or continue to plot validation if available

        # --- Prepare Validation Loss Data ---
        # Filter for epochs where val_loss was recorded
        val_loss_data = metrics_df[['epoch', 'step', 'val_loss']].dropna(subset=['val_loss'])

        # Find the step corresponding to the end of each epoch for alignment
        # Need to handle potential multiple entries per epoch if logged differently, max step is usually safe
        if not val_loss_data.empty:
            # Get the maximum step for each epoch where val_loss is present
            epoch_end_steps = val_loss_data.loc[val_loss_data.groupby('epoch')['step'].idxmax()]
            # Alternative: If val_loss logged only once per epoch, simple dropna might suffice
            # epoch_end_steps = metrics_df[['epoch', 'step', 'val_loss']].dropna(subset=['val_loss'])

        else:
             print("Warning: No 'val_loss' values found in metrics file.")
             epoch_end_steps = pd.DataFrame() # Empty dataframe if no validation loss


        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid') # Example style, choose one you like
        plt.figure(figsize=(12, 8)) # Slightly adjusted size

        # Plot Training Loss (if available)
        if not train_loss_data.empty:
            plt.plot(
                train_loss_data['step'],
                train_loss_data['train_loss'],
                label='Training Loss (per Step)',
                color='forestgreen',      # Example: Use a standard blue color
                linestyle='-',       # Example: Solid line
                linewidth=1.5,       # Slightly thicker line
            )

        # Plot Validation Loss (if available)
        if not epoch_end_steps.empty:
            plt.plot(
                epoch_end_steps['step'],
                epoch_end_steps['val_loss'],
                label='Validation Loss (per Epoch)',
                color='yellow',       # Example: Use a standard red color
                linestyle='-',      # Example: Dashed line
                linewidth=1.5,       # Match training line width
            )

        # --- Customize Plot ---
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Loss (Scaled MSE)", fontsize=12)
        plt.yscale('log') # Keep log scale if desired, comment out for linear
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5) # Customize grid
        plt.tight_layout() # Adjust layout to prevent labels overlapping

        # --- Save Plot ---
        try:
            plt.savefig(plot_path, dpi=300) # Save with higher resolution
            print(f"Training loss plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")

        plt.close() # Close the figure to free memory

    except pd.errors.EmptyDataError:
        print(f"Warning: Metrics file is empty: {metrics_path}")
    except Exception as e:
        print(f"Error reading or plotting metrics file '{metrics_path}': {e}")

def plot_model_volatility_comparison(
        df, 
        plot_path : str,
        date_col="DATE",
        vol1_col="daily_volatility", 
        vol2_col="LTC_predictions",
        vol3_col="realGARCH_predictions",
        vol1_label="Ground Truth", 
        vol2_label="LTC predictions",
        vol3_label="realGARCH predictions",
        figsize=(12, 8)
      ):
    # Ensure date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot both volatility measures with the same line style
    ax.plot(df[date_col], df[vol1_col], marker=',', linestyle='-', label=vol1_label, color='royalblue', linewidth=1)
    ax.plot(df[date_col], df[vol2_col], marker=',', linestyle='-', label=vol2_label, color='orangered', linewidth=1)
    ax.plot(df[date_col], df[vol3_col], marker=',', linestyle='-', label=vol3_label, color='mediumorchid', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Realized Variance', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show dates properly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=180))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Sets approximately 5 ticks
    plt.xticks(rotation=45)
    
    # Add legend
    ax.legend()
    
    # Ensure everything fits nicely
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300) # Save with higher resolution


# Example usage
if __name__ == "__main__":
    file_path = "./data/output/ibm_realized_vol_prc_2000-2024.csv"
    early_stop_callback_params = {
            "name" : "EarlyStopping",
            "monitor" : "val_loss",
            "patience" : 25
            }
    adam_optimizer_params = {
            "name" : "Adam",
            "learning_rate" : 1e-4,
            "weight_decay" : 1e-5,
            }
    reduce_on_plateau_scheduler_params = {
            "name" : "ReduceLROnPlateau",
            "mode" : "min",
            "factor" : 0.5,
            "patience" : 5,
            "min_lr" : 1e-6,
            "monitor" : "val_loss",
            "interval" : "epoch",
            "frequency" : 1,
            }
    cosine_annealing_params = {
            "name": "CosineAnnealing",
            "T_max": 10,  # Complete cycle length
            "min_lr": 1e-6,
            "interval": "epoch",
            "frequency" : 1,
            }
    cosine_annealing_warm_rest_params = {
            "name": "CosineAnnealingWarmRestarts",
            "T_0": 5,  # Initial restart cycle length
            "T_mult": 2,  # Each cycle gets twice as long
            "min_lr": 1e-6,
            "verbose": True,
            "interval": "epoch",
            "frequency" : 1,
            }
    one_cycle_lr_params = {
            "name": "OneCycleLR",
            "max_lr_mult": 10,  # Peak learning rate (10x base is common)
            "epochs": 10,  # Total number of epochs
            "total_steps": 10_000,
            "steps_per_epoch": 100,  # Batches per epoch
            "pct_start": 0.3,  # 30% for warmup
            "div_factor": 25,
            "final_div_factor": 10000,
            "anneal_strategy": "cos",
            "interval" : "step",
            "frequency" : 1
            }
    parameters = {
            "window_size" : 10,
            "batch_size" : 40,
            "max_epochs" : 250,
            "devices" : 1, # number of CPU cores
            "feature_cols": ['daily_volatility'], # alternative ['daily_volatility', 'log_returns']
            "shuffle_train": False,
            "neurons" : 15,
            "include_timdeltas" : True,
            "time_delta_scale" : 20,
            "time_scale" : "log",
            "batch_first" : True,
            "ode_unfolds" : 12,
            "gradient_clip_val" : 1,
            "callbacks" : [early_stop_callback_params],
            "optimizer" : adam_optimizer_params,
            "scheduler" : cosine_annealing_params,
            }
    
    target_col = 'daily_volatility'
    
    data = load_volatility_data(file_path)
    data_scaled, min_vol, max_vol = scale_volatility(data.copy(), vol_vol='daily_volatility')
    # Create train and validation datasets
    vol_dataset = VolatilityDatasetTimedelta(
        dataframe=data_scaled,
        window_size=parameters['window_size'], 
        target_col=target_col,
        feature_cols=parameters['feature_cols'],
        include_time_deltas=parameters['include_timdeltas'],
        time_delta_scale=parameters['time_delta_scale'],
    )
    train_dataset, val_dataset, test_dataset = create_splits(
        vol_dataset, 
        val_ratio=0.1,
        test_ratio=0.2,
    )

    #print_examples(train_dataset, num_examples=3)
    
    # Create DataLoaders
    train_loader = create_dataloaders(train_dataset, batch_size=parameters['batch_size'], shuffle=parameters['shuffle_train'])
    val_loader = create_dataloaders(val_dataset, batch_size=parameters['batch_size'], shuffle=False)
    test_loader = create_dataloaders(test_dataset, batch_size=parameters['batch_size'], shuffle=False)

    # Define the model
    out_features = 1
    in_features = len(parameters['feature_cols'])  # Number of input features 

    # Create the wiring and LTC model
    wiring = AutoNCP(parameters['neurons'], out_features)
    ltc_model = LTC(in_features, wiring, batch_first=parameters['batch_first'], ode_unfolds=parameters['ode_unfolds'])
    
    # Create the trainer
    learn = VolatilityTrainer(ltc_model, configuration=parameters)

    callbacks = list()
    for callback_params in parameters["callbacks"]:
        if callback_params["name"] == "EarlyStopping":
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor=callback_params.get("monitor"),
                    patience=callback_params.get("patience"),
                    verbose=True
                    )
                )

    # Add Model Checkpoint callback (optional but recommended)
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='best-volatility-model-{epoch:02d}-{val_loss:.6f}',
            save_top_k=1,
            mode='min',
        )
    )

    csv_logger = CSVLogger(".")
    log_dir = csv_logger.log_dir # Store log directory path
    
    # Training trainer
    trainer = pl.Trainer(
        max_epochs=parameters['max_epochs'],
        log_every_n_steps=10,
        devices=parameters['devices'],
        gradient_clip_val=parameters['gradient_clip_val'],
        callbacks=callbacks,
        logger=csv_logger
    )

    # Train the model
    print("--- Starting Training ---")
    start = time.time()
    trainer.fit(learn, train_loader, val_loader)
    end = time.time()
    train_time = end - start
    n_epochs = trainer.current_epoch
    print("--- Training Finished ---")


    print("\n--- Getting Predictions for DM Test ---")
    # Load the best model checkpoint for prediction if not already loaded by .test
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    # Need to instantiate the model structure again before loading state_dict
    inference_model = VolatilityTrainer.load_from_checkpoint(
        best_model_path,
        model=ltc_model, # Provide the model structure
        configuration=parameters,
    )

    ltc_preds_scaled, _ = get_predictions(inference_model, test_loader)
    print(f"Generated {len(ltc_preds_scaled)} LTC predictions.")

    # Unscale predictions and true values
    def unscale(scaled_values, min_v, max_v):
        return scaled_values * (max_v - min_v) + min_v

    ltc_preds_unscaled = unscale(ltc_preds_scaled, min_vol, max_vol)

    val_split = int(len(data) * 1 - 0.156)
    results_df = data.iloc[val_split:]
    # Get the dates your model actually predicted
    test_dates = [test_dataset.dataset.get_target_date(i) for i in test_dataset.indices]

    # Create a clean DataFrame with predictions
    results_df = pd.DataFrame({
        'DATE': test_dates,
        'LTC_predictions': ltc_preds_unscaled
    })

    # Now merge with original data to get other columns
    results_df = results_df.merge(data, on='DATE', how='left')

    rgarch_df = pd.read_csv("./data/rgarch_test_preds_v2.csv")
    rgarch_df['DATE'] = pd.to_datetime(rgarch_df['DATE'])

    comparison_df = pd.merge(results_df, rgarch_df, on='DATE', how='inner')
    print(f"Aligned {len(comparison_df)} predictions for comparison.")

    crisis = comparison_df[comparison_df['DATE'] < '2020-01-01']
    after_crisis = comparison_df[comparison_df['DATE'] >= '2020-01-01']

    print("\n--- Performing Diebold-Mariano Test ---")
    global_dm_stat, global_p_value = one_step_ahead_dm_test(
            actual=comparison_df['daily_volatility'].values, # ground truth
            forecast1=comparison_df['LTC_predictions'].values,
            forecast2=comparison_df['realGARCH_predictions'].values,
            alternative='less', # See if P1 outperforms P2
        )
    crisis_dm_stat, crisis_p_value = one_step_ahead_dm_test(
            actual=crisis['daily_volatility'],
            forecast1=crisis['LTC_predictions'],
            forecast2=crisis['realGARCH_predictions'],
            alternative='less'
            )
    after_crisis_dm_stat, after_crisis_p_value = one_step_ahead_dm_test(
            actual=after_crisis['daily_volatility'],
            forecast1=after_crisis['LTC_predictions'],
            forecast2=after_crisis['realGARCH_predictions'],
            alternative='less'
            )
    print("--- Finished Diebold-Mariano Test ---")

    #ltc_loss = mean_squared_error(comparison_df['daily_volatility'].values, comparison_df['LTC_predictions'].values)
    global_ltc_loss = mean_squared_error(comparison_df['daily_volatility'], comparison_df['LTC_predictions'])
    global_rgarch_loss = mean_squared_error(comparison_df['daily_volatility'], comparison_df['realGARCH_predictions'])

    crisis_ltc_loss = mean_squared_error(crisis['daily_volatility'], crisis['LTC_predictions'])
    crisis_rgarch_loss = mean_squared_error(crisis['daily_volatility'], crisis['realGARCH_predictions'])

    after_crisis_ltc_loss = mean_squared_error(after_crisis['daily_volatility'], after_crisis['LTC_predictions'])
    after_crisis_rgarch_loss = mean_squared_error(after_crisis['daily_volatility'], after_crisis['realGARCH_predictions'])

    # The hash for the file is calculated only on the parameters and not also on the result
    # (which would make it unique) because two runs with the same parameters should be recognizeable
    # from their filename
    parameters_hash = hashlib.md5(str(parameters).encode()).hexdigest()[:10]

    metrics_path = os.path.join(log_dir, "metrics.csv")
    try:
        metrics_df = pd.read_csv(metrics_path)
        best_val_loss = metrics_df['val_loss'].min()
        best_val_loss_scaled = round(best_val_loss * 1e3, 15)
        best_val_loss_scaled = str(best_val_loss_scaled).replace(".", "0")
    except Exception as e:
        print(f"Could not extract val_loss from logs: {e}")

    def count_parameters(model):
        """Count trainable and total parameters in a PyTorch model"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        return trainable_params, total_params

    # Use it after model initialization
    trainable_params, total_params = count_parameters(learn.model)

    global_results = {
            "LTC_loss" : global_ltc_loss, 
            "realGARCH_loss" : global_rgarch_loss, 
            "diebold_mariano_stat" : global_dm_stat,
            "p_value" : global_p_value
            }
    crisis_results = {
            "LTC_loss" : crisis_ltc_loss, 
            "realGARCH_loss" : crisis_rgarch_loss, 
            "diebold_mariano_stat" : crisis_dm_stat,
            "p_value" : crisis_p_value
            }
    after_crisis_results = {
            "LTC_loss" : after_crisis_ltc_loss, 
            "realGARCH_loss" : after_crisis_rgarch_loss, 
            "diebold_mariano_stat" : after_crisis_dm_stat,
            "p_value" : after_crisis_p_value
            }

    result = {
            "global" : global_results,
            "crisis" : crisis_results,
            "after_crisis" : after_crisis_results,
            "train_time" : train_time,
            "epochs" : n_epochs,
            "best_val_loss" : best_val_loss,
            "total_params" : total_params,
            "trainable_params" : trainable_params,
            "parameters" : parameters,
            }

    # The name of the directory is ordered by validation loss and not test loss, for the validation loss is 
    # what determines which model is expected to be best, and hence should be chosen.
    results_dir = f"./results/{best_val_loss_scaled}-{parameters_hash}" 
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    loss_plot_path = os.path.join(results_dir, "training_loss.pdf")
    comparison_plot_path = os.path.join(results_dir, "model_comparison.pdf")

    plot_training_loss(metrics_path=metrics_path, plot_path=loss_plot_path)
    plot_model_volatility_comparison(comparison_df, plot_path=comparison_plot_path)

    comparison_path = os.path.join(results_dir, "comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)

    shutil.copy2(best_model_path, results_dir)

    with open(f"{results_dir}/{best_val_loss_scaled}-{parameters_hash}.json", "w") as parameters_file:
        json.dump(result, parameters_file, indent=2)

