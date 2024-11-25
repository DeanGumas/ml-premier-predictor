import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from typing import Tuple, List
from final_project.cnn.preprocess import generate_cnn_data, split_preprocess_cnn_data
from final_project.cnn.evaluate import plot_learning_curve, eval_cnn, log_evals

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES


device = "cuda"    

class CNNModel(nn.Module):
    def __init__(self, 
                 X_input_shape: Tuple, 
                 d_input_shape: Tuple, 
                 kernel_size: int, 
                 num_filters: int, 
                 num_dense: int, 
                 conv_activation: str = 'relu', 
                 dense_activation: str = 'relu', 
                 regularization: float = 0.001):
        
        super(CNNModel, self).__init__()
        
        self.flatten = nn.Flatten()

        # Set up convolutional layer with L1L2 regularization (equivalent)
        self.conv1d = nn.Conv1d(in_channels=X_input_shape[0], 
                                out_channels=num_filters, 
                                kernel_size=kernel_size)
        
        self.convolutional_stack = nn.Sequential(
            nn.Conv1d(in_channels=X_input_shape[0], 
                                out_channels=X_input_shape[0], 
                                kernel_size=kernel_size),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=X_input_shape[0], 
                                out_channels=num_filters, 
                                kernel_size=kernel_size-1),
            nn.AvgPool1d(kernel_size=2, stride=2)
        )

        # Attempt to combine both linear layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(num_filters + d_input_shape[0], num_dense),
            nn.ReLU(),
            nn.Linear(num_dense, 1)
        )

        # Store activations
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        
        # Regularization (L1L2 applied in optimizer)
        self.regularization = regularization

    def forward(self, x_input, d_input):
        # Pass through Conv1D layer
        #print(f"x_input size before conv: {x_input.size()}")
        #x = self.flatten(x, dim=1)
        x = self.convolutional_stack(x_input)  # Shape: (batch_size, num_filters, new_length)

        #print(f"x size after conv: {x.size()}")
        # Flatten only the non-batch dimensions
        x = torch.flatten(x, start_dim=1)  # Shape: (batch_size, flattened_features)
        #print(f"x size after flatten: {x.size()}")

        # Ensure d_input is (batch_size, 1) before concatenating
        #print(f"d_input size before adjustment: {d_input.size()}")
        if d_input.dim() == 1:  
            d_input = d_input.unsqueeze(1)
        #print(f"d_input size after adjustment: {d_input.size()}")

        # Concatenate along the feature dimension
        x = torch.cat((x, d_input), dim=1)  # Shape: (batch_size, flattened_features + 1)
        #print(f"x size after adjustment: {x.size()}")

        # Dense layer
        #x = self.dense1(x)
        #if self.dense_activation == 'relu':
        #    x = relu(x)

        # Output layer
        #x = self.output_layer(x).squeeze(-1)

        # Combined dense/output relu stack
        x = self.linear_relu_stack(x).squeeze(-1)
        return x

def create_cnn(X_input_shape: Tuple, 
               d_input_shape: Tuple, 
               kernel_size: int, 
               num_filters: int, 
               num_dense: int, 
               conv_activation: str = 'relu', 
               dense_activation: str = 'relu', 
               optimizer: str = 'adam', 
               learning_rate: float = 0.001, 
               loss: str = 'mse', 
               metrics: List[str] = ['mae'], 
               verbose: bool = False, 
               regularization: float = 0.001):
    
    if verbose:
        print("====== Building CNN Architecture ======")

    # Initialize model
    model = CNNModel(X_input_shape, d_input_shape, kernel_size, num_filters, num_dense, 
                     conv_activation, dense_activation, regularization).to(device)

    
    # Set up optimizer with L2 regularization as weight_decay
    if optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=regularization)
    else:
        raise ValueError("Unsupported optimizer type. Use 'adam' or 'sgd'")
    
    # Set loss function
    if loss == 'mse':
        loss_fn = nn.MSELoss()
    elif loss == 'mae':
        loss_fn = nn.L1Loss()
    else:
        raise ValueError("Unsupported loss type. Use 'mse' or 'mae'")
    
    if verbose:
        print("====== Done Building CNN Architecture ======")
    
    return model, optimizer, loss_fn

def generate_datasets(data_dir: str,
              season: str, 
              position: str, 
              window_size: int,
              drop_low_playtime : bool = True,
              low_playtime_cutoff : int = 25,
              num_features: List[str] = STANDARD_NUM_FEATURES,
              cat_features: List[str] = STANDARD_CAT_FEATURES, 
              test_size: float = 0.15, 
              val_size: float = 0.3, 
              stratify_by: str = 'skill',
              standardize: bool = True,
              verbose: bool = False):
    
    # =========== Generate CNN Dataset  ============
    # == for Desired Season, Posn, Window Size =====
    # ===== and Feature Engineering Settings =======
    df, features_df = generate_cnn_data(data_dir=data_dir,
                         season=season, 
                         position=position, 
                         window_size=window_size,
                         num_features=num_features, 
                         cat_features=cat_features,
                         drop_low_playtime=drop_low_playtime,
                         low_playtime_cutoff=low_playtime_cutoff,
                         verbose = verbose)
    
    (X_train, d_train, y_train, 
     X_val, d_val, y_val, 
     X_test, d_test, y_test, pipeline) = split_preprocess_cnn_data(df, 
                                                            features_df, 
                                                            test_size=test_size,
                                                            val_size=val_size,
                                                            stratify_by=stratify_by, 
                                                            num_features=num_features,
                                                            cat_features=cat_features,
                                                            standardize=standardize,
                                                            return_pipeline=True,
                                                            verbose=verbose)
    
    return X_train, d_train, y_train, X_val, d_val, y_val, X_test, d_test, y_test, pipeline  #return split data and stdscale pipe

# Custom Early Stopping
class EarlyStopping:
    def __init__(self, patience=40, tolerance=1e-5, verbose=False):
        self.patience = patience
        self.tolerance = tolerance
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.tolerance:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def plot_learning_curve(history, season, position):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f"Learning Curve for Season {season}, Position {position}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def build_train_cnn(X_train, d_train, y_train,
                    X_val, d_val, y_val,
                    X_test, d_test, y_test,
                    season: str,
                    position: str,
                    window_size: int,
                    kernel_size: int,
                    num_filters: int,
                    num_dense: int,
                    batch_size: int = 50,
                    epochs: int = 500,
                    drop_low_playtime: bool = True,
                    low_playtime_cutoff: int = 25,
                    num_features: List[str] = STANDARD_NUM_FEATURES,
                    cat_features: List[str] = STANDARD_CAT_FEATURES, 
                    conv_activation: str = 'relu',
                    dense_activation: str = 'relu',
                    optimizer: str = 'adam',
                    learning_rate: float = 0.001,
                    loss: str = 'mse',
                    metrics: List[str] = ['mae'],
                    verbose: bool = False,
                    regularization: float = 0.001,
                    early_stopping: bool = False,
                    tolerance: float = 1e-5,
                    patience: int = 40,
                    plot: bool = False,
                    draw_model: bool = False, 
                    standardize: bool = True):

    # Store training history for plotting
    history = {'train_loss': [], 'val_loss': []}

    # Prepare data loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(d_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                torch.tensor(d_val, dtype=torch.float32), 
                                torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set input shapes
    X_input_shape = (window_size, X_train.shape[2])
    d_input_shape = (1,)
    # Initialize model, optimizer, and loss function
    model, optimizer, loss_fn = create_cnn(X_input_shape=X_input_shape, 
                                           d_input_shape=d_input_shape,
                                           kernel_size=kernel_size,
                                           num_filters=num_filters,
                                           num_dense=num_dense,
                                           conv_activation=conv_activation,
                                           dense_activation=dense_activation,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           loss=loss,
                                           metrics=metrics,
                                           regularization=regularization,
                                           verbose=verbose)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, tolerance=tolerance, verbose=verbose) if early_stopping else None

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, d_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            d_batch = d_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch, d_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        # Validation loop
        model.eval()
        val_loss = 0.0
        y_pred_list = []
        y_true_list = []
        with torch.no_grad():
            for X_val_batch, d_val_batch, y_val_batch in val_loader:
                X_val_batch = X_val_batch.to(device)
                d_val_batch = d_val_batch.to(device)
                y_val_batch = y_val_batch.to(device)
                output = model(X_val_batch, d_val_batch)
                loss = loss_fn(output, y_val_batch)
                val_loss += loss.item() * X_val_batch.size(0)
                y_pred_list.extend(output.cpu().numpy())
                y_true_list.extend(y_val_batch.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        y_pred = np.array(y_pred_list).flatten()
        y_true = np.array(y_true_list).flatten()
        
        # Calculate additional metrics if required
        if 'mae' in metrics:
            val_mae = mean_absolute_error(y_true, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val MAE: {val_mae}")
        
        # Early stopping check
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    # Evaluate model on test set
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    d_test_tensor = torch.tensor(d_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    #print(f'Test Loss (MSE): {X_test_tensor.size()}')
    #print(f'Test Loss (MSE): {d_test_tensor.size()}')
    #print(f'Test Loss (MSE): {y_test_tensor.size()}')
    with torch.no_grad():
        output = model(X_test_tensor, d_test_tensor)
        
    y_test_pred = output.cpu().numpy().flatten()
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = loss_fn(output, y_test_tensor)
    print(f'Test Loss (MSE): {test_mse}')
    print(f'Test Mean Absolute Error (MAE): {test_mae}')
    
    # Draw model if specified
    plot_learning_curve(history, season, position)

    return model, {'test_mae': test_mae}

def full_cnn_pipeline(data_dir: str, 
                    season: str, 
                    position: str,  
                    window_size: int,
                    kernel_size: int,
                    num_filters: int,
                    num_dense: int,
                    batch_size: int = 50,
                    epochs: int = 500,  
                    drop_low_playtime : bool = True,
                    low_playtime_cutoff : int = 25,
                    num_features: List[str] = STANDARD_NUM_FEATURES,
                    cat_features: List[str] = STANDARD_CAT_FEATURES, 
                    conv_activation: str = 'relu',
                    dense_activation: str = 'relu',
                    optimizer: str = 'adam',
                    learning_rate: float = 0.001,
                    loss: str = 'mse',
                    metrics: List[str] = ['mae'],
                    verbose: bool = False,
                    regularization: float = 0.001,
                    early_stopping: bool = False,
                    tolerance: float = 1e-5,
                    patience: int = 40, 
                    plot: bool = False, 
                    draw_model: bool = False, 
                    standardize: bool = True,
                    test_size: float = 0.15, 
                    val_size: float = 0.3,
                    stratify_by: str = 'skill' 
                    ):
    
    # Generate datasets

    (X_train, d_train, y_train, 
     X_val, d_val, y_val, 
     X_test, d_test, y_test, pipeline) = generate_datasets(data_dir=data_dir,
                                season=season,
                                position=position, 
                                window_size=window_size,
                                num_features=num_features,
                                cat_features=cat_features,
                                stratify_by=stratify_by,
                                test_size=test_size,
                                val_size=val_size,
                                drop_low_playtime=drop_low_playtime,
                                low_playtime_cutoff=low_playtime_cutoff,
                                verbose=verbose)
    
    #call build_train_cnn passing on all params 
    model, expt_res = build_train_cnn(
        X_train=X_train, d_train=d_train, y_train=y_train,
        X_val=X_val, d_val=d_val, y_val=y_val,
        X_test=X_test, d_test=d_test, y_test=y_test,
        season=season,
        position=position,
        window_size=window_size,
        kernel_size=kernel_size,
        num_filters=num_filters,
        num_dense=num_dense,
        batch_size=batch_size,
        epochs=epochs,
        drop_low_playtime=drop_low_playtime,
        low_playtime_cutoff=low_playtime_cutoff,
        num_features=num_features,
        cat_features=cat_features,
        conv_activation=conv_activation,
        dense_activation=dense_activation,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
        metrics=metrics,
        verbose=verbose,
        regularization=regularization,
        early_stopping=early_stopping,
        tolerance=tolerance,
        patience=patience,
        plot=plot,
        draw_model=draw_model,
        standardize=standardize
    )

    return model, expt_res
