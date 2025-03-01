import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx as onnx
import matplotlib.pyplot as plt
from torch.nn.functional import relu
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, List
from premier_league_models.processing.preprocess import generate_cnn_data, split_preprocess_cnn_data
from premier_league_models.processing.evaluate import plot_learning_curve, eval_cnn, log_evals, evaluate

from config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES


device = "cuda"  

class TemporalAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(input_dim, attention_dim, bias=False)  # Learnable weights
        self.v = nn.Linear(attention_dim, 1, bias=False)          # Attention scores

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, time_steps, input_dim)
        
        Returns:
            context: Tensor of shape (batch_size, input_dim)
                     Weighted sum of inputs based on attention scores.
            attention_weights: Tensor of shape (batch_size, time_steps)
                                Attention weights for each time step.
        """
        # Linear transformation and tanh activation
        attention_scores = torch.tanh(self.W(x))  # (batch_size, time_steps, attention_dim)

        # Compute attention weights
        attention_weights = self.v(attention_scores)  # (batch_size, time_steps, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize weights along time_steps

        # Weighted sum of inputs
        context = torch.sum(attention_weights * x, dim=1)  # (batch_size, input_dim)

        return context, attention_weights.squeeze(-1)  # Remove last dimension for weights

class rnnModel(nn.Module):
    def __init__(self, 
                 X_input_shape: Tuple, 
                 d_input_shape: Tuple, 
                 num_filters: int, 
                 num_dense: int, 
                 bidirectional: bool = True,
                 temporal_attention: bool = False,
                 conv_activation: str = 'relu', 
                 dense_activation: str = 'relu', 
                 regularization: float = 0.001):
        
        super(rnnModel, self).__init__()
        
        self.flatten = nn.Flatten()

        # Set up RNN
        self.rnn = nn.RNN(X_input_shape[1], num_filters, batch_first=True, bidirectional=bidirectional)

        # Set up Temporal Attention
        if bidirectional:
            self.attention = TemporalAttention(num_filters*2, num_filters)
        else:
            self.attention = TemporalAttention(num_filters, num_filters)
        # Output layer
        if temporal_attention:
            if bidirectional:
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(num_dense * 2 + d_input_shape[0], 1)
                )
            else:
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(num_dense + d_input_shape[0], 1)
                )
        elif bidirectional:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear((2*X_input_shape[0]*num_filters) + d_input_shape[0], num_dense*2),
                nn.ReLU(),
                nn.Linear(num_dense*2, num_dense),
                nn.ReLU(),
                nn.Linear(num_dense, 1)
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear((X_input_shape[0]*num_filters) + d_input_shape[0], num_dense),
                nn.ReLU(),
                nn.Linear(num_dense, 1)
            )

        # Store activations
        self.conv_activation = conv_activation
        self.dense_activation = dense_activation
        
        # Regularization (L1L2 applied in optimizer)
        self.regularization = regularization

        self.temporal_attention = temporal_attention

    def forward(self, x_input, d_input):
        # Pass through Conv1D layer
        #print(f"x_input size before rnn: {x_input.size()}")
        x, _ = self.rnn(x_input)  # Shape: (batch_size, num_filters, new_length)
        #print(f"x size before flatten: {x.size()}")

        if self.temporal_attention:
            x, _ = self.attention(x)

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

        # Combined dense/output relu stack
        x = self.linear_relu_stack(x).squeeze(-1)
        return x

def create_rnn(X_input_shape: Tuple, 
               d_input_shape: Tuple, 
               num_filters: int, 
               num_dense: int, 
               bidirectional: bool = True,
               temporal_attention: bool = False,
               conv_activation: str = 'relu', 
               dense_activation: str = 'relu', 
               optimizer: str = 'adam', 
               learning_rate: float = 0.001, 
               loss: str = 'mse', 
               verbose: bool = False, 
               regularization: float = 0.001):
    
    if verbose:
        print("====== Building rnn Architecture ======")

    # Initialize model
    model = rnnModel(X_input_shape, d_input_shape, num_filters, num_dense, bidirectional, 
                     temporal_attention, conv_activation, dense_activation, regularization).to(device)

    
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
        print("====== Done Building rnn Architecture ======")
    
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
    
    # =========== Generate rnn Dataset  ============
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

def build_train_rnn(X_train, d_train, y_train,
                    X_val, d_val, y_val,
                    X_test, d_test, y_test,
                    season: str,
                    position: str,
                    window_size: int,
                    num_filters: int,
                    num_dense: int,
                    bidirectional: bool,
                    temporal_attention: bool,
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
    model, optimizer, loss_fn = create_rnn(X_input_shape=X_input_shape, 
                                           d_input_shape=d_input_shape,
                                           num_filters=num_filters,
                                           num_dense=num_dense,
                                           bidirectional=bidirectional,
                                           temporal_attention=temporal_attention,
                                           conv_activation=conv_activation,
                                           dense_activation=dense_activation,
                                           optimizer=optimizer,
                                           learning_rate=learning_rate,
                                           loss=loss,
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
        if 'mae' in metrics and verbose:
            val_mae = mean_absolute_error(y_true, y_pred)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val MAE: {val_mae}")
        
        # Early stopping check
        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop and verbose: 
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
    if verbose:
        print(f'Test Loss (MSE): {test_mse}')
        print(f'Test Mean Absolute Error (MAE): {test_mae}')
    results = evaluate(model, loss_fn, device, X_train, d_train, y_train, X_val, d_val, y_val, X_test, d_test, y_test, y_test_pred)
    
    # Draw model if specified
    if plot:
        plot_learning_curve(history, season, position)

    # Export model for onnx
    x_input = torch.randn(1, *X_input_shape).to(device)
    d_input = torch.randn(1, *d_input_shape).to(device)
    print("X input shape:")
    print(X_input_shape)
    print("d input shape:")
    print(d_input_shape)
    # Export model for onnx
    torch.onnx.export(
        model,
        (x_input, d_input),  # Pass tuple of inputs
        position+"_model.onnx",  # Output ONNX filename
        export_params=True,  # Store trained parameters
        opset_version=11,  # ONNX opset version
        do_constant_folding=True,  # Optimize constants
        input_names=["player_data", "team_rating"],  # Multiple input names
        output_names=["score_prediction"],  # Name for output
        dynamic_axes={
            "player_data": {0: "batch_size"},  # Allow variable batch size
            "team_rating": {0: "batch_size"},
            "score_prediction": {0: "batch_size"},
        }
    )

    return model, results

def full_rnn_pipeline(data_dir: str, 
                    season: str, 
                    position: str,  
                    window_size: int,
                    num_filters: int,
                    num_dense: int,
                    bidirectional: bool,
                    temporal_attention: bool,
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
    
    #call build_train_rnn passing on all params 
    model, expt_res = build_train_rnn(
        X_train=X_train, d_train=d_train, y_train=y_train,
        X_val=X_val, d_val=d_val, y_val=y_val,
        X_test=X_test, d_test=d_test, y_test=y_test,
        season=season,
        position=position,
        window_size=window_size,
        num_filters=num_filters,
        num_dense=num_dense,
        bidirectional=bidirectional,
        temporal_attention=temporal_attention,
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
