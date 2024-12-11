
import torch
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

def evaluate(model, loss_fn, device, X_train, d_train, y_train, 
             X_val, d_val, y_val, 
             X_test, d_test, y_test, y_test_pred):
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    d_train_tensor = torch.tensor(d_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    d_val_tensor = torch.tensor(d_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    d_test_tensor = torch.tensor(d_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_output = model(X_train_tensor, d_train_tensor).detach()
    val_output = model(X_val_tensor, d_val_tensor).detach()
    test_output = model(X_test_tensor, d_test_tensor).detach()

    y_train_pred = train_output.cpu().numpy().flatten()
    y_val_pred = val_output.cpu().numpy().flatten()
    y_test_pred = test_output.cpu().numpy().flatten()

    res = {
        'train_mse': loss_fn(train_output, y_train_tensor).item(),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'val_mse': loss_fn(val_output, y_val_tensor).item(),
        'val_mae': mean_absolute_error(y_val, y_val_pred),
        'test_mse': loss_fn(test_output, y_test_tensor).item(),
        'test_mae':  mean_absolute_error(y_test, y_test_pred),
        'spear_corr': spearmanr(y_test, y_test_pred).correlation
        }
    
    return res



 