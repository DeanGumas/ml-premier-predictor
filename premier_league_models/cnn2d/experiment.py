"""
Run end-to-end experiments on the FPL Regression CNN.

Wrapper for preprocessing, model construction, model training, model evaluation,
and results visualization.
"""

import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..','..'))
import tensorflow as tf
import numpy as np
import pandas as pd
from premier_league_models.cnn2d.model import build_train_cnn, generate_datasets
import tensorflow as tf
import itertools
from datetime import date
from tqdm import tqdm
import torch
import pickle

from config import STANDARD_CAT_FEATURES, NUM_FEATURES_DICT

def gridsearch_cnn(experiment_name: str = 'gridsearch',
                   verbose : bool = False):
    """
    GridSearch for Best Hyperparameters
    """
    log_file = os.path.join('gridsearch', 'results', f'{experiment_name}_{date.today().strftime("%m-%d")}.csv')
    data_log_file = os.path.join('gridsearch', 'results', f'{experiment_name}_{date.today().strftime("%m-%d")}_data.pkl')
    DATA_DIR = os.path.join(os.getcwd(), '..', 'data', 'clean_data')

    if verbose:
        print("======= Running GridSearch Experiment ========")

    EPOCHS = 750
    SEED = 229
    BATCH_SIZE = 32

    tf.random.set_seed(SEED)
    np.random.seed(SEED)


    # ___Fixed Parameters________________________
    SEASON = ['2020-21', '2021-22']
    TOLERANCE = 1e-4 #early stopping tolderance 
    PATIENCE = 20  #num of iterations of no minimization of val loss
    STANDARDIZE = True
    CONV_ACTIVATION = 'relu'
    DENSE_ACTIVATION = 'relu'
    DROP_LOW_PLAYTIME = True
    LOW_PLAYTIME_CUTOFF = 1e-6
    OPTIMIZER = 'adam'
    REGULARIZATION = 0.01
    LEARNING_RATE = 0.00001
    KERNEL_SIZE = 2
    BIDIRECTIONAL = True
    TEMPORAL_ATTENTION= True


    #____Variable Ranges________________________

    #POSITIONS = ['GK', 'DEF', 'MID', 'FWD']
    #WINDOW_SIZES = [3, 6, 9]
    #NUM_DENSE = [64, 128, 256]  # drop players who never play
    #AMT_NUM_FEATURES = ['pts_ict', 'medium', 'large'] 
    #CAT_FEATURES = STANDARD_CAT_FEATURES
    #STRATIFY_BY = ['skill', 'stdev']
    #TOLERANCE_VALUES = [1e-4, 1e-6]

    POSITIONS = ['GK']
    WINDOW_SIZES = [6]
    NUM_DENSE = [64]  # drop players who never play
    AMT_NUM_FEATURES = ['large'] 
    CAT_FEATURES = STANDARD_CAT_FEATURES
    STRATIFY_BY = ['stdev']
    TOLERANCE_VALUES = [1e-4]
    SEEDS = [400, 401, 402, 403, 404, 405]

    # Loop through all combinations of parameters
    experiment_result = []

    total_iterations = (
        len(POSITIONS) * #Y
        len(WINDOW_SIZES) * #Y
        len(NUM_DENSE) * #Y
        len(AMT_NUM_FEATURES) * #Y
        len(TOLERANCE_VALUES) *
        len(STRATIFY_BY) *
        len(SEEDS))
    
    iteration_index = 0
    print(f"===== Total Number of Iterations: ", total_iterations)

    for (position,
         window_size, 
         num_dense,
         tolerance,
         amt_num_feature, 
         stratify_by,
         seed) in tqdm(itertools.product(
            POSITIONS, 
            WINDOW_SIZES, 
            NUM_DENSE,
            TOLERANCE_VALUES,
            AMT_NUM_FEATURES,
            STRATIFY_BY,
            SEEDS), total=total_iterations):
        

        variable_parameters = {
            'seed': seed,
            'position': position,
            'window_size': window_size,
            'num_dense': num_dense,
            "tolerance": tolerance,
            'amt_num_features': amt_num_feature,
            "stratify_by": stratify_by,
        }
        num_features = NUM_FEATURES_DICT[position][amt_num_feature]


        torch.manual_seed(seed)

        print(f"===== Running Experiment for Parameters: =====\n {variable_parameters}\n")
        for [name, val] in variable_parameters.items():
            print(name, val)

        print("Running Iteration: ", iteration_index)
        iteration_index += 1

        # Run the experiment for the current set of hyperparameters
        (X_train, d_train, y_train, 
        X_val, d_val, y_val, 
        X_test, d_test, y_test, pipeline) = generate_datasets(data_dir=DATA_DIR,
                                            season=SEASON,
                                            position=position, 
                                            window_size=window_size,
                                            num_features=num_features,
                                            cat_features=CAT_FEATURES,
                                            stratify_by=stratify_by,
                                            drop_low_playtime=DROP_LOW_PLAYTIME,
                                            low_playtime_cutoff=LOW_PLAYTIME_CUTOFF,
                                            verbose=verbose)
    
        #call build_train_cnn passing on all params 
        model, iteration_result = build_train_cnn(
                X_train=X_train, d_train=d_train, y_train=y_train,
                X_val=X_val, d_val=d_val, y_val=y_val,
                X_test=X_test, d_test=d_test, y_test=y_test,
                season=SEASON,
                position=position,
                kernel_size=KERNEL_SIZE,
                window_size=window_size,
                num_filters=num_dense*2,
                num_dense=num_dense,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                drop_low_playtime=DROP_LOW_PLAYTIME,
                low_playtime_cutoff=LOW_PLAYTIME_CUTOFF,
                num_features=num_features,
                cat_features=CAT_FEATURES,
                conv_activation=CONV_ACTIVATION,
                dense_activation=DENSE_ACTIVATION,
                optimizer=OPTIMIZER,
                learning_rate=LEARNING_RATE,
                loss='mse',
                metrics=['mae'],
                verbose=verbose,
                regularization=REGULARIZATION,
                early_stopping=True,
                tolerance=tolerance,
                patience=PATIENCE,
                plot=False,
                draw_model=False,
                standardize=STANDARDIZE
            )
        
        iteration_result.update(variable_parameters)
        experiment_result.append(iteration_result)

    if verbose:
        print(f"Updating GridSearch Results Log File: {log_file}...")

    log_experiment(experiment_result, log_file)
    if verbose:
        print("======= Done with GridSearch Experiment ========")

    return 


def log_experiment(experiment_result, log_file):
    """
    Log the results of several evaluations to a csv file specified by log_file.

    Creates the log_file if it doesn't exists, appends results otherwise.
    """
    COLUMNS = list(experiment_result[0].keys())

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a pandas DataFrame if the log_file exists, otherwise create a new one
    log_df = None
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
    else:
        log_df = pd.DataFrame(columns=COLUMNS)

    experiment_df = pd.DataFrame(experiment_result, columns=COLUMNS)
    log_df = pd.concat([log_df, experiment_df], ignore_index=True)

    
    print(f"Logging experiment results to {log_file}.")

    log_df.to_csv(log_file, index=False)

    return

def main():
    """
    Run the experiment specified by gridsearch_cnn constants
    """
    gridsearch_cnn(verbose=False)
    return

if __name__ == '__main__':
    main()
