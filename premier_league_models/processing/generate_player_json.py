import os
import pandas as pd
from typing import Tuple, List, Union
from premier_league_models.processing.preprocess import get_avg_playtime
import json

from premier_league_models.cnn.config import STANDARD_CAT_FEATURES, STANDARD_NUM_FEATURES

def generate_player_json(data_dir : str, 
                        season : Union[str,List[str]],
                        position : str, 
                        window_size : int,
                        num_features: List[str] = STANDARD_NUM_FEATURES,
                        cat_features: List[str] = STANDARD_CAT_FEATURES,
                        drop_low_playtime : bool = True,
                        low_playtime_cutoff : int = 25,
                        verbose: bool = False) -> Tuple[pd.DataFrame]:
    """
    Load and shape cnn data for a specific season and position. 

    :param str data_dir: Path to the top-level directory containing player data.
    :param Union[str,List[str]] season: Season(s) of data to preprocess. 
        (Should match title(s) of desired season folder(s)). 
    :param str position: Position (GK, DEF, MID, FWD).
    :param int window_size: Size of the data window.
    :param bool drop_low_playtime: Whether or not to drop players that have low 
        average game minutes (threshold defined by `low_playtime_cutoff`).
    :param int low_playtime_cutoff: The cutoff (in avg. minutes) for which to drop 
        players for low playtime. Only used if `drop_low_playtime` set to True. 

    :return: Tuple of DataFrames
        (1) DataFrame containing preprocessed data with columns as follows:
            'name' - name of the player
            'avg_score' - avg score of the player, used to stratify skill for 
                train/val/test split later
            'stdev_score' - standard deviation socre of the player, used to stratify skill
            for the train/val/test split later
            'features' - DF containing window_size of data starting at some gw
            'target' - prediction value for the week following the 'features' window
            'matchup_difficulty' - additional feature treated differently in the
                CNN architecture since it is known for the upcoming week ahead of
                time
        (2) DataFrame containing all player-weeks features combined for the full_data
    :rtype: Tuple(pd.DataFrame)
    """

    ct_players = 0
    ct_dropped_players = 0

    if verbose:
        print(f"======= Generating JSON Data for Season: {season}, Position: {position} =======")
        if drop_low_playtime:
            print(f"Dropping Players with Avg. Playtime < {low_playtime_cutoff}...\n")

    if type(season) == list:
        seasons = season
    else:
        seasons = [season]

    # Player dictionary, to be exported for use on webpage
    player_dict = {}

    # ================ Iterate through Clean Data and generate CNN data =================
    for season in seasons:
        position_folder = os.path.join(data_dir, season, position)
        for player_folder in os.listdir(position_folder):
            ct_players += 1
            player_path = os.path.join(position_folder, player_folder)
            if os.path.isdir(player_path):
                player_csv = os.path.join(player_path, 'gw.csv')
                if os.path.isfile(player_csv):
                    player_data = pd.read_csv(player_csv)
                    # drop players with low avg playtime if requested
                    if drop_low_playtime and get_avg_playtime(player_data) < low_playtime_cutoff:
                        ct_dropped_players += 1
                        continue

                    features = player_data.copy()
                    #name used only for subsetting to training data when running pipeline transform
                    cols_to_keep = num_features + cat_features + ['name'] 
                    features = features.loc[:, cols_to_keep] 
                    targets = player_data['total_points'].values
                    # matchup difficulty is treated separately, and has to be extracted separately
                    try:
                        difficulties = player_data['matchup_difficulty'].values
                    except: 
                        print(f"{player_data['name']} missing matchup difficulty info. Check data cleaning scripts.")
                        raise Exception

                    # Create training samples using the specified window size
                    X, y, d, player_names, avg_scores, stdev_scores, seasons = [], [], [], [], [], [], []
                    player_name = player_data['name'][0]
                    avg_score = player_data['total_points'].mean()
                    stdev_score = player_data['total_points'].std()
                    for i in range(len(player_data) - window_size):
                        X.append(features.iloc[i:i + window_size]) # get window of features
                        y.append(targets[i + window_size]) # get next weeks FPL points
                        d.append(difficulties[i + window_size]) # get next weeks matchup difficulty
                        player_names.append(player_name)
                        avg_scores.append(avg_score)
                        stdev_scores.append(stdev_score)
                        seasons.append(season)

                    if len(d) >= 6:
                        player_dict[player_name] = {"player_team": player_data.sample(n=1)["team"].values[0], "player_data": [row[:-1] for row in X[-1].values.tolist()], "team_rating": list(d[-6:]), "actual_score": int(y[-1])}


    with open('./json/' + season + '_' + position + '.json', 'w') as file:
        json.dump(player_dict, file, indent=2)

    if verbose:
        print(f"Total players of type {position} = {ct_players}.")
        print(f"{ct_dropped_players} players dropped due to low average playtime.")
        print(f"========== Done Generating JSON Data ==========\n")
