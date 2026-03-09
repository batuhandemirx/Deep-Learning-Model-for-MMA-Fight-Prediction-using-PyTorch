# UFC Fight Outcome Prediction – Deep Learning Project

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the file is in the current directory or mounted to Google 
Drive
# If not, update the file path accordingly
file_path = 'ufc-masterr.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify the data
df.head()

{"type":"dataframe","variable_name":"df"}

# We agree to drop these columns.
columns_to_drop = ['Location', 'Country', 'FinishDetails', 
'FinishRoundTime']
df.drop(columns=columns_to_drop, inplace=True)

missing_finish_count = df['Finish'].isnull().sum()
missing_finish_count

238

# Assuming your DataFrame is named 'df'
unique_finishes = df['Finish'].unique()

# Print the unique values
print(unique_finishes)

# Calculate the mode of the 'Finish' column
mode_finish = df['Finish'].mode()[0]
# Impute missing values with the mode
df['Finish'].fillna(mode_finish, inplace=True)

['KO/TKO' 'S-DEC' 'U-DEC' 'SUB' 'M-DEC' 'DQ' nan 'Overturned']

<ipython-input-4-e7e5aa4639b4>:10: FutureWarning: A value is trying to 
be set on a copy of a DataFrame or Series through chained assignment 
using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never 
work because the intermediate object on which we are setting values 

always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try 
using 'df.method({col: value}, inplace=True)' or df[col] = 
df[col].method(value) instead, to perform the operation inplace on the 
original object.

  df['Finish'].fillna(mode_finish, inplace=True)

# Create a boolean mask for missing 'FinishRound' values
missing_finish_round = df['FinishRound'].isnull()

# Create a boolean mask for missing 'TotalFightTimeSecs' values
missing_total_fight_time_secs = df['TotalFightTimeSecs'].isnull()

# Compare the masks
are_same = (missing_finish_round == 
missing_total_fight_time_secs).all()

are_same

True

# 1. Impute 'FinishRound' with the mode (most common value)
mode_finish_round = df['FinishRound'].mode()[0]
df['FinishRound'].fillna(mode_finish_round, inplace=True)

# Ensure 'FinishRound' is integer
df['FinishRound'] = df['FinishRound'].astype(int)

# 2. Define a function to impute 'TotalFightTimeSecs' based on 
'FinishRound'
def impute_total_fight_time(row):
    if pd.isna(row['TotalFightTimeSecs']):
        finish_round = row['FinishRound']
        # Each round is 5 minutes = 300 seconds
        # Minimum fight time: (FinishRound - 1)*300 + 60 (at least 1 
minute into the final round)
        # Maximum fight time: FinishRound * 300 - 1 second
        min_time = (finish_round - 1) * 300 + 60
        max_time = finish_round * 300
        return np.random.randint(min_time, max_time)
    else:
        return row['TotalFightTimeSecs']

# 3. Apply the function to impute missing 'TotalFightTimeSecs'
df['TotalFightTimeSecs'] = df.apply(impute_total_fight_time, axis=1)

# 4. Verify that there are no remaining missing values
missing_after = df[['FinishRound', 

'TotalFightTimeSecs']].isnull().sum()
print("Missing values after imputation:")
print(missing_after)

<ipython-input-6-505893e2acda>:3: FutureWarning: A value is trying to 
be set on a copy of a DataFrame or Series through chained assignment 
using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never 
work because the intermediate object on which we are setting values 
always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try 
using 'df.method({col: value}, inplace=True)' or df[col] = 
df[col].method(value) instead, to perform the operation inplace on the 
original object.

  df['FinishRound'].fillna(mode_finish_round, inplace=True)

Missing values after imputation:
FinishRound           0
TotalFightTimeSecs    0
dtype: int64

missing_after = df.isnull().sum()
print("Missing Values After Imputation:\n", 
missing_after[missing_after > 0])

Missing Values After Imputation:
 RedOdds                   227
BlueOdds                  226
RedExpectedValue          227
BlueExpectedValue         226
BlueAvgSigStrLanded       930
BlueAvgSigStrPct          765
BlueAvgSubAtt             832
BlueAvgTDLanded           833
BlueAvgTDPct              842
BlueStance                  3
RedAvgSigStrLanded        455
RedAvgSigStrPct           357
RedAvgSubAtt              357
RedAvgTDLanded            357
RedAvgTDPct               367
EmptyArena               1499
BMatchWCRank             5339
RMatchWCRank             4760
RWFlyweightRank          6445
RWFeatherweightRank      6532
RWStrawweightRank        6395

RWBantamweightRank       6387
RHeavyweightRank         6355
RLightHeavyweightRank    6357
RMiddleweightRank        6359
RWelterweightRank        6349
RLightweightRank         6357
RFeatherweightRank       6364
RBantamweightRank        6360
RFlyweightRank           6352
RPFPRank                 6288
BWFlyweightRank          6468
BWFeatherweightRank      6540
BWStrawweightRank        6441
BWBantamweightRank       6434
BHeavyweightRank         6393
BLightHeavyweightRank    6421
BMiddleweightRank        6404
BWelterweightRank        6421
BLightweightRank         6421
BFeatherweightRank       6417
BBantamweightRank        6422
BFlyweightRank           6410
BPFPRank                 6474
RedDecOdds               1087
BlueDecOdds              1117
RSubOdds                 1336
BSubOdds                 1360
RKOOdds                  1334
BKOOdds                  1361
dtype: int64

# =============================
# STEP 3.4: CREATE A SINGLE RANK CATEGORY PER FIGHTER CORNER
# =============================

# 3.4.1 Identify Red rank columns and Blue rank columns
# 3.4.1 Identify Red rank columns and Blue rank columns
red_rank_cols = [c for c in df.columns if c.startswith('R') and 
c.endswith('Rank') and 'weight' in c.lower() and c != 'BetterRank']
blue_rank_cols = [c for c in df.columns if c.startswith('B') and 
c.endswith('Rank') and 'weight' in c.lower() and  c != 'BetterRank']

def get_rank_label(row, rank_cols):
    """
    Given a row and a list of rank columns (e.g. for Red or Blue),
    return a label: high level, good level, or okay fighter.
    """
    # Gather all rank values
    ranks = row[rank_cols].dropna()
    if len(ranks) == 0:

        # No rank at all => okay fighter
        return "okay fighter"

    min_rank = ranks.min()
    if min_rank <= 5:
        return "high level"
    elif min_rank <= 15:
        return "good level"
    else:
        return "okay fighter"

# Create new columns:
df['RedRankCategory'] = df.apply(lambda x: get_rank_label(x, 
red_rank_cols), axis=1)
df['BlueRankCategory'] = df.apply(lambda x: get_rank_label(x, 
blue_rank_cols), axis=1)

# Now drop the old rank columns
df.drop(columns=red_rank_cols + blue_rank_cols, inplace=True)

print("Shape after consolidating rank columns:", df.shape)
df[['RedRankCategory','BlueRankCategory']].head(10)

Shape after consolidating rank columns: (6541, 92)

{"summary":"{\n  \"name\": 
\"df[['RedRankCategory','BlueRankCategory']]\",\n  \"rows\": 10,\n  
\"fields\": [\n    {\n      \"column\": \"RedRankCategory\",\n      
\"properties\": {\n        \"dtype\": \"category\",\n        
\"num_unique_values\": 2,\n        \"samples\": [\n          \"okay 
fighter\",\n          \"good level\"\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"BlueRankCategory\",\n      
\"properties\": {\n        \"dtype\": \"category\",\n        
\"num_unique_values\": 2,\n        \"samples\": [\n          \"okay 
fighter\",\n          \"good level\"\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    }\n  ]\n}","type":"dataframe"}

# Filter rows where either 'BMatchWCRank' or 'RMatchWCRank' is not 
null
filtered_df = df[df['BMatchWCRank'].notnull() | 
df['RMatchWCRank'].notnull()]

# Select only fighter-related columns and 'BMatchWCRank', 
'RMatchWCRank'
fighter_cols = [col for col in df.columns if col.startswith(('R', 
'B'))]
selected_cols = fighter_cols + ['BMatchWCRank', 'RMatchWCRank']
filtered_df = filtered_df[selected_cols]

# Display the filtered DataFrame
filtered_df.head()

{"type":"dataframe","variable_name":"filtered_df"}

# 1. Drop the EmptyArena column
if 'EmptyArena' in df.columns:
    df.drop(columns=['EmptyArena'], inplace=True)
    print("Dropped 'EmptyArena' column.")

# 2. Impute BlueStance with its mode (most frequent value)
if 'BlueStance' in df.columns:
    mode_blue_stance = df['BlueStance'].mode()[0]
    df['BlueStance'] = df['BlueStance'].fillna(mode_blue_stance)
    print("Imputed missing BlueStance with mode:", mode_blue_stance)

# 3. Impute 'Avg' columns and 'Odds' columns per fighter using group 
means.
#    Define the lists of columns for red and blue fighters.

# Average performance columns for red and blue fighters:
red_avg_cols = ['RedAvgSigStrLanded', 'RedAvgSigStrPct', 
'RedAvgSubAtt', 'RedAvgTDLanded', 'RedAvgTDPct']
blue_avg_cols = ['BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 
'BlueAvgSubAtt', 'BlueAvgTDLanded', 'BlueAvgTDPct']

# Odds columns for red and blue fighters:
red_odds_cols = ['RedOdds', 'RedExpectedValue', 'RedDecOdds', 
'RSubOdds', 'RKOOdds']
blue_odds_cols = ['BlueOdds', 'BlueExpectedValue', 'BlueDecOdds', 
'BSubOdds', 'BKOOdds']

# Function to impute missing values per fighter group using the mean 
for that fighter.
def impute_by_fighter(df, fighter_col, cols_to_impute):
    for col in cols_to_impute:
        if col in df.columns:
            # Group by the fighter (e.g. RedFighter or BlueFighter) 
and fill missing with that fighter's mean.
            df[col] = df.groupby(fighter_col)[col].transform(lambda x: 
x.fillna(x.mean()))
    return df

# Impute for red fighter columns:
df = impute_by_fighter(df, 'RedFighter', red_avg_cols + red_odds_cols)

# Impute for blue fighter columns:
df = impute_by_fighter(df, 'BlueFighter', blue_avg_cols + 
blue_odds_cols)

# Optional: Print the number of missing values in these columns to 
check the imputation
missing_after = df[red_avg_cols + blue_avg_cols + red_odds_cols + 
blue_odds_cols].isnull().sum()
print("Missing values after imputation:")
print(missing_after)

Dropped 'EmptyArena' column.
Imputed missing BlueStance with mode: Orthodox
Missing values after imputation:
RedAvgSigStrLanded      95
RedAvgSigStrPct         87
RedAvgSubAtt            87
RedAvgTDLanded          87
RedAvgTDPct             87
BlueAvgSigStrLanded    149
BlueAvgSigStrPct       129
BlueAvgSubAtt          138
BlueAvgTDLanded        138
BlueAvgTDPct           139
RedOdds                 37
RedExpectedValue        37
RedDecOdds             333
RSubOdds               421
RKOOdds                420
BlueOdds                36
BlueExpectedValue       36
BlueDecOdds            384
BSubOdds               498
BKOOdds                498
dtype: int64

# List all columns we want to impute overall (after group-based 
imputation)
cols_to_impute = red_avg_cols + blue_avg_cols + red_odds_cols + 
blue_odds_cols

# For each column, fill missing values with the overall mean of that 
column
for col in cols_to_impute:
    overall_mean = df[col].mean()
    df[col] = df[col].fillna(overall_mean)
    print(f"Imputed remaining missing values in {col} with overall 
mean: {overall_mean:.2f}")

# Verify that missing values are handled:
missing_after_overall = df[cols_to_impute].isnull().sum()
print("Missing values after overall imputation:")
print(missing_after_overall)

Imputed remaining missing values in RedAvgSigStrLanded with overall 
mean: 21.27
Imputed remaining missing values in RedAvgSigStrPct with overall mean: 
0.46
Imputed remaining missing values in RedAvgSubAtt with overall mean: 
0.54
Imputed remaining missing values in RedAvgTDLanded with overall mean: 
1.40
Imputed remaining missing values in RedAvgTDPct with overall mean: 
0.34
Imputed remaining missing values in BlueAvgSigStrLanded with overall 
mean: 20.30
Imputed remaining missing values in BlueAvgSigStrPct with overall 
mean: 0.45
Imputed remaining missing values in BlueAvgSubAtt with overall mean: 
0.49
Imputed remaining missing values in BlueAvgTDLanded with overall mean: 
1.30
Imputed remaining missing values in BlueAvgTDPct with overall mean: 
0.32
Imputed remaining missing values in RedOdds with overall mean: -117.34
Imputed remaining missing values in RedExpectedValue with overall 
mean: 96.30
Imputed remaining missing values in RedDecOdds with overall mean: 
309.09
Imputed remaining missing values in RSubOdds with overall mean: 882.32
Imputed remaining missing values in RKOOdds with overall mean: 518.98
Imputed remaining missing values in BlueOdds with overall mean: 58.97
Imputed remaining missing values in BlueExpectedValue with overall 
mean: 164.67
Imputed remaining missing values in BlueDecOdds with overall mean: 
429.85
Imputed remaining missing values in BSubOdds with overall mean: 
1105.11
Imputed remaining missing values in BKOOdds with overall mean: 644.79
Missing values after overall imputation:
RedAvgSigStrLanded     0
RedAvgSigStrPct        0
RedAvgSubAtt           0
RedAvgTDLanded         0
RedAvgTDPct            0
BlueAvgSigStrLanded    0
BlueAvgSigStrPct       0
BlueAvgSubAtt          0
BlueAvgTDLanded        0
BlueAvgTDPct           0
RedOdds                0
RedExpectedValue       0
RedDecOdds             0
RSubOdds               0

RKOOdds                0
BlueOdds               0
BlueExpectedValue      0
BlueDecOdds            0
BSubOdds               0
BKOOdds                0
dtype: int64

missing_after = df.isnull().sum()
print("Missing Values After Imputation:\n", 
missing_after[missing_after > 0])

Missing Values After Imputation:
 BMatchWCRank    5339
RMatchWCRank    4760
RPFPRank        6288
BPFPRank        6474
dtype: int64

# List of ranking columns to handle
ranking_cols = ['BMatchWCRank', 'RMatchWCRank', 'RPFPRank', 
'BPFPRank']

for col in ranking_cols:
    # Compute the maximum observed value (ignoring missing values)
    max_rank = df[col].max(skipna=True)
    # Impute missing values with a constant worse than any observed 
rank.
    # Here, we use max observed + 1. Alternatively, you can use a 
fixed value like 999 if appropriate.
    impute_value = max_rank + 1
    df[col] = df[col].fillna(impute_value)
    print(f"For column {col}, imputed missing values with 
{impute_value}")

# Verify that there are no missing values in these columns now
print("Missing values after ranking imputation:")
print(df[ranking_cols].isnull().sum())

For column BMatchWCRank, imputed missing values with 16.0
For column RMatchWCRank, imputed missing values with 16.0
For column RPFPRank, imputed missing values with 16.0
For column BPFPRank, imputed missing values with 16.0
Missing values after ranking imputation:
BMatchWCRank    0
RMatchWCRank    0
RPFPRank        0
BPFPRank        0
dtype: int64

# Filter the DataFrame for rows where either RedFighter or BlueFighter 
is 'Conor McGregor'
mcgregor_rows = df[(df['RedFighter'] == 'Conor McGregor') | 
(df['BlueFighter'] == 'Conor McGregor')]

# Select the desired columns
selected_columns = ['RedFighter', 'BlueFighter', 
'BlueTotalRoundsFought', 'RedTotalRoundsFought', 
'BlueAvgSigStrLanded', 'RedAvgSigStrLanded']
mcgregor_data = mcgregor_rows[selected_columns]

# Display the last three rows using tail()
last_three_rows = mcgregor_data.tail(5)
print(last_three_rows)

          RedFighter     BlueFighter  BlueTotalRoundsFought  \
4811  Conor McGregor    Dennis Siver                     42   
4957  Dustin Poirier  Conor McGregor                      5   
5048  Conor McGregor   Diego Brandao                     12   
5490  Conor McGregor    Max Holloway                     12   
5622  Marcus Brimage  Conor McGregor                      0   

      RedTotalRoundsFought  BlueAvgSigStrLanded  RedAvgSigStrLanded  
4811                     6             39.55560             25.2500  
4957                    23             30.66670             47.4000  
5048                     4             23.66670             37.0000  
5490                     1             70.60000             21.0000  
5622                     9             30.06355             74.3333  

df.columns.to_list()

['RedFighter',
 'BlueFighter',
 'RedOdds',
 'BlueOdds',
 'RedExpectedValue',
 'BlueExpectedValue',
 'Date',
 'Winner',
 'TitleBout',
 'WeightClass',
 'Gender',
 'NumberOfRounds',
 'BlueCurrentLoseStreak',
 'BlueCurrentWinStreak',
 'BlueDraws',
 'BlueAvgSigStrLanded',
 'BlueAvgSigStrPct',
 'BlueAvgSubAtt',
 'BlueAvgTDLanded',

 'BlueAvgTDPct',
 'BlueLongestWinStreak',
 'BlueLosses',
 'BlueTotalRoundsFought',
 'BlueTotalTitleBouts',
 'BlueWinsByDecisionMajority',
 'BlueWinsByDecisionSplit',
 'BlueWinsByDecisionUnanimous',
 'BlueWinsByKO',
 'BlueWinsBySubmission',
 'BlueWinsByTKODoctorStoppage',
 'BlueWins',
 'BlueStance',
 'BlueHeightCms',
 'BlueReachCms',
 'BlueWeightLbs',
 'RedCurrentLoseStreak',
 'RedCurrentWinStreak',
 'RedDraws',
 'RedAvgSigStrLanded',
 'RedAvgSigStrPct',
 'RedAvgSubAtt',
 'RedAvgTDLanded',
 'RedAvgTDPct',
 'RedLongestWinStreak',
 'RedLosses',
 'RedTotalRoundsFought',
 'RedTotalTitleBouts',
 'RedWinsByDecisionMajority',
 'RedWinsByDecisionSplit',
 'RedWinsByDecisionUnanimous',
 'RedWinsByKO',
 'RedWinsBySubmission',
 'RedWinsByTKODoctorStoppage',
 'RedWins',
 'RedStance',
 'RedHeightCms',
 'RedReachCms',
 'RedWeightLbs',
 'RedAge',
 'BlueAge',
 'LoseStreakDif',
 'WinStreakDif',
 'LongestWinStreakDif',
 'WinDif',
 'LossDif',
 'TotalRoundDif',
 'TotalTitleBoutDif',
 'KODif',

 'SubDif',
 'HeightDif',
 'ReachDif',
 'AgeDif',
 'SigStrDif',
 'AvgSubAttDif',
 'AvgTDDif',
 'BMatchWCRank',
 'RMatchWCRank',
 'RPFPRank',
 'BPFPRank',
 'BetterRank',
 'Finish',
 'FinishRound',
 'TotalFightTimeSecs',
 'RedDecOdds',
 'BlueDecOdds',
 'RSubOdds',
 'BSubOdds',
 'RKOOdds',
 'BKOOdds',
 'RedRankCategory',
 'BlueRankCategory']

Assigning ID's To Fighters

# 1) Gather all unique names from both Red and Blue columns.
all_fighters = pd.concat([df["RedFighter"], 
df["BlueFighter"]]).unique()

print(f"Total unique fighter names: {len(all_fighters)}")

Total unique fighter names: 2113

fighter_to_id = {name: idx for idx, name in enumerate(all_fighters)}

# Example:
# {
#   'Conor McGregor': 0,
#   'Khabib Nurmagomedov': 1,
#   ...
# }

df["RedFighterID"] = df["RedFighter"].map(fighter_to_id)
df["BlueFighterID"] = df["BlueFighter"].map(fighter_to_id)

df.head(10)

{"type":"dataframe","variable_name":"df"}

df['Finish'].value_counts()

Finish
U-DEC         2647
KO/TKO        2016
SUB           1157
S-DEC          655
M-DEC           46
DQ              18
Overturned       2
Name: count, dtype: int64

Aggregations

df = df.sort_values(by='Date').reset_index(drop=True)
df['FightID'] = df.index  # or some other unique identifier

df.head()

{"type":"dataframe","variable_name":"df"}

import pandas as pd

# ---------------------
# Red corner subset
# ---------------------
red_df = df[[
    'FightID', 'Date', 'RedFighterID',
    'RedAvgSigStrLanded', 'RedAvgSigStrPct',
    'RedAvgSubAtt', 'RedAvgTDLanded', 'RedAvgTDPct',
    'FinishRound'  # if you want to do cumulative finish round as well
]].copy()

red_df.rename(columns={
    'RedFighterID': 'FighterID',
    'RedAvgSigStrLanded': 'AvgSigStrLanded',
    'RedAvgSigStrPct':    'AvgSigStrPct',
    'RedAvgSubAtt':       'AvgSubAtt',
    'RedAvgTDLanded':     'AvgTDLanded',
    'RedAvgTDPct':        'AvgTDPct'
}, inplace=True)

# ---------------------
# Blue corner subset
# ---------------------
blue_df = df[[
    'FightID', 'Date', 'BlueFighterID',
    'BlueAvgSigStrLanded', 'BlueAvgSigStrPct',
    'BlueAvgSubAtt', 'BlueAvgTDLanded', 'BlueAvgTDPct',
    'FinishRound'

]].copy()

blue_df.rename(columns={
    'BlueFighterID':       'FighterID',
    'BlueAvgSigStrLanded': 'AvgSigStrLanded',
    'BlueAvgSigStrPct':    'AvgSigStrPct',
    'BlueAvgSubAtt':       'AvgSubAtt',
    'BlueAvgTDLanded':     'AvgTDLanded',
    'BlueAvgTDPct':        'AvgTDPct'
}, inplace=True)

# ---------------------
# Concatenate (long_df)
# ---------------------
fighter_long = pd.concat([red_df, blue_df], axis=0, ignore_index=True)

# Sort again by Date + FighterID (just to keep a consistent order)
#fighter_long.sort_values(by=['FighterID','Date'], inplace=True)

fighter_long.head(10)

{"summary":"{\n  \"name\": \"fighter_long\",\n  \"rows\": 13082,\n  
\"fields\": [\n    {\n      \"column\": \"FightID\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
1888,\n        \"min\": 0,\n        \"max\": 6540,\n        
\"num_unique_values\": 6541,\n        \"samples\": [\n          2877,\
n          233,\n          2131\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"Date\",\n      \"properties\": {\n 
\"dtype\": \"object\",\n        \"num_unique_values\": 565,\n        
\"samples\": [\n          \"2024-05-11\",\n          \"2024-01-13\",\n 
\"2017-01-15\"\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"FighterID\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 530,\n        \"min\": 0,\n        
\"max\": 2112,\n        \"num_unique_values\": 2113,\n        
\"samples\": [\n          1229,\n          1759,\n          680\n      
],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n  
}\n    },\n    {\n      \"column\": \"AvgSigStrLanded\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
19.728279390590792,\n        \"min\": 0.0,\n        \"max\": 154.0,\n  
\"num_unique_values\": 3338,\n        \"samples\": [\n          6.6,\n 
16.2375,\n          52.8\n        ],\n        \"semantic_type\": 
\"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      
\"column\": \"AvgSigStrPct\",\n      \"properties\": {\n        
\"dtype\": \"number\",\n        \"std\": 0.104635925391437,\n        
\"min\": 0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 
1000,\n        \"samples\": [\n          0.48675,\n          
0.33224999999999993,\n          0.4363333333333333\n        ],\n       
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\

n    },\n    {\n      \"column\": \"AvgSubAtt\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
0.6774831906040781,\n        \"min\": 0.0,\n        \"max\": 8.4,\n    
\"num_unique_values\": 708,\n        \"samples\": [\n          
1.5333,\n          0.766675,\n          0.5555666666666667\
n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"AvgTDLanded\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 1.3154307190029486,\n        \"min\": 
0.0,\n        \"max\": 12.5,\n        \"num_unique_values\": 1419,\n   
\"samples\": [\n          0.6,\n          1.0047333333333333,\n        
12.5\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"AvgTDPct\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 0.22867403997345281,\n        \"min\": 
0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 1155,\n    
\"samples\": [\n          0.026727272727272725,\n          0.56,\n     
0.26566666666666666\n        ],\n        \"semantic_type\": \"\",\n    
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"FinishRound\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 0,\n        \"min\": 1,\n        
\"max\": 5,\n        \"num_unique_values\": 5,\n        \"samples\": 
[\n          1,\n          4,\n          2\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    }\n  ]\n}","type":"dataframe","variable_name":"fighter_long"}

# We'll define a helper function that uses expanding().mean().shift(1)
# to create a "HistoricalAvgSigStrLanded" column for each fighter.

def create_historical_column(df, col_name):
    """
    For the given col_name (e.g. 'AvgSigStrLanded'),
    compute the cumulative average for each fighter,
    shifted by 1 so the current row's fight is excluded.
    """
    # Group by fighter, then apply expanding mean, then shift(1)
    df[f'Historical_{col_name}'] = (
        df.groupby('FighterID')[col_name]
          .apply(lambda x: x.expanding().mean().shift(1))
          .reset_index(level=0, drop=True)  # Reset index to align 
with original DataFrame
    )
    return df

# Let's do it for each stat you care about:
stats_to_lag = [
    'AvgSigStrLanded',
    'AvgSigStrPct',
    'AvgSubAtt',
    'AvgTDLanded',

    'AvgTDPct',
    'FinishRound'  # if you want to get average finish round from past 
fights
]

for stat_col in stats_to_lag:
    fighter_long = create_historical_column(fighter_long, stat_col)

fighter_long.head(15)

{"summary":"{\n  \"name\": \"fighter_long\",\n  \"rows\": 13082,\n  
\"fields\": [\n    {\n      \"column\": \"FightID\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
1888,\n        \"min\": 0,\n        \"max\": 6540,\n        
\"num_unique_values\": 6541,\n        \"samples\": [\n          2877,\
n          233,\n          2131\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"Date\",\n      \"properties\": {\n 
\"dtype\": \"object\",\n        \"num_unique_values\": 565,\n        
\"samples\": [\n          \"2024-05-11\",\n          \"2024-01-13\",\n 
\"2017-01-15\"\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"FighterID\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 530,\n        \"min\": 0,\n        
\"max\": 2112,\n        \"num_unique_values\": 2113,\n        
\"samples\": [\n          1229,\n          1759,\n          680\n      
],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n  
}\n    },\n    {\n      \"column\": \"AvgSigStrLanded\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
19.728279390590792,\n        \"min\": 0.0,\n        \"max\": 154.0,\n  
\"num_unique_values\": 3338,\n        \"samples\": [\n          6.6,\n 
16.2375,\n          52.8\n        ],\n        \"semantic_type\": 
\"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      
\"column\": \"AvgSigStrPct\",\n      \"properties\": {\n        
\"dtype\": \"number\",\n        \"std\": 0.104635925391437,\n        
\"min\": 0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 
1000,\n        \"samples\": [\n          0.48675,\n          
0.33224999999999993,\n          0.4363333333333333\n        ],\n       
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"AvgSubAtt\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
0.6774831906040781,\n        \"min\": 0.0,\n        \"max\": 8.4,\n    
\"num_unique_values\": 708,\n        \"samples\": [\n          
1.5333,\n          0.766675,\n          0.5555666666666667\
n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"AvgTDLanded\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 1.3154307190029486,\n        \"min\": 
0.0,\n        \"max\": 12.5,\n        \"num_unique_values\": 1419,\n   

\"samples\": [\n          0.6,\n          1.0047333333333333,\n        
12.5\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"AvgTDPct\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 0.22867403997345281,\n        \"min\": 
0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 1155,\n    
\"samples\": [\n          0.026727272727272725,\n          0.56,\n     
0.26566666666666666\n        ],\n        \"semantic_type\": \"\",\n    
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"FinishRound\",\n      \"properties\": {\n        \"dtype\": 
\"number\",\n        \"std\": 0,\n        \"min\": 1,\n        
\"max\": 5,\n        \"num_unique_values\": 5,\n        \"samples\": 
[\n          1,\n          4,\n          2\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"Historical_AvgSigStrLanded\",\n    
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
17.079569784004295,\n        \"min\": 0.0,\n        \"max\": 
113.6667,\n        \"num_unique_values\": 8829,\n        \"samples\": 
[\n          27.562525,\n          3.196363636363636,\n          
35.25\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"Historical_AvgSigStrPct\",\n      \"properties\": {\n        
\"dtype\": \"number\",\n        \"std\": 0.09952441697478792,\n        
\"min\": 0.0,\n        \"max\": 1.0,\n        \"num_unique_values\": 
6290,\n        \"samples\": [\n          0.5488999999999999,\n         
0.498,\n          0.5094202898550725\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"Historical_AvgSubAtt\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
0.6607920794184887,\n        \"min\": 0.0,\n        \"max\": 8.4,\n    
\"num_unique_values\": 5343,\n        \"samples\": [\n          
0.14666,\n          0.3464333333333333,\n          0.832175\
n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"Historical_AvgTDLanded\",\n      \"properties\": {\n        
\"dtype\": \"number\",\n        \"std\": 1.2541382783477995,\n        
\"min\": 0.0,\n        \"max\": 12.5,\n        \"num_unique_values\": 
7301,\n        \"samples\": [\n          0.86459375,\n          
0.89312,\n          2.2450400000000004\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    },\n    {\n      \"column\": \"Historical_AvgTDPct\",\n      
\"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 
0.21283854726629425,\n        \"min\": 0.0,\n        \"max\": 1.0,\n   
\"num_unique_values\": 6413,\n        \"samples\": [\n          
0.23650000000000002,\n          0.6196666666666666,\n          
0.35175\n        ],\n        \"semantic_type\": \"\",\n        
\"description\": \"\"\n      }\n    },\n    {\n      \"column\": 
\"Historical_FinishRound\",\n      \"properties\": {\n        
\"dtype\": \"number\",\n        \"std\": 0.5999723743632367,\n        

\"min\": 1.0,\n        \"max\": 5.0,\n        \"num_unique_values\": 
379,\n        \"samples\": [\n          2.0689655172413794,\n          
1.8823529411764706,\n          3.4210526315789473\n        ],\n        
\"semantic_type\": \"\",\n        \"description\": \"\"\n      }\
n    }\n  ]\n}","type":"dataframe","variable_name":"fighter_long"}

# 1) For red corner, we select columns from fighter_long we want
red_merge_cols = [
    'FightID', 'FighterID',
    'Historical_AvgSigStrLanded',
    'Historical_AvgSigStrPct',
    'Historical_AvgSubAtt',
    'Historical_AvgTDLanded',
    'Historical_AvgTDPct',
    'Historical_FinishRound'
    # etc.
]

red_historical = fighter_long[red_merge_cols].copy()

df = pd.merge(
    df,
    red_historical,
    how='left',
    left_on=['FightID','RedFighterID'],
    right_on=['FightID','FighterID']
)

# rename
df.rename(columns={
    'Historical_AvgSigStrLanded': 'RedHistorical_AvgSigStrLanded',
    'Historical_AvgSigStrPct':    'RedHistorical_AvgSigStrPct',
    'Historical_AvgSubAtt':       'RedHistorical_AvgSubAtt',
    'Historical_AvgTDLanded':     'RedHistorical_AvgTDLanded',
    'Historical_AvgTDPct':        'RedHistorical_AvgTDPct',
    'Historical_FinishRound':     'RedHistorical_FinishRound'
}, inplace=True)

# drop the right-merge FighterID
df.drop(columns='FighterID', inplace=True)

# 2) Do the same for the blue corner
blue_merge_cols = red_merge_cols  # same structure
blue_historical = fighter_long[blue_merge_cols].copy()

df = pd.merge(
    df,
    blue_historical,
    how='left',
    left_on=['FightID','BlueFighterID'],

    right_on=['FightID','FighterID']
)

df.rename(columns={
    'Historical_AvgSigStrLanded': 'BlueHistorical_AvgSigStrLanded',
    'Historical_AvgSigStrPct':    'BlueHistorical_AvgSigStrPct',
    'Historical_AvgSubAtt':       'BlueHistorical_AvgSubAtt',
    'Historical_AvgTDLanded':     'BlueHistorical_AvgTDLanded',
    'Historical_AvgTDPct':        'BlueHistorical_AvgTDPct',
    'Historical_FinishRound':     'BlueHistorical_FinishRound'
}, inplace=True)

df.drop(columns='FighterID', inplace=True, errors='ignore')

bo_nickal_id = fighter_to_id.get('Bo Nickal')

if bo_nickal_id is not None:
    print(f"Bo Nickal ID: {bo_nickal_id}")

    bo_nickal_rows = fighter_long[fighter_long['FighterID'] == 
bo_nickal_id]

    historical_cols = [col for col in bo_nickal_rows.columns if 
col.startswith('Historical_')]

    print(bo_nickal_rows[['FightID','FighterID'] + historical_cols])

else:
    print("no")

Bo Nickal ID: 42
      FightID  FighterID  Historical_AvgSigStrLanded  
Historical_AvgSigStrPct  \
5598     5598         42                         NaN                   
NaN   
5784     5784         42                        1.64                   
0.62   
6179     6179         42                        1.64                   
0.62   
6491     6491         42                        1.64                   
0.62   

      Historical_AvgSubAtt  Historical_AvgTDLanded  
Historical_AvgTDPct  \
5598                   NaN                     NaN                  
NaN   
5784                   7.5                    7.46                  

0.5   
6179                   7.5                    7.46                  
0.5   
6491                   7.5                    7.46                  
0.5   

      Historical_FinishRound  
5598                     NaN  
5784                1.000000  
6179                1.000000  
6491                1.333333  

historical_cols = [col for col in df.columns if 
col.startswith(('RedHistorical_', 'BlueHistorical_'))]
for col in historical_cols:
   df[col] = df[col].fillna(0)

ELO

from collections import defaultdict, deque

def get_finish_points(finish):
    """
    Returns (winner_base_pts, loser_base_pts)
    based on finish method.
    """
    if finish in ['DQ', 'Overturned', None]:
        # No change if it's a DQ or Overturned (custom choice)
        return (0.5, 0.5)

    finish = finish.upper()
    if finish in ['U-DEC', 'M-DEC']:
        return (5.0, -3.0)
    elif finish == 'S-DEC':
        return (3.0, -2.0)
    elif finish in ['KO/TKO', 'SUB']:
        return (6.5, -4.5)
    else:
        # Unrecognized finish, no change
        return (0, 0)

def get_winner_rank_bonus(opponent_rank):
    """
    Additional points for the winner based on the opponent's rank 
category.
    """
    if opponent_rank == 'okay fighter':
        return 1.0
    elif opponent_rank == 'good level':
        return 3

    elif opponent_rank == 'high level':
        return 6.0
    return 0.0

def get_loser_rank_penalty(my_rank, opponent_rank):
    """
    Additional penalty for the loser based on rank relationship.
    - Opponent > me => 0
    - Opponent == me => -1
    - Opponent < me => -2.5
    """
    rank_map = {
        'okay fighter': 1,
        'good level': 2,
        'high level': 3
    }
    my_val  = rank_map.get(my_rank, 1)
    opp_val = rank_map.get(opponent_rank, 1)

    if opp_val > my_val:
        return -0.5  # lost to higher rank => no extra penalty
    elif opp_val == my_val:
        return -2.0
    else:
        # opp_val < my_val
        return -3.0

def transitive_bonus(winner_id, loser_id, fighter_recent_wins, 
fighter_recent_losses):
    """
    +1.5 if there's at least one fighter that the winner has beaten 
(in last 5 fights)
    who also has beaten the loser (in last 5 fights).
    That means:
        intersection( winner's recent_wins, loser's recent_losses ) != 
empty
    """
    w_recent_wins = set(fighter_recent_wins[winner_id])
    l_recent_losses = set(fighter_recent_losses[loser_id])
    intersection = w_recent_wins.intersection(l_recent_losses)
    if len(intersection) > 0:
        return 1.5
    return 0.0

import pandas as pd
from collections import defaultdict, deque

# 1) Sort the DataFrame by Date
#df = df.sort_values('Date').reset_index(drop=True)

# 2) Initialize
BASE_ELO = 10.0

# For storing the last 5 fight deltas
fighter_deltas = defaultdict(lambda: deque(maxlen=5))
# For storing the last 5 opponents each fighter has beaten or lost to
fighter_recent_wins = defaultdict(lambda: deque(maxlen=5))
fighter_recent_losses = defaultdict(lambda: deque(maxlen=5))

# We'll store pre-fight ELO in these columns
df['RedElo'] = 0.0
df['BlueElo'] = 0.0

# 3) Iterate fights in chronological order
for idx, row in df.iterrows():
    red_id = row['RedFighterID']
    blue_id = row['BlueFighterID']
    red_rank = row['RedRankCategory']
    blue_rank = row['BlueRankCategory']
    finish   = row['Finish']
    winner   = row['Winner']  # 'Red','Blue', or something else

    # PRE-FIGHT ELOs (sum of last 5 deltas + baseline)
    red_pre_elo  = BASE_ELO + sum(fighter_deltas[red_id])
    blue_pre_elo = BASE_ELO + sum(fighter_deltas[blue_id])

    # Store them
    df.at[idx, 'RedElo']  = red_pre_elo
    df.at[idx, 'BlueElo'] = blue_pre_elo

    # If no clear winner, skip ELO updates
    if winner not in ['Red','Blue']:
        continue

    # Identify winner & loser
    if winner == 'Red':
        w_id, w_rank = red_id, red_rank
        l_id, l_rank = blue_id, blue_rank
    else:  # winner == 'Blue'
        w_id, w_rank = blue_id, blue_rank
        l_id, l_rank = red_id, red_rank

    # Base points from finish method
    w_pts, l_pts = get_finish_points(finish)  # w_pts > 0, l_pts < 0

    # Winner rank bonus
    # Opponent's rank is the loser's rank

    winner_rank_bonus = get_winner_rank_bonus(l_rank)

    # Loser rank penalty
    # from the loser's perspective, my_rank = l_rank, opp_rank = 
w_rank
    loser_rank_penalty = get_loser_rank_penalty(l_rank, w_rank)

    # Transitive bonus
    # only consider last 5 fights
    trans_bonus = transitive_bonus(w_id, l_id, fighter_recent_wins, 
fighter_recent_losses)

    # Sum up final deltas
    winner_delta = w_pts + winner_rank_bonus + trans_bonus   # 
typically positive
    loser_delta  = l_pts + loser_rank_penalty                # 
typically negative

    # Update the fighter's deque with new deltas
    fighter_deltas[w_id].append(winner_delta)
    fighter_deltas[l_id].append(loser_delta)

    # Update recent wins/losses
    fighter_recent_wins[w_id].append(l_id)
    fighter_recent_losses[l_id].append(w_id)

fighters = [
    "Dricus Du Plessis",
    "Sean Strickland",
    "Nassourdine Imavov",
    "Khamzat Chimaev",
    "Israel Adesanya",
    "Robert Whittaker",
    "Caio Borralho",
    "Jared Cannonier",
    "Marvin Vettori",
    "Brendan Allen",
    "Roman Dolidze",
    "Paulo Costa",
    "Anthony Hernandez",
    "Michel Pereira",
    "Roman Kopylov",
    "Bo Nickal",
]

# Create a dictionary to store the most recent Elo points for each 
fighter
most_recent_elo = {}

# Iterate through the fighters
for fighter in fighters:
    # Filter the DataFrame for fights involving the current fighter
    fighter_fights = df[(df['RedFighter'] == fighter) | 
(df['BlueFighter'] == fighter)]

    # Get the most recent fight (last row)
    most_recent_fight = fighter_fights.iloc[-1]

    # Extract the Elo point for the fighter from the most recent fight
    if most_recent_fight['RedFighter'] == fighter:
        elo_point = most_recent_fight['RedElo']
    else:
        elo_point = most_recent_fight['BlueElo']

    # Store the Elo point in the dictionary
    most_recent_elo[fighter] = elo_point

# Print the most recent Elo points for each fighter
for fighter, elo in most_recent_elo.items():
    print(f"{fighter}: {elo:.2f}")

Dricus Du Plessis: 61.50
Sean Strickland: 30.00
Nassourdine Imavov: 41.00
Khamzat Chimaev: 56.50
Israel Adesanya: 34.50
Robert Whittaker: 26.50
Caio Borralho: 45.00
Jared Cannonier: 28.50
Marvin Vettori: 27.00
Brendan Allen: 50.00
Roman Dolidze: 26.50
Paulo Costa: 13.50
Anthony Hernandez: 46.00
Michel Pereira: 44.50
Roman Kopylov: 35.00
Bo Nickal: 32.50

fighters = [
    "Islam Makhachev",
    "Arman Tsarukyan",
    "Charles Oliveira",
    "Justin Gaethje",
    "Dustin Poirier",

    "Max Holloway",
    "Dan Hooker",
    "Michael Chandler",
    "Mateusz Gamrot",
    "Beneil Dariush",
    "Renato Moicano",
    "Rafael Fiziev",
    "Paddy Pimblett",
    "Jalin Turner",
    "Benoit Saint Denis",
    "Grant Dawson"
]

# Create a dictionary to store the most recent Elo points for each 
fighter
most_recent_elo = {}

# Iterate through the fighters
for fighter in fighters:
    # Filter the DataFrame for fights involving the current fighter
    fighter_fights = df[(df['RedFighter'] == fighter) | 
(df['BlueFighter'] == fighter)]

    # Get the most recent fight (last row)
    if not fighter_fights.empty:
        most_recent_fight = fighter_fights.iloc[-1]

        # Extract the Elo point for the fighter from the most recent 
fight
        if most_recent_fight['RedFighter'] == fighter:
            elo_point = most_recent_fight['RedElo']
        else:
            elo_point = most_recent_fight['BlueElo']

        # Store the Elo point in the dictionary
        most_recent_elo[fighter] = elo_point
    else:
        print(f"No fights found for {fighter}, Elo set to base: 
{BASE_ELO}")
        most_recent_elo[fighter] = BASE_ELO # handle new fighters or 
those with no data yet

# Print the most recent Elo points for each fighter
for fighter, elo in most_recent_elo.items():
    print(f"{fighter}: {elo:.2f}")

Islam Makhachev: 64.50
Arman Tsarukyan: 40.50
Charles Oliveira: 38.50
Justin Gaethje: 28.50

Dustin Poirier: 26.50
Max Holloway: 52.00
Dan Hooker: 18.00
Michael Chandler: 11.00
Mateusz Gamrot: 38.00
Beneil Dariush: 36.00
Renato Moicano: 40.50
Rafael Fiziev: 39.00
Paddy Pimblett: 44.50
Jalin Turner: 26.50
Benoit Saint Denis: 37.00
Grant Dawson: 33.00

fighters = [
    "Belal Muhammad",
    "Leon Edwards",
    "Shavkat Rakhmonov",
    "Kamaru Usman",
    "Jack Della Maddalena",
    "Sean Brady",
    "Joaquin Buckley",
    "Ian Machado Garry",
    "Gilbert Burns",
    "Colby Covington",
    "Geoff Neal",
    "Stephen Thompson",
    "Michael Morales",
    "Carlos Prates",
    "Vicente Luque",
    "Michael Page"
]

# Create a dictionary to store the most recent Elo points for each 
fighter
most_recent_elo = {}

# Iterate through the fighters
for fighter in fighters:
    # Filter the DataFrame for fights involving the current fighter
    fighter_fights = df[(df['RedFighter'] == fighter) | 
(df['BlueFighter'] == fighter)]

    # Get the most recent fight (last row) if available
    if not fighter_fights.empty:
        most_recent_fight = fighter_fights.iloc[-1]

        # Extract the Elo point for the fighter from the most recent 
fight
        if most_recent_fight['RedFighter'] == fighter:
            elo_point = most_recent_fight['RedElo']

        else:
            elo_point = most_recent_fight['BlueElo']

        # Store the Elo point in the dictionary
        most_recent_elo[fighter] = elo_point
    else:
        print(f"No fights found for {fighter}, Elo set to base: 
{BASE_ELO}")
        most_recent_elo[fighter] = BASE_ELO  # Handle new or fighters 
with no data

# Print the most recent Elo points for each fighter
for fighter, elo in most_recent_elo.items():
    print(f"{fighter}: {elo:.2f}")

Belal Muhammad: 62.00
Leon Edwards: 64.50
Shavkat Rakhmonov: 55.00
Kamaru Usman: 34.50
Jack Della Maddalena: 42.50
Sean Brady: 37.50
Joaquin Buckley: 48.50
Ian Machado Garry: 50.50
Gilbert Burns: 10.50
Colby Covington: 15.50
Geoff Neal: 11.50
Stephen Thompson: 7.50
Michael Morales: 37.00
Carlos Prates: 32.50
Vicente Luque: 12.00
Michael Page: 18.00

# Filter for Colby Covington's fights
covington_fights = df[(df['RedFighter'] == 'Colby Covington') | 
(df['BlueFighter'] == 'Colby Covington')]

# Get the last 6 fights
last_6_fights = covington_fights.tail(6)

# Select relevant columns and display
selected_columns = ['Date', 'RedFighter', 'BlueFighter', 'RedElo', 
'BlueElo', 'Winner']
covington_fight_data = last_6_fights[selected_columns]

# Display the data
print(covington_fight_data)

            Date       RedFighter      BlueFighter  RedElo  BlueElo 
Winner
4047  2019-12-14     Kamaru Usman  Colby Covington    56.5     55.5    

Red
4375  2020-09-19  Colby Covington    Tyron Woodley    43.0     33.5    
Red
4943  2021-11-06     Kamaru Usman  Colby Covington    71.0     47.5    
Red
5089  2022-03-05  Colby Covington   Jorge Masvidal    31.5     33.0    
Red
6016  2023-12-16     Leon Edwards  Colby Covington    58.0     30.0    
Red
6540  2024-12-14  Colby Covington  Joaquin Buckley    15.5     48.5   
Blue

df.columns.to_list()

['RedFighter',
 'BlueFighter',
 'RedOdds',
 'BlueOdds',
 'RedExpectedValue',
 'BlueExpectedValue',
 'Date',
 'Winner',
 'TitleBout',
 'WeightClass',
 'Gender',
 'NumberOfRounds',
 'BlueCurrentLoseStreak',
 'BlueCurrentWinStreak',
 'BlueDraws',
 'BlueAvgSigStrLanded',
 'BlueAvgSigStrPct',
 'BlueAvgSubAtt',
 'BlueAvgTDLanded',
 'BlueAvgTDPct',
 'BlueLongestWinStreak',
 'BlueLosses',
 'BlueTotalRoundsFought',
 'BlueTotalTitleBouts',
 'BlueWinsByDecisionMajority',
 'BlueWinsByDecisionSplit',
 'BlueWinsByDecisionUnanimous',
 'BlueWinsByKO',
 'BlueWinsBySubmission',
 'BlueWinsByTKODoctorStoppage',
 'BlueWins',
 'BlueStance',
 'BlueHeightCms',
 'BlueReachCms',
 'BlueWeightLbs',
 'RedCurrentLoseStreak',

 'RedCurrentWinStreak',
 'RedDraws',
 'RedAvgSigStrLanded',
 'RedAvgSigStrPct',
 'RedAvgSubAtt',
 'RedAvgTDLanded',
 'RedAvgTDPct',
 'RedLongestWinStreak',
 'RedLosses',
 'RedTotalRoundsFought',
 'RedTotalTitleBouts',
 'RedWinsByDecisionMajority',
 'RedWinsByDecisionSplit',
 'RedWinsByDecisionUnanimous',
 'RedWinsByKO',
 'RedWinsBySubmission',
 'RedWinsByTKODoctorStoppage',
 'RedWins',
 'RedStance',
 'RedHeightCms',
 'RedReachCms',
 'RedWeightLbs',
 'RedAge',
 'BlueAge',
 'LoseStreakDif',
 'WinStreakDif',
 'LongestWinStreakDif',
 'WinDif',
 'LossDif',
 'TotalRoundDif',
 'TotalTitleBoutDif',
 'KODif',
 'SubDif',
 'HeightDif',
 'ReachDif',
 'AgeDif',
 'SigStrDif',
 'AvgSubAttDif',
 'AvgTDDif',
 'BMatchWCRank',
 'RMatchWCRank',
 'RPFPRank',
 'BPFPRank',
 'BetterRank',
 'Finish',
 'FinishRound',
 'TotalFightTimeSecs',
 'RedDecOdds',
 'BlueDecOdds',

 'RSubOdds',
 'BSubOdds',
 'RKOOdds',
 'BKOOdds',
 'RedRankCategory',
 'BlueRankCategory',
 'RedFighterID',
 'BlueFighterID',
 'FightID',
 'RedHistorical_AvgSigStrLanded',
 'RedHistorical_AvgSigStrPct',
 'RedHistorical_AvgSubAtt',
 'RedHistorical_AvgTDLanded',
 'RedHistorical_AvgTDPct',
 'RedHistorical_FinishRound',
 'BlueHistorical_AvgSigStrLanded',
 'BlueHistorical_AvgSigStrPct',
 'BlueHistorical_AvgSubAtt',
 'BlueHistorical_AvgTDLanded',
 'BlueHistorical_AvgTDPct',
 'BlueHistorical_FinishRound',
 'RedElo',
 'BlueElo']

# 1. Define columns to exclude to prevent data leakage
exclude_cols = ['Finish', 'FinishDetails', 'FinishRound', 
'FinishRoundTime', 'TotalFightTimeSecs',
                'RedAvgSigStrLanded', 'RedAvgSigStrPct', 
'RedAvgSubAtt', 'RedAvgTDLanded', 'RedAvgTDPct',
                'BlueAvgSigStrLanded', 'BlueAvgSigStrPct', 
'BlueAvgSubAtt', 'BlueAvgTDLanded', 'BlueAvgTDPct',
                'Winner','Date','SigStrDif', 
'AvgSubAttDif','AvgTDDif',]  # Exclude 'Winner' as we'll create a 
binary version

# 2. Create final_feature_cols using list comprehension
final_feature_cols = [col for col in df.columns if col not in 
exclude_cols]

# 3. Encode 'Winner' column as binary
if 'WinnerBinary' not in df.columns:
        df['WinnerBinary'] = df['Winner'].map({'Red': 0, 'Blue': 1})

target_col = 'WinnerBinary'

# 4. Create df_model
df_model = df[final_feature_cols + [target_col]].copy()

print("Columns in df_model:", df_model.columns.tolist())
df_model.head()

Columns in df_model: ['RedFighter', 'BlueFighter', 'RedOdds', 
'BlueOdds', 'RedExpectedValue', 'BlueExpectedValue', 'TitleBout', 
'WeightClass', 'Gender', 'NumberOfRounds', 'BlueCurrentLoseStreak', 
'BlueCurrentWinStreak', 'BlueDraws', 'BlueLongestWinStreak', 
'BlueLosses', 'BlueTotalRoundsFought', 'BlueTotalTitleBouts', 
'BlueWinsByDecisionMajority', 'BlueWinsByDecisionSplit', 
'BlueWinsByDecisionUnanimous', 'BlueWinsByKO', 'BlueWinsBySubmission', 
'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueStance', 
'BlueHeightCms', 'BlueReachCms', 'BlueWeightLbs', 
'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 
'RedLongestWinStreak', 'RedLosses', 'RedTotalRoundsFought', 
'RedTotalTitleBouts', 'RedWinsByDecisionMajority', 
'RedWinsByDecisionSplit', 'RedWinsByDecisionUnanimous', 'RedWinsByKO', 
'RedWinsBySubmission', 'RedWinsByTKODoctorStoppage', 'RedWins', 
'RedStance', 'RedHeightCms', 'RedReachCms', 'RedWeightLbs', 'RedAge', 
'BlueAge', 'LoseStreakDif', 'WinStreakDif', 'LongestWinStreakDif', 
'WinDif', 'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif', 'KODif', 
'SubDif', 'HeightDif', 'ReachDif', 'AgeDif', 'BMatchWCRank', 
'RMatchWCRank', 'RPFPRank', 'BPFPRank', 'BetterRank', 'RedDecOdds', 
'BlueDecOdds', 'RSubOdds', 'BSubOdds', 'RKOOdds', 'BKOOdds', 
'RedRankCategory', 'BlueRankCategory', 'RedFighterID', 
'BlueFighterID', 'FightID', 'RedHistorical_AvgSigStrLanded', 
'RedHistorical_AvgSigStrPct', 'RedHistorical_AvgSubAtt', 
'RedHistorical_AvgTDLanded', 'RedHistorical_AvgTDPct', 
'RedHistorical_FinishRound', 'BlueHistorical_AvgSigStrLanded', 
'BlueHistorical_AvgSigStrPct', 'BlueHistorical_AvgSubAtt', 
'BlueHistorical_AvgTDLanded', 'BlueHistorical_AvgTDPct', 
'BlueHistorical_FinishRound', 'RedElo', 'BlueElo', 'WinnerBinary']

{"type":"dataframe","variable_name":"df_model"}

categorical_cols = ['RedFighter', 'BlueFighter', 'WeightClass', 
'Gender',
    'BlueStance', 'RedStance', 'BetterRank', 'RedRankCategory', 
'BlueRankCategory','TitleBout']

# Label Encoding for fighters
!pip install scikit-learn
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

le_red = LabelEncoder()
le_blue = LabelEncoder()
df_model['RedFighter'] = le_red.fit_transform(df_model['RedFighter'])
df_model['BlueFighter'] = 
le_blue.fit_transform(df_model['BlueFighter'])

Requirement already satisfied: scikit-learn in 
/usr/local/lib/python3.11/dist-packages (1.6.1)
Requirement already satisfied: numpy>=1.19.5 in 
/usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.26.4)

Requirement already satisfied: scipy>=1.6.0 in 
/usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.13.1)
Requirement already satisfied: joblib>=1.2.0 in 
/usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in 
/usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.5.0)

rank_level_mapping = {'okay fighter': 0, 'good level': 1, 'high 
level': 2}
df_model['RedRankCategory'] = 
df_model['RedRankCategory'].map(rank_level_mapping)
df_model['BlueRankCategory'] = 
df_model['BlueRankCategory'].map(rank_level_mapping)

# One-Hot Encoding for other categorical variables
# Exclude rank levels as they've been ordinally encoded
one_hot_cols = [col for col in categorical_cols if col not in 
['RedRankCategory', 'BlueRankCategory', 'RedFighter', 'BlueFighter']]

df_model = pd.get_dummies(df_model, columns=one_hot_cols, 
drop_first=True)

print("Categorical variables encoded. One-Hot Encoding applied to:", 
one_hot_cols)

Categorical variables encoded. One-Hot Encoding applied to: 
['WeightClass', 'Gender', 'BlueStance', 'RedStance', 'BetterRank', 
'TitleBout']

TRYING BRO

df_model.dtypes.tolist()

[dtype('int64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),

 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),

 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('int64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('float64'),
 dtype('int64'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool'),
 dtype('bool')]

fighter_elo_dict = {}

df_model.columns.tolist()

['RedFighter',
 'BlueFighter',
 'RedOdds',

 'BlueOdds',
 'RedExpectedValue',
 'BlueExpectedValue',
 'NumberOfRounds',
 'BlueCurrentLoseStreak',
 'BlueCurrentWinStreak',
 'BlueDraws',
 'BlueLongestWinStreak',
 'BlueLosses',
 'BlueTotalRoundsFought',
 'BlueTotalTitleBouts',
 'BlueWinsByDecisionMajority',
 'BlueWinsByDecisionSplit',
 'BlueWinsByDecisionUnanimous',
 'BlueWinsByKO',
 'BlueWinsBySubmission',
 'BlueWinsByTKODoctorStoppage',
 'BlueWins',
 'BlueHeightCms',
 'BlueReachCms',
 'BlueWeightLbs',
 'RedCurrentLoseStreak',
 'RedCurrentWinStreak',
 'RedDraws',
 'RedLongestWinStreak',
 'RedLosses',
 'RedTotalRoundsFought',
 'RedTotalTitleBouts',
 'RedWinsByDecisionMajority',
 'RedWinsByDecisionSplit',
 'RedWinsByDecisionUnanimous',
 'RedWinsByKO',
 'RedWinsBySubmission',
 'RedWinsByTKODoctorStoppage',
 'RedWins',
 'RedHeightCms',
 'RedReachCms',
 'RedWeightLbs',
 'RedAge',
 'BlueAge',
 'LoseStreakDif',
 'WinStreakDif',
 'LongestWinStreakDif',
 'WinDif',
 'LossDif',
 'TotalRoundDif',
 'TotalTitleBoutDif',
 'KODif',
 'SubDif',

 'HeightDif',
 'ReachDif',
 'AgeDif',
 'BMatchWCRank',
 'RMatchWCRank',
 'RPFPRank',
 'BPFPRank',
 'RedDecOdds',
 'BlueDecOdds',
 'RSubOdds',
 'BSubOdds',
 'RKOOdds',
 'BKOOdds',
 'RedRankCategory',
 'BlueRankCategory',
 'RedFighterID',
 'BlueFighterID',
 'FightID',
 'RedHistorical_AvgSigStrLanded',
 'RedHistorical_AvgSigStrPct',
 'RedHistorical_AvgSubAtt',
 'RedHistorical_AvgTDLanded',
 'RedHistorical_AvgTDPct',
 'RedHistorical_FinishRound',
 'BlueHistorical_AvgSigStrLanded',
 'BlueHistorical_AvgSigStrPct',
 'BlueHistorical_AvgSubAtt',
 'BlueHistorical_AvgTDLanded',
 'BlueHistorical_AvgTDPct',
 'BlueHistorical_FinishRound',
 'RedElo',
 'BlueElo',
 'WinnerBinary',
 'WeightClass_Catch Weight',
 'WeightClass_Featherweight',
 'WeightClass_Flyweight',
 'WeightClass_Heavyweight',
 'WeightClass_Light Heavyweight',
 'WeightClass_Lightweight',
 'WeightClass_Middleweight',
 'WeightClass_Welterweight',
 "WeightClass_Women's Bantamweight",
 "WeightClass_Women's Featherweight",
 "WeightClass_Women's Flyweight",
 "WeightClass_Women's Strawweight",
 'Gender_MALE',
 'BlueStance_Orthodox',
 'BlueStance_Southpaw',
 'BlueStance_Switch',

 'BlueStance_Switch ',
 'RedStance_Orthodox',
 'RedStance_Southpaw',
 'RedStance_Switch',
 'BetterRank_Red',
 'BetterRank_neither',
 'TitleBout_True']

import random
# Set random seeds for reproducibility
random.seed(40)
np.random.seed(40)
torch.manual_seed(40)

<torch._C.Generator at 0x7d4105cd76f0>

target_col = 'WinnerBinary'
X = df_model.drop(columns=[target_col])
y = df_model[target_col].values

print("Feature shape:", X.shape, "Target shape:", y.shape)

Feature shape: (6541, 107) Target shape: (6541,)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

Train shape: (5232, 107) Test shape: (1309, 107)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Convert X to a numpy array before scaling
X_train_array = X_train.values.astype(float)
X_test_array  = X_test.values.astype(float)

# Fit on train, transform train & test
X_train_scaled = scaler.fit_transform(X_train_array)
X_test_scaled  = scaler.transform(X_test_array)

print("After scaling: ", X_train_scaled.shape, X_test_scaled.shape)

After scaling:  (5232, 107) (1309, 107)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Convert to torch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)  # classification 
=> long
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

class UFCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(UFCNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Create model
input_dim = X_train_scaled.shape[1]  # number of features
model = UFCNet(input_dim=input_dim, hidden_dim=64, output_dim=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

import os
# ============== TRAINING LOOP ==============
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # For train accuracy
    train_correct = 0
    train_total = 0

    # --- Training Phase ---
    for features_batch, labels_batch in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features_batch)
        loss = criterion(outputs, labels_batch)

        # Backprop
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * features_batch.size(0)

        # Compute train accuracy
        _, predicted = torch.max(outputs, dim=1)
        train_correct += (predicted == labels_batch).sum().item()
        train_total   += labels_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_acc = train_correct / train_total

    # --- Evaluation Phase ---
    model.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for feat_test, lab_test in test_loader:
            out_test = model(feat_test)
            _, pred_test = torch.max(out_test, dim=1)
            test_correct += (pred_test == lab_test).sum().item()
            test_total   += lab_test.size(0)

    test_acc = test_correct / test_total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {epoch_loss:.4f}, "
          f"Train Accuracy: {train_acc:.4f}, "
          f"Test Accuracy: {test_acc:.4f}")

# Create the directory if it doesn't exist
os.makedirs('ufc_model', exist_ok=True)

# Save the model
torch.save(model.state_dict(), 'ufc_model/model.pth')

Epoch [1/50], Loss: 0.6136, Train Accuracy: 0.6569, Test Accuracy: 
0.6860
Epoch [2/50], Loss: 0.5813, Train Accuracy: 0.6907, Test Accuracy: 
0.6944
Epoch [3/50], Loss: 0.5697, Train Accuracy: 0.7043, Test Accuracy: 
0.6952
Epoch [4/50], Loss: 0.5624, Train Accuracy: 0.7078, Test Accuracy: 
0.7097
Epoch [5/50], Loss: 0.5612, Train Accuracy: 0.7118, Test Accuracy: 
0.6952
Epoch [6/50], Loss: 0.5543, Train Accuracy: 0.7160, Test Accuracy: 
0.7036
Epoch [7/50], Loss: 0.5566, Train Accuracy: 0.7173, Test Accuracy: 
0.7036
Epoch [8/50], Loss: 0.5545, Train Accuracy: 0.7225, Test Accuracy: 
0.6990
Epoch [9/50], Loss: 0.5514, Train Accuracy: 0.7250, Test Accuracy: 
0.7005
Epoch [10/50], Loss: 0.5461, Train Accuracy: 0.7322, Test Accuracy: 
0.7005
Epoch [11/50], Loss: 0.5498, Train Accuracy: 0.7229, Test Accuracy: 
0.7013
Epoch [12/50], Loss: 0.5447, Train Accuracy: 0.7269, Test Accuracy: 
0.7013
Epoch [13/50], Loss: 0.5405, Train Accuracy: 0.7273, Test Accuracy: 
0.7028
Epoch [14/50], Loss: 0.5359, Train Accuracy: 0.7274, Test Accuracy: 
0.6952
Epoch [15/50], Loss: 0.5352, Train Accuracy: 0.7322, Test Accuracy: 
0.7082
Epoch [16/50], Loss: 0.5360, Train Accuracy: 0.7328, Test Accuracy: 
0.6929
Epoch [17/50], Loss: 0.5358, Train Accuracy: 0.7286, Test Accuracy: 
0.7097
Epoch [18/50], Loss: 0.5349, Train Accuracy: 0.7353, Test Accuracy: 
0.7051
Epoch [19/50], Loss: 0.5325, Train Accuracy: 0.7378, Test Accuracy: 
0.7074
Epoch [20/50], Loss: 0.5267, Train Accuracy: 0.7387, Test Accuracy: 
0.6937
Epoch [21/50], Loss: 0.5217, Train Accuracy: 0.7404, Test Accuracy: 
0.6944
Epoch [22/50], Loss: 0.5260, Train Accuracy: 0.7414, Test Accuracy: 
0.6960
Epoch [23/50], Loss: 0.5248, Train Accuracy: 0.7397, Test Accuracy: 
0.6982
Epoch [24/50], Loss: 0.5267, Train Accuracy: 0.7359, Test Accuracy: 
0.6998
Epoch [25/50], Loss: 0.5242, Train Accuracy: 0.7431, Test Accuracy: 
0.7005

Epoch [26/50], Loss: 0.5196, Train Accuracy: 0.7479, Test Accuracy: 
0.6967
Epoch [27/50], Loss: 0.5223, Train Accuracy: 0.7437, Test Accuracy: 
0.6883
Epoch [28/50], Loss: 0.5206, Train Accuracy: 0.7489, Test Accuracy: 
0.7143
Epoch [29/50], Loss: 0.5186, Train Accuracy: 0.7500, Test Accuracy: 
0.6975
Epoch [30/50], Loss: 0.5139, Train Accuracy: 0.7531, Test Accuracy: 
0.7021
Epoch [31/50], Loss: 0.5190, Train Accuracy: 0.7498, Test Accuracy: 
0.6960
Epoch [32/50], Loss: 0.5152, Train Accuracy: 0.7525, Test Accuracy: 
0.7066
Epoch [33/50], Loss: 0.5179, Train Accuracy: 0.7508, Test Accuracy: 
0.7059
Epoch [34/50], Loss: 0.5146, Train Accuracy: 0.7494, Test Accuracy: 
0.7097
Epoch [35/50], Loss: 0.5060, Train Accuracy: 0.7552, Test Accuracy: 
0.6975
Epoch [36/50], Loss: 0.5066, Train Accuracy: 0.7531, Test Accuracy: 
0.6998
Epoch [37/50], Loss: 0.5072, Train Accuracy: 0.7559, Test Accuracy: 
0.6990
Epoch [38/50], Loss: 0.5085, Train Accuracy: 0.7615, Test Accuracy: 
0.7120
Epoch [39/50], Loss: 0.5070, Train Accuracy: 0.7546, Test Accuracy: 
0.7021
Epoch [40/50], Loss: 0.5057, Train Accuracy: 0.7523, Test Accuracy: 
0.6830
Epoch [41/50], Loss: 0.5054, Train Accuracy: 0.7624, Test Accuracy: 
0.6853
Epoch [42/50], Loss: 0.5045, Train Accuracy: 0.7592, Test Accuracy: 
0.6937
Epoch [43/50], Loss: 0.5083, Train Accuracy: 0.7561, Test Accuracy: 
0.6960
Epoch [44/50], Loss: 0.5009, Train Accuracy: 0.7592, Test Accuracy: 
0.6845
Epoch [45/50], Loss: 0.5057, Train Accuracy: 0.7594, Test Accuracy: 
0.6860
Epoch [46/50], Loss: 0.5027, Train Accuracy: 0.7626, Test Accuracy: 
0.6921
Epoch [47/50], Loss: 0.5001, Train Accuracy: 0.7620, Test Accuracy: 
0.6875
Epoch [48/50], Loss: 0.4971, Train Accuracy: 0.7624, Test Accuracy: 
0.7028
Epoch [49/50], Loss: 0.4967, Train Accuracy: 0.7668, Test Accuracy: 
0.6967

Epoch [50/50], Loss: 0.4990, Train Accuracy: 0.7538, Test Accuracy: 
0.7028

from sklearn.metrics import classification_report, confusion_matrix

# ============== FINAL EVALUATION ==============
all_preds = []
all_true = []

model.eval()
with torch.no_grad():
    for feat_test, lab_test in test_loader:
        out_test = model(feat_test)
        _, pred_test = torch.max(out_test, dim=1)
        all_preds.extend(pred_test.cpu().numpy())
        all_true.extend(lab_test.cpu().numpy())

print("\n=== Final Test Evaluation ===")
print("Confusion Matrix:")
print(confusion_matrix(all_true, all_preds))
print()
print("Classification Report:")
print(classification_report(all_true, all_preds, 
target_names=["BlueWin","RedWin"]))

=== Final Test Evaluation ===
Confusion Matrix:
[[573 186]
 [203 347]]

Classification Report:
              precision    recall  f1-score   support

     BlueWin       0.74      0.75      0.75       759
      RedWin       0.65      0.63      0.64       550

    accuracy                           0.70      1309
   macro avg       0.69      0.69      0.69      1309
weighted avg       0.70      0.70      0.70      1309

# ============== OPTIONAL: PREDICTION FUNCTION ==============
def predict_new_fights(model, new_fights_features, scaler=None):
    """
    Predict outcomes (Blue=0 or Red=1) for new fights.
    new_fights_features: shape (n_samples, input_dim)
    If you used a scaler in training, pass the same 'scaler' to 
transform.
    Returns: (probabilities, predictions)
    """

    model.eval()

    if scaler is not None:
        new_fights_features = scaler.transform(new_fights_features)

    feats_t = torch.tensor(new_fights_features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(feats_t)
        probs = nn.Softmax(dim=1)(logits)
        _, preds = torch.max(probs, dim=1)

    return probs.cpu().numpy(), preds.cpu().numpy()

# Example usage:
# new_data = df_new_fights.values.astype(float)  # preprocessed the 
same as training data
# probs, preds = predict_new_fights(model, new_data, scaler=scaler)
# for i, (p, pr) in enumerate(zip(probs, preds)):
#     print(f"Fight {i}: Prob(Blue, Red) = {p}, Prediction = {pr} 
(0=Blue,1=Red)")

for column in df_model.columns:
    print(f"{column}: {df_model[column].isnull().sum()}")

RedFighter: 0
BlueFighter: 0
RedOdds: 0
BlueOdds: 0
RedExpectedValue: 0
BlueExpectedValue: 0
NumberOfRounds: 0
BlueCurrentLoseStreak: 0
BlueCurrentWinStreak: 0
BlueDraws: 0
BlueLongestWinStreak: 0
BlueLosses: 0
BlueTotalRoundsFought: 0
BlueTotalTitleBouts: 0
BlueWinsByDecisionMajority: 0
BlueWinsByDecisionSplit: 0
BlueWinsByDecisionUnanimous: 0
BlueWinsByKO: 0
BlueWinsBySubmission: 0
BlueWinsByTKODoctorStoppage: 0
BlueWins: 0
BlueHeightCms: 0

BlueReachCms: 0
BlueWeightLbs: 0
RedCurrentLoseStreak: 0
RedCurrentWinStreak: 0
RedDraws: 0
RedLongestWinStreak: 0
RedLosses: 0
RedTotalRoundsFought: 0
RedTotalTitleBouts: 0
RedWinsByDecisionMajority: 0
RedWinsByDecisionSplit: 0
RedWinsByDecisionUnanimous: 0
RedWinsByKO: 0
RedWinsBySubmission: 0
RedWinsByTKODoctorStoppage: 0
RedWins: 0
RedHeightCms: 0
RedReachCms: 0
RedWeightLbs: 0
RedAge: 0
BlueAge: 0
LoseStreakDif: 0
WinStreakDif: 0
LongestWinStreakDif: 0
WinDif: 0
LossDif: 0
TotalRoundDif: 0
TotalTitleBoutDif: 0
KODif: 0
SubDif: 0
HeightDif: 0
ReachDif: 0
AgeDif: 0
BMatchWCRank: 0
RMatchWCRank: 0
RPFPRank: 0
BPFPRank: 0
RedDecOdds: 0
BlueDecOdds: 0
RSubOdds: 0
BSubOdds: 0
RKOOdds: 0
BKOOdds: 0
RedRankCategory: 0
BlueRankCategory: 0
RedFighterID: 0
BlueFighterID: 0
FightID: 0
RedHistorical_AvgSigStrLanded: 0

RedHistorical_AvgSigStrPct: 0
RedHistorical_AvgSubAtt: 0
RedHistorical_AvgTDLanded: 0
RedHistorical_AvgTDPct: 0
RedHistorical_FinishRound: 0
BlueHistorical_AvgSigStrLanded: 0
BlueHistorical_AvgSigStrPct: 0
BlueHistorical_AvgSubAtt: 0
BlueHistorical_AvgTDLanded: 0
BlueHistorical_AvgTDPct: 0
BlueHistorical_FinishRound: 0
RedElo: 0
BlueElo: 0
WinnerBinary: 0
WeightClass_Catch Weight: 0
WeightClass_Featherweight: 0
WeightClass_Flyweight: 0
WeightClass_Heavyweight: 0
WeightClass_Light Heavyweight: 0
WeightClass_Lightweight: 0
WeightClass_Middleweight: 0
WeightClass_Welterweight: 0
WeightClass_Women's Bantamweight: 0
WeightClass_Women's Featherweight: 0
WeightClass_Women's Flyweight: 0
WeightClass_Women's Strawweight: 0
Gender_MALE: 0
BlueStance_Orthodox: 0
BlueStance_Southpaw: 0
BlueStance_Switch: 0
BlueStance_Switch : 0
RedStance_Orthodox: 0
RedStance_Southpaw: 0
RedStance_Switch: 0
BetterRank_Red: 0
BetterRank_neither: 0
TitleBout_True: 0

# Filter for null values in historical columns
historical_cols = [col for col in df.columns if 
col.startswith(('RedHistorical_', 'BlueHistorical_'))]
null_rows = df[df[historical_cols].isnull().any(axis=1)]

# Get unique fighter IDs
unique_fighters = pd.concat([null_rows['RedFighterID'], 
null_rows['BlueFighterID']]).unique()

# Count fight appearances for each unique fighter
fighter_counts = {}
for fighter_id in unique_fighters:

    fighter_counts[fighter_id] = len(df[(df['RedFighterID'] == 
fighter_id) | (df['BlueFighterID'] == fighter_id)])

# Check if counts are 1 (indicating first fight)
first_fights = [fighter_id for fighter_id, count in 
fighter_counts.items() if count == 1]

print("Fighters with null historical values and only one fight (likely 
first fights):")
print(first_fights)

print("Total Number of these:")
print(len(first_fights))
# Filter the original DataFrame to see the rows corresponding to these 
fighters
first_fight_rows = df[(df['RedFighterID'].isin(first_fights)) | 
(df['BlueFighterID'].isin(first_fights))]
# Display or further analyze the first_fight_rows DataFrame
# ...

Fighters with null historical values and only one fight (likely first 
fights):
[]
Total Number of these:
0

# historical_cols = [col for col in df.columns if 
col.startswith(('RedHistorical_', 'BlueHistorical_'))]
# for col in historical_cols:
 #    df[col] = df[col].fillna(0)

print(X_train.columns.tolist())

['RedFighter', 'BlueFighter', 'RedOdds', 'BlueOdds', 
'RedExpectedValue', 'BlueExpectedValue', 'NumberOfRounds', 
'BlueCurrentLoseStreak', 'BlueCurrentWinStreak', 'BlueDraws', 
'BlueLongestWinStreak', 'BlueLosses', 'BlueTotalRoundsFought', 

'BlueTotalTitleBouts', 'BlueWinsByDecisionMajority', 
'BlueWinsByDecisionSplit', 'BlueWinsByDecisionUnanimous', 
'BlueWinsByKO', 'BlueWinsBySubmission', 'BlueWinsByTKODoctorStoppage', 
'BlueWins', 'BlueHeightCms', 'BlueReachCms', 'BlueWeightLbs', 
'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 
'RedLongestWinStreak', 'RedLosses', 'RedTotalRoundsFought', 
'RedTotalTitleBouts', 'RedWinsByDecisionMajority', 
'RedWinsByDecisionSplit', 'RedWinsByDecisionUnanimous', 'RedWinsByKO', 
'RedWinsBySubmission', 'RedWinsByTKODoctorStoppage', 'RedWins', 
'RedHeightCms', 'RedReachCms', 'RedWeightLbs', 'RedAge', 'BlueAge', 
'LoseStreakDif', 'WinStreakDif', 'LongestWinStreakDif', 'WinDif', 
'LossDif', 'TotalRoundDif', 'TotalTitleBoutDif', 'KODif', 'SubDif', 
'HeightDif', 'ReachDif', 'AgeDif', 'BMatchWCRank', 'RMatchWCRank', 
'RPFPRank', 'BPFPRank', 'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 
'BSubOdds', 'RKOOdds', 'BKOOdds', 'RedRankCategory', 
'BlueRankCategory', 'RedFighterID', 'BlueFighterID', 'FightID', 
'RedHistorical_AvgSigStrLanded', 'RedHistorical_AvgSigStrPct', 
'RedHistorical_AvgSubAtt', 'RedHistorical_AvgTDLanded', 
'RedHistorical_AvgTDPct', 'RedHistorical_FinishRound', 
'BlueHistorical_AvgSigStrLanded', 'BlueHistorical_AvgSigStrPct', 
'BlueHistorical_AvgSubAtt', 'BlueHistorical_AvgTDLanded', 
'BlueHistorical_AvgTDPct', 'BlueHistorical_FinishRound', 'RedElo', 
'BlueElo', 'WeightClass_Catch Weight', 'WeightClass_Featherweight', 
'WeightClass_Flyweight', 'WeightClass_Heavyweight', 'WeightClass_Light 
Heavyweight', 'WeightClass_Lightweight', 'WeightClass_Middleweight', 
'WeightClass_Welterweight', "WeightClass_Women's Bantamweight", 
"WeightClass_Women's Featherweight", "WeightClass_Women's Flyweight", 
"WeightClass_Women's Strawweight", 'Gender_MALE', 
'BlueStance_Orthodox', 'BlueStance_Southpaw', 'BlueStance_Switch', 
'BlueStance_Switch ', 'RedStance_Orthodox', 'RedStance_Southpaw', 
'RedStance_Switch', 'BetterRank_Red', 'BetterRank_neither', 
'TitleBout_True']

model_columns = X_train.columns

# 1. Filter for rows where either RedFighter or BlueFighter is Ian 
Garry
ian_garry_rows = df_model[(df_model['RedFighter'] == 
le_red.transform(['Ian Machado Garry'])[0]) | (df_model['BlueFighter'] 
== le_blue.transform(['Ian Machado Garry'])[0])]

# 2. Get the last row using tail(1)
last_ian_garry_row = ian_garry_rows.tail(1)

# 3. Display the row (optional: you can use to_string for full 
display)
print(last_ian_garry_row)  # or print(last_ian_garry_row.to_string())

      RedFighter  BlueFighter  RedOdds  BlueOdds  RedExpectedValue  \
6516        1442          718   -210.0     295.0            47.619   

      BlueExpectedValue  NumberOfRounds  BlueCurrentLoseStreak  \
6516              295.0               3                      0   

      BlueCurrentWinStreak  BlueDraws  ...  BlueStance_Orthodox  \
6516                     8          0  ...                 True   

      BlueStance_Southpaw  BlueStance_Switch  BlueStance_Switch   \
6516                False              False               False   

      RedStance_Orthodox  RedStance_Southpaw  RedStance_Switch  \
6516                True               False             False   

      BetterRank_Red  BetterRank_neither  TitleBout_True  
6516            True               False           False  

[1 rows x 108 columns]

# Assuming you used LabelEncoder to encode your fighter names:
# ... your LabelEncoder code from previous cells ...

# Get the original names back using inverse_transform
red_fighter_name  = le_red.inverse_transform([1441])[0]
blue_fighter_name = le_blue.inverse_transform([718])[0]

print(f"RedFighter 499: {red_fighter_name}")
print(f"BlueFighter 806: {blue_fighter_name}")

RedFighter 499: Shauna Bannon
BlueFighter 806: Ian Machado Garry

   model = UFCNet(input_dim=input_dim, hidden_dim=64, output_dim=2) # 
Create the model instance
   model.load_state_dict(torch.load('ufc_model/model.pth')) # Load the 
saved state
   model.eval()  # Set the model to evaluation mode

<ipython-input-57-d6c6335fd48c>:2: FutureWarning: You are using 
`torch.load` with `weights_only=False` (the current default value), 
which uses the default pickle module implicitly. It is possible to 
construct malicious pickle data which will execute arbitrary code 
during unpickling (See 
https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-
models for more details). In a future release, the default value for 
`weights_only` will be flipped to `True`. This limits the functions 
that could be executed during unpickling. Arbitrary objects will no 

longer be allowed to be loaded via this mode unless they are 
explicitly allowlisted by the user via 
`torch.serialization.add_safe_globals`. We recommend you start setting 
`weights_only=True` for any use case where you don't have full control 
of the loaded file. Please open an issue on GitHub for any issues 
related to this experimental feature.
  model.load_state_dict(torch.load('ufc_model/model.pth')) # Load the 
saved state

UFCNet(
  (net): Sequential(
    (0): Linear(in_features=107, out_features=128, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.25, inplace=False)
    (3): Linear(in_features=128, out_features=64, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.25, inplace=False)
    (6): Linear(in_features=64, out_features=2, bias=True)
  )
)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F

# 
----------------------------------------------------------------------
--------
# Define expected JIM columns for each side.
# 
----------------------------------------------------------------------
--------
red_columns = [
    'RedCurrentLoseStreak', 'RedCurrentWinStreak', 'RedDraws', 
'RedLongestWinStreak',
    'RedLosses', 'RedTotalRoundsFought', 'RedTotalTitleBouts', 
'RedWinsByDecisionMajority',
    'RedWinsByDecisionSplit', 'RedWinsByDecisionUnanimous', 
'RedWinsByKO', 'RedWinsBySubmission',
    'RedWinsByTKODoctorStoppage', 'RedWins', 'RedHeightCms', 
'RedReachCms', 'RedWeightLbs',
    'RedAge', 'RMatchWCRank', 'RPFPRank', 'RedRankCategory', 
'RedFighterID',
    'RedHistorical_AvgSigStrLanded', 'RedHistorical_AvgSigStrPct', 
'RedHistorical_AvgSubAtt',
    'RedHistorical_AvgTDLanded', 'RedHistorical_AvgTDPct', 
'RedHistorical_FinishRound', 'RedElo'

]

blue_columns = [
    'BlueCurrentLoseStreak', 'BlueCurrentWinStreak', 'BlueDraws', 
'BlueLongestWinStreak',
    'BlueLosses', 'BlueTotalRoundsFought', 'BlueTotalTitleBouts', 
'BlueWinsByDecisionMajority',
    'BlueWinsByDecisionSplit', 'BlueWinsByDecisionUnanimous', 
'BlueWinsByKO', 'BlueWinsBySubmission',
    'BlueWinsByTKODoctorStoppage', 'BlueWins', 'BlueHeightCms', 
'BlueReachCms', 'BlueWeightLbs',
    'BlueAge', 'BMatchWCRank', 'BPFPRank', 'BlueRankCategory', 
'BlueFighterID',
    'BlueHistorical_AvgSigStrLanded', 'BlueHistorical_AvgSigStrPct', 
'BlueHistorical_AvgSubAtt',
    'BlueHistorical_AvgTDLanded', 'BlueHistorical_AvgTDPct', 
'BlueHistorical_FinishRound', 'BlueElo'
]

# Create mapping dictionaries to convert between blue and red feature 
names
blue_to_red = {blue: red for blue, red in zip(blue_columns, 
red_columns)}
red_to_blue = {red: blue for red, blue in zip(red_columns, 
blue_columns)}

# 
----------------------------------------------------------------------
--------
# 1. Updated fighter stats extraction function (using encoded values)
# 
----------------------------------------------------------------------
--------
def get_fighter_stats(fighter_name, desired_side, df_model, le_red, 
le_blue):
    """
    Extract the most recent match stats (JIM features) for a fighter 
from both sides
    and choose the one with the highest FightID. Then, convert the 
stats so that they
    are in the format corresponding to the desired side ('red' or 
'blue').

    Parameters:
        fighter_name: the original fighter name (string)
        desired_side: 'red' or 'blue' (the side you want the stats 
returned in)
        df_model: the DataFrame with historical fight data (already 
encoded for fighter names)
        le_red, le_blue: the fitted label encoders for RedFighter and 

BlueFighter respectively.

    Returns:
        A dictionary of fighter stats with keys corresponding to the 
desired side.
    """
    # Get encoded values for the fighter
    fighter_encoded_red = le_red.transform([fighter_name])[0]
    fighter_encoded_blue = le_blue.transform([fighter_name])[0]

    # Filter matches where the fighter appears as Red or Blue
    subset_red = df_model[df_model['RedFighter'] == 
fighter_encoded_red]
    subset_blue = df_model[df_model['BlueFighter'] == 
fighter_encoded_blue]

    # If no matches found at all, raise an error.
    if subset_red.empty and subset_blue.empty:
        raise ValueError(f"No stats found for fighter {fighter_name} 
in either role.")

    # Choose the most recent match based on FightID (assuming higher 
FightID means more recent)
    red_latest = subset_red.loc[subset_red['FightID'].idxmax()] if not 
subset_red.empty else None
    blue_latest = subset_blue.loc[subset_blue['FightID'].idxmax()] if 
not subset_blue.empty else None

    # Compare FightID values if both exist; if only one exists, choose 
that one.
    if red_latest is not None and blue_latest is not None:
        if red_latest['FightID'] >= blue_latest['FightID']:
            chosen_row = red_latest
            source_side = 'red'
        else:
            chosen_row = blue_latest
            source_side = 'blue'
    elif red_latest is not None:
        chosen_row = red_latest
        source_side = 'red'
    else:
        chosen_row = blue_latest
        source_side = 'blue'

    # Based on the source side and the desired side, extract and 
possibly remap the stats.
    if desired_side.lower() == 'red':
        if source_side == 'red':
            stats = {col: chosen_row[col] for col in red_columns}
        else:

            # Remap blue stats to red keys
            stats_blue = {col: chosen_row[col] for col in 
blue_columns}
            stats = {blue_to_red[k]: v for k, v in stats_blue.items()}
    elif desired_side.lower() == 'blue':
        if source_side == 'blue':
            stats = {col: chosen_row[col] for col in blue_columns}
        else:
            # Remap red stats to blue keys
            stats_red = {col: chosen_row[col] for col in red_columns}
            stats = {red_to_blue[k]: v for k, v in stats_red.items()}
    else:
        raise ValueError("Side must be either 'red' or 'blue'.")

    return stats

# 
----------------------------------------------------------------------
--------
# 2. Function to compute DIF columns.
# 
----------------------------------------------------------------------
--------
def compute_differences(red_stats, blue_stats):
    """
    Compute the difference columns (DIF) from the red and blue stats.
    """
    diffs = {}
    diffs['LoseStreakDif']         = 
blue_stats['BlueCurrentLoseStreak'] - 
red_stats['RedCurrentLoseStreak']
    diffs['WinStreakDif']            = 
blue_stats['BlueCurrentWinStreak'] - red_stats['RedCurrentWinStreak']
    diffs['LongestWinStreakDif']     = 
blue_stats['BlueLongestWinStreak'] - red_stats['RedLongestWinStreak']
    diffs['WinDif']                  = blue_stats['BlueWins'] - 
red_stats['RedWins']
    diffs['LossDif']                 = blue_stats['BlueLosses'] - 
red_stats['RedLosses']
    diffs['TotalRoundDif']           = 
blue_stats['BlueTotalRoundsFought'] - 
red_stats['RedTotalRoundsFought']
    diffs['TotalTitleBoutDif']       = 
blue_stats['BlueTotalTitleBouts'] - red_stats['RedTotalTitleBouts']
    diffs['KODif']                   = (blue_stats['BlueWinsByKO'] + 
blue_stats['BlueWinsByTKODoctorStoppage']) - (red_stats['RedWinsByKO'] 
+ red_stats['RedWinsByTKODoctorStoppage'])
    diffs['SubDif']                  = 
blue_stats['BlueWinsBySubmission'] - red_stats['RedWinsBySubmission']

    diffs['HeightDif']               = blue_stats['BlueHeightCms'] - 
red_stats['RedHeightCms']
    diffs['ReachDif']                = blue_stats['BlueReachCms'] - 
red_stats['RedReachCms']
    diffs['AgeDif']                  = blue_stats['BlueAge'] - 
red_stats['RedAge']
    return diffs

# 
----------------------------------------------------------------------
--------
# 3. Function to create the new fight feature row combining manual 
inputs, fighter stats, and DIF columns.
# 
----------------------------------------------------------------------
--------
def create_fight_feature_row(red_fighter, blue_fighter, df_model, 
manual_inputs, le_red, le_blue):
    """
    Build a new row (as a DataFrame) that the model can use to predict 
a fight.

    Parameters:
      - red_fighter, blue_fighter: original fighter names (strings)
      - df_model: the training dataframe (used here to extract last-
match stats)
      - manual_inputs: a dictionary of values for columns you want to 
enter manually.
      - le_red, le_blue: fitted label encoders for fighter names.

    Returns:
      A pandas DataFrame with one row containing all pre-fight 
features.
    """
    # Get fighter stats (JIM features) using the updated extraction 
function.
    red_stats = get_fighter_stats(red_fighter, 'red', df_model, 
le_red, le_blue)
    blue_stats = get_fighter_stats(blue_fighter, 'blue', df_model, 
le_red, le_blue)

    # Compute difference columns (DIF)
    diffs = compute_differences(red_stats, blue_stats)

    # Create the new row dictionary.
    new_row = {}

    # Manual inputs: these are columns you want to enter manually.
    manual_columns = [
        'RedOdds', 'BlueOdds', 'RedExpectedValue', 

'BlueExpectedValue', 'NumberOfRounds',
        'RedDecOdds', 'BlueDecOdds', 'RSubOdds', 'BSubOdds', 
'RKOOdds', 'BKOOdds',
        'BetterRank_Red', 'BetterRank_neither', 'TitleBout_True', 
'FightID'
    ]

    # Add fighter names (they'll be encoded later in prepare_features)
    new_row['RedFighter'] = red_fighter
    new_row['BlueFighter'] = blue_fighter

    for col in manual_columns:
        new_row[col] = manual_inputs.get(col)

    # Add the JIM features (extracted fighter stats)
    for key, value in red_stats.items():
        new_row[key] = value
    for key, value in blue_stats.items():
        new_row[key] = value

    # Add the DIF columns
    for key, value in diffs.items():
        new_row[key] = value

    return pd.DataFrame([new_row])

# 
----------------------------------------------------------------------
--------
# 4. Function to prepare the features (encoding and scaling) before 
prediction.
# 
----------------------------------------------------------------------
--------
def prepare_features(new_df, le_red, le_blue, scaler, model_columns):
    """
    Prepare the new fight DataFrame:
      - Label encode fighter names.
      - One-hot encode categorical variables.
      - Align the columns with the training set (model_columns).
      - Scale the features.
    """
    # Label encode fighter names using the already fitted encoders.
    new_df['RedFighter'] = le_red.transform(new_df['RedFighter'])
    new_df['BlueFighter'] = le_blue.transform(new_df['BlueFighter'])

    # One-hot encode any other categorical variables if needed.
    new_df = pd.get_dummies(new_df, drop_first=True)

    # Reindex to match the training model columns (fill missing with 

0)
    new_df = new_df.reindex(columns=model_columns, fill_value=0)

    # Scale the features.
    scaled_features = scaler.transform(new_df.values.astype(float))

    return scaled_features

# 
----------------------------------------------------------------------
--------
# 5. Function to predict a new fight.
# 
----------------------------------------------------------------------
--------
def predict_new_fight(red_fighter, blue_fighter, manual_inputs, 
df_model, model, le_red, le_blue, scaler, model_columns):
    """
    Build the new fight feature row, prepare the features, and use the 
model to predict the outcome.

    Returns:
      predicted_winner: 'Red' or 'Blue'
      win_likelihood: probability for the predicted class
      new_fight_df: the constructed feature row (pre-scaled)
    """
     # Create new fight features.
    new_fight_df = create_fight_feature_row(red_fighter, blue_fighter, 
df_model, manual_inputs, le_red, le_blue)

    # Prepare features for prediction (encoding, aligning, scaling).
    X_new = prepare_features(new_fight_df.copy(), le_red, le_blue, 
scaler, model_columns)

     # Get prediction probabilities.
    with torch.no_grad():
        logits = model(torch.tensor(X_new, dtype=torch.float32)) # Get 
model output (logits)
        prob = F.softmax(logits, dim=1).cpu().numpy()[0]       # Apply 
softmax and get probabilities

    # Map model classes to fighter sides.
    # (Assuming WinnerBinary was mapped as {'Red': 0, 'Blue': 1})
    # Instead of model.classes_, use the index of the max probability
    predicted_class = np.argmax(prob)
    predicted_winner = 'Red' if predicted_class == 0 else 'Blue'
    win_likelihood = np.max(prob)

    return predicted_winner, win_likelihood, new_fight_df

# 
----------------------------------------------------------------------
--------
# 6. Example usage
# 
----------------------------------------------------------------------
--------
# Assume you already have:
# - df_model (the DataFrame with training data, where fighter names 
are encoded)
# - model (your trained classifier)
# - scaler (your fitted StandardScaler)
# - le_red and le_blue (fitted LabelEncoders for 'RedFighter' and 
'BlueFighter')
# - model_columns (the columns used during training; e.g., 
X_train.columns)

# For example, if X_train is your training feature DataFrame:
# model_columns = X_train.columns

# Manual inputs for columns that need to be provided manually.
manual_inputs = {
    'RedOdds': 100,
    'BlueOdds': 100,
    'RedExpectedValue': 0.5,
    'BlueExpectedValue': 0.5,
    'NumberOfRounds': 3,
    'RedDecOdds': 100,
    'BlueDecOdds': 100,
    'RSubOdds': 100,
    'BSubOdds': 100,
    'RKOOdds': 100,
    'BKOOdds': 100,
    'BetterRank_Red': 1,         # e.g., 1 means Red is better ranked
    'BetterRank_neither': 0,     # 0 means not neutral
    'TitleBout_True': False,
    'FightID': 3263.5              # arbitrary ID since model needs it
}

# Specify fighter names for the new fight.
red_fighter = "Tom Aspinall"
blue_fighter = "Serghei Spivac"

# Example try/except for prediction.
try:
    predicted_winner, win_likelihood, fight_features = 
predict_new_fight(
        red_fighter, blue_fighter, manual_inputs,
        df_model, model, le_red, le_blue, scaler, model_columns
    )

    print(f"Predicted winner: {predicted_winner}")
    print(f"Win likelihood: {win_likelihood:.2f}")

    print("\nNew fight feature row (pre-scaled):")
    print(fight_features)

except ValueError as e:
    print("Error during prediction:", e)

Predicted winner: Red
Win likelihood: 0.70

New fight feature row (pre-scaled):
     RedFighter     BlueFighter  RedOdds  BlueOdds  
RedExpectedValue  \
0  Tom Aspinall  Serghei Spivac      100       100               0.5   

   BlueExpectedValue  NumberOfRounds  RedDecOdds  BlueDecOdds  
RSubOdds  ...  \
0                0.5               3         100          100       
100  ...   

   LongestWinStreakDif  WinDif  LossDif  TotalRoundDif  
TotalTitleBoutDif  \
0                   -2       0        3             12                 
-1   

   KODif  SubDif  HeightDif  ReachDif  AgeDif  
0     -2       0      -5.08       0.0      -2  

[1 rows x 87 columns]

mean_fightid = df_model['FightID'].mean()
median_fightid = df_model['FightID'].median()

print("Mean FightID:", mean_fightid)
print("Median FightID:", median_fightid)

Mean FightID: 3270.0
Median FightID: 3270.0

fight_scaled = scaler.transform(fight_array)
print("Scaled fight row:", fight_scaled)

----------------------------------------------------------------------
-----

NameError                                 Traceback (most recent call 
last)
<ipython-input-60-f54c89ae3cec> in <cell line: 0>()
----> 1 fight_scaled = scaler.transform(fight_array)
      2 print("Scaled fight row:", fight_scaled)

NameError: name 'fight_array' is not defined

highest_fightid = df_model['FightID'].max()
most_recent_match = df_model[df_model['FightID'] == highest_fightid]
most_recent_match

# Assuming 'df_model' is your DataFrame
class_balance = df_model['WinnerBinary'].value_counts()
print(class_balance)

input_dim = X_train_scaled.shape[1]
model = UFCNet(input_dim=input_dim, hidden_dim=64, output_dim=2)
input_dim

column_order = X_train.columns.tolist()
print(column_order)

print("Scaler scales:", scaler.scale_)
print("Scaler means:", scaler.mean_)

