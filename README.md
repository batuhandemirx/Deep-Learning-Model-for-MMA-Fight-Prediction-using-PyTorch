# 🥊 UFC Fight Outcome Prediction using Deep Learning and Advanced Feature Engineering

## Overview

This project develops a machine learning framework for predicting UFC fight outcomes using historical fight statistics, fighter-level historical performance metrics, betting market information, and a custom-designed ELO rating system.

The goal of this project is to construct a robust predictive pipeline that incorporates:

- **Historical fighter performance**  
- **Dynamic fighter strength estimation**  
- **Ranking signals**  
- **Betting odds information**  
- **Stylistic matchup statistics**

The resulting dataset is transformed into a supervised learning problem where the model predicts whether the **Red** or **Blue** fighter wins the bout.

This project demonstrates several advanced data science techniques including:

- Large-scale feature engineering  
- Temporal feature construction  
- Dynamic rating systems  
- Prevention of data leakage  
- Domain-specific modeling strategies

The project is implemented in **Python** using **PyTorch**, **Pandas**, **NumPy**, and **Scikit-Learn**.

---

## Problem Statement

Predicting the outcome of mixed martial arts fights is a challenging problem due to:

- Stylistic matchups  
- Incomplete information  
- Rapidly evolving fighter skill levels  
- High variance in fight outcomes  

The objective of this project is to build a predictive model that estimates the probability that a fighter wins a fight, using **only information available before the fight occurs**.

> This constraint is critical in order to avoid data leakage and to ensure the model can be realistically deployed.

---

## Dataset

The dataset contains **6,541 UFC fights** and **2,113 unique fighters**.  

Each observation represents a single fight with two competitors:

- **Red corner fighter**  
- **Blue corner fighter**

The dataset includes:

### Fighter Attributes

- Age  
- Reach  
- Height  
- Weight  
- Stance

### Performance Statistics

- Average significant strikes landed  
- Striking accuracy  
- Takedown success rate  
- Submission attempts  
- Win/loss records

### Career Statistics

- Win streak  
- Losing streak  
- Number of rounds fought  
- Total wins  
- Wins by KO / submission / decision

### Betting Market Information

- Betting odds  
- Expected value  
- Finish method odds

### Ranking Signals

- Division ranking  
- Pound-for-pound ranking

### Fight Metadata

- Fight date  
- Number of rounds  
- Title bout indicator  
- Weight class

---

## Data Preprocessing

Several preprocessing steps were required before modeling.

### Column Removal

Columns with little predictive value or excessive missing data were removed, including:

- Location  
- Country  
- FinishDetails  
- FinishRoundTime  

> These fields were either redundant or unavailable before the fight.

### Handling Missing Values

Several strategies were used depending on the feature type:

**Mode Imputation**  
Categorical variables such as:

- Finish  
- BlueStance  

were imputed using the mode.

**Fighter-Level Imputation**  
For fighter statistics such as:

- Average strikes  
- Takedowns  
- Submission attempts  

Missing values were filled using the fighter's historical average, preserving fighter-specific skill characteristics.

**Global Mean Imputation**  
Remaining missing values were filled using overall dataset averages.

**Rank Imputation**  
Ranking columns contained many missing values because many fighters are unranked.  
Missing values were imputed as:  
max_rank + 1


> This encodes the idea that unranked fighters are worse than ranked fighters.

---

### Rank Category Engineering

Weight class ranking columns were consolidated into a simpler rank category feature.  

| Category       | Definition                  |
|----------------|-----------------------------|
| High Level     | Rank ≤ 5                    |
| Good Level     | Rank ≤ 15                   |
| Okay Fighter   | Unranked or lower           |

> This reduces dimensionality while preserving ranking signal.

---

### Fighter Identification System

Each fighter was mapped to a unique numerical ID.  

Example:
Conor McGregor → 0
Khabib Nurmagomedov → 1


> This allows efficient tracking of fighter histories across fights.

---

### Historical Performance Features

A key requirement is that the model must only use information available **before the fight occurs**.  

- Dataset transformed into a long fighter-history format  
- Each fighter's statistics aggregated using **expanding historical averages**

Example:

- `Historical_AvgSigStrLanded`  
- `Historical_AvgSigStrPct`  
- `Historical_AvgSubAtt`  
- `Historical_AvgTDLanded`  
- `Historical_AvgTDPct`  

> Computed using `expanding().mean().shift(1)` to exclude current fight statistics.  

These historical features were merged back into the fight dataset for both fighters.

---

## Custom ELO Rating System

A custom ELO rating system was designed to dynamically track fighter strength over time.

### Base ELO

- Each fighter begins with a baseline rating:
- BASE_ELO = 10

- - ELO scores are updated after each fight

### Finish Method Scoring

| Finish Type               | Winner Points | Loser Points |
|---------------------------|---------------|--------------|
| Decision (Unanimous/Majority) | +5            | -3           |
| Split Decision            | +3            | -2           |
| KO / TKO                  | +6.5          | -4.5         |
| Submission                | +6.5          | -4.5         |

> Dominant finishes impact rankings more strongly.

### Rank-Based Bonus System

| Opponent Rank  | Bonus Points |
|----------------|--------------|
| Okay Fighter    | +1           |
| Good Level      | +3           |
| High Level      | +6           |

> Rewards upsets and elite victories.

### Rank-Based Loss Penalty

| Situation               | Penalty          |
|-------------------------|----------------|
| Lost to stronger opponent | Small penalty  |
| Lost to equal opponent    | Moderate penalty |
| Lost to weaker opponent   | Large penalty |

### Transitive Victory Bonus

- If Fighter A beat Fighter B, and Fighter B beat Fighter C  
- Then Fighter A may receive additional bonus when beating Fighter C  

> Captures indirect dominance relationships.

### Recent Performance Memory

- Only last **5 fight deltas** are kept  
- Ensures rating reflects **recent performance** rather than career totals

---

## Feature Engineering Summary

The final dataset contains:

- **Fighter Attributes:** height, reach, weight, age, stance  
- **Performance Metrics:** historical striking & grappling statistics, win/loss differentials  
- **Ranking Signals:** division ranking, pound-for-pound ranking, rank category  
- **Market Information:** betting odds, finish method odds  
- **Dynamic Features:** custom ELO rating, historical averages, performance differentials

---

## Preventing Data Leakage

Columns removed because they contain **post-fight information**:

- Finish  
- FinishRound  
- TotalFightTimeSecs  
- Winner  

> Raw fight statistics for the current fight were removed. Only historical statistics were retained.

---

## Target Variable

Binary classification:

| WinnerBinary | Meaning          |
|--------------|----------------|
| 0            | Red fighter wins |
| 1            | Blue fighter wins |

---

## Modeling Approach

**Input:**  

- Fighter attributes  
- Historical statistics  
- ELO ratings  
- Betting market signals  
- Ranking signals  

**Output:**  

- Probability that Blue fighter wins  

**Modeling Framework:**

- Implemented in **PyTorch**  
- Potential architectures:
  - Feedforward neural networks  
  - Tabular deep learning models  
  - Gradient boosting models

---

### Example Fighter ELO Scores

| Fighter             | ELO  |
|--------------------|------|
| Islam Makhachev     | 64.5 |
| Leon Edwards        | 64.5 |
| Belal Muhammad      | 62   |
| Khamzat Chimaev     | 56.5 |
| Max Holloway        | 52   |

---

## Key Insights

- Historical performance metrics significantly improve predictive signal  
- Custom ELO ratings provide strong momentum summary  
- Betting market odds contain valuable information  
- Rank category features capture competitive tier differences  

---

## Limitations

- Stylistic matchups are difficult to quantify  
- Fighter injuries and camp changes are not included  
- Fight preparation quality is unobserved  
- Judging variance introduces randomness  

---

## Future Improvements

- **Graph Neural Networks:** model fighter interaction network  
- **Fight Style Embeddings:** represent fighters using learned stylistic embeddings  
- **Temporal Models:** LSTM, Transformer to model career progression  
- **Betting Strategy Simulation:** convert predictions into a sports betting strategy  

---

## Technologies Used

- Python  
- PyTorch  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  

---

## Conclusion

This project demonstrates how **advanced feature engineering**, **domain-specific rating systems**, and **temporal modeling techniques** can be combined to build a robust predictive system for UFC fight outcomes.

The pipeline illustrates real-world machine learning challenges such as:

- Temporal data leakage  
- Dynamic skill estimation  
- Missing data handling  
- Large-scale feature engineering  

> The resulting framework provides a strong foundation for future research into sports analytics and predictive modeling in combat sports.


