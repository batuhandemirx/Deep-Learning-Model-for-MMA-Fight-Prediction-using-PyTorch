## 1. Advanced Feature Engineering: Capturing Fighter Evolution
Static data is the enemy of sports prediction because it fails to account for the "narrative" of an athlete’s career. To solve this, we developed features that capture growth and momentum.

### Rank Categorization
We transformed granular weight class ranks into a streamlined, **three-tier ordinal system**:
* **High Level:** Top 5
* **Good Level:** Top 15
* **Okay Fighter:** Unranked

This categorization reduces noise and ensures the model prioritizes the massive impact of elite status over the minor fluctuations of lower-tier rankings.

### Temporal Lagging
To prevent **data leakage**, we utilized `expanding().mean().shift(1)` to create historical stat columns (e.g., `RedHistorical_AvgSigStrLanded`). 

> [!IMPORTANT]
> The `shift(1)` is the most critical operation; without it, the model would have 100% data leakage by "knowing" the fight's outcome before it happens. This ensures the model only evaluates what was known before the walk-outs.

### The Custom ELO Framework
We developed a proprietary, **zero-sum ELO system** to quantify momentum. This system moves beyond binary outcomes by weighting the quality of victory and the caliber of the opposition:

| Scenario | Point Delta (Winner / Loser) |
| :--- | :--- |
| **KO/TKO or Submission** | +6.5 / -4.5 |
| **Unanimous/Majority Decision** | +5.0 / -3.0 |
| **Split Decision** | +3.0 / -2.0 |
| **Defeating "High Level" Fighter** | +6.0 (Bonus) |
| **Losing to Lower-Ranked** | -3.0 (Penalty) |
| **Transitive Property Bonus** | +1.5 |

**Case Study:** Islam Makhachev’s ELO (**64.50**) vs. Michael Chandler’s (**11.00**) demonstrates the model's ability to differentiate between a dominant champion and a veteran facing a downturn.

---

## 2. UFCNet: Neural Network Architecture & Design
For a classification task involving **107 complex features**, we selected **PyTorch** to build **UFCNet**. Neural networks are uniquely capable of capturing interaction effects that linear models often struggle to detect.

### Architecture Specifications
* **Input Layer:** 107 features (Historical stats, ELO, vitals).
* **Hidden Layers:** 128 neurons → 64 neurons (ReLU activation).
* **Stability:** Implemented **Dropout (0.25)** and **Weight Decay (1e-2)** as L2 Regularization to prevent overfitting.

### Training Hyperparameters
* **Optimizer:** Adam (Learning Rate: `1e-3`)
* **Loss Function:** CrossEntropyLoss
* **Epochs:** 50
* **Batch Size:** 32

---

## 3. Empirical Results and Performance Metrics
The model's validity was tested using an **80/20 train-test split**, with `StandardScaler` applied to normalize the feature space.

### Final Performance Metrics
The model achieved a **70% Test Accuracy**. 

**Confusion Matrix (0: Red, 1: Blue):**
* **RedWin (0):** 573 True Positives / 186 False Positives
* **BlueWin (1):** 347 True Positives / 203 False Positives

**Classification Report:**
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **RedWin (0)** | 0.74 | 0.75 | 0.75 |
| **BlueWin (1)** | 0.65 | 0.63 | 0.64 |

---

## 4. Deployment and Predictive Inference
We utilized the `predict_new_fight` function to synthesize manual inputs (Odds, TitleBout, FightID) with extracted **JIM (Just-In-Match)** statistics.

**Test Case: Tom Aspinall vs. Serghei Spivac**
* **Result:** Predicted Winner: **Red (Aspinall)** with **70% Likelihood**.

### Technical Outlook and Enhancements
* **Cold-Start Solution:** Incorporating regional circuit data for debut fighters.
* **Financial Validation:** Calculating the **Sharpe Ratio** of predictions against betting lines.
* **Real-time Integration:** Integrating live betting shifts or bio-metric data.
