# Importing libraries for analysis
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm

# Load data (replace with your dataset)
data = pd.read_csv('research_data.csv')

# Quick look at the data to ensure it loaded correctly
print("Preview of data:")
print(data.head())

# Calculate descriptive statistics
print("\nSummary statistics for key variables:")
summary_stats = data[['Resilience', 'Emotional_Adjustment', 'Menstrual_Pain']].describe()
print(summary_stats)

# Correlation between variables
print("\nCorrelation Analysis:")
corr_resilience_pain, p_val_rp = pearsonr(data['Resilience'], data['Menstrual_Pain'])
corr_adjustment_pain, p_val_ap = pearsonr(data['Emotional_Adjustment'], data['Menstrual_Pain'])
print(f"Resilience and Menstrual Pain: r = {corr_resilience_pain:.2f}, p = {p_val_rp:.3f}")
print(f"Emotional Adjustment and Menstrual Pain: r = {corr_adjustment_pain:.2f}, p = {p_val_ap:.3f}")

# Simple linear regression for each predictor
print("\nSimple Linear Regression:")
# Emotional Adjustment
X_adj = sm.add_constant(data['Emotional_Adjustment'])
model_adj = sm.OLS(data['Menstrual_Pain'], X_adj).fit()
print("Emotional Adjustment Model:")
print(model_adj.summary())

# Resilience
X_res = sm.add_constant(data['Resilience'])
model_res = sm.OLS(data['Menstrual_Pain'], X_res).fit()
print("Resilience Model:")
print(model_res.summary())

# Multiple regression: Emotional Adjustment and Resilience combined
print("\nMultiple Regression:")
X_combined = sm.add_constant(data[['Resilience', 'Emotional_Adjustment']])
multi_model = sm.OLS(data['Menstrual_Pain'], X_combined).fit()
print("Combined Predictors Model:")
print(multi_model.summary())

# Mediation Analysis (conceptual implementation)
# Mediator model: Resilience → Emotional Adjustment
med_model = sm.OLS(data['Emotional_Adjustment'], sm.add_constant(data['Resilience'])).fit()

# Outcome model: Emotional Adjustment + Resilience → Menstrual Pain
outcome_model = sm.OLS(data['Menstrual_Pain'], sm.add_constant(data[['Emotional_Adjustment', 'Resilience']])).fit()

# Calculate and interpret mediation manually or use a library if needed
# Note: Specific steps or libraries for mediation analysis would depend on your chosen method.
print("\nMediation Analysis:")
print("Resilience's effect on Emotional Adjustment (mediator):")
print(med_model.summary())
print("Full model including Emotional Adjustment and Resilience:")
print(outcome_model.summary())
