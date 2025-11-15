# XGBoost for Regression - Complete Guide for Beginners

## What is XGBoost Regression?

XGBoost Regression is an optimized implementation of gradient boosting for predicting continuous numbers. It is faster, more accurate, and more feature-rich than standard gradient boosting regression.

Remember:
- **Classification**: Predicts categories (Happy, Sad, Angry)
- **Regression**: Predicts numbers (25.5, 150.7, 99.2)

Examples where you would use XGBoost Regression:
- Predicting house prices
- Predicting a person's age from features
- Predicting temperature
- Predicting emotion intensity on a scale
- Predicting stock prices
- Predicting exam scores

## Why XGBoost is Better for Regression

### Regular Gradient Boosting Regression:
- Basic implementation
- Slower training
- No built-in regularization
- Manual missing value handling
- Limited optimization

### XGBoost Regression:
- Highly optimized (10x faster)
- Parallel processing (uses all CPU cores)
- Built-in L1 and L2 regularization
- Automatic missing value handling
- Advanced tree pruning
- Better accuracy
- More control over training

## Main Innovations in XGBoost

### 1. Regularized Objective Function

XGBoost adds penalties to prevent overfitting:

```
Objective = Loss + Regularization

Loss = How wrong predictions are
Regularization = Penalty for model complexity
```

Full formula:
```
Objective = Σ (y_i - ŷ_i)² + Σ Ω(tree)

Where:
Ω(tree) = γ × T + (λ/2) × Σ(w²) + α × Σ|w|

γ (gamma) = penalty for number of leaves
T = number of leaves
λ (lambda) = L2 regularization
α (alpha) = L1 regularization
w = leaf weights
```

### 2. Second-Order Taylor Approximation

XGBoost uses both first and second derivatives:

**Gradient (first derivative)**: Direction to move
**Hessian (second derivative)**: How fast to move

This is like using both speed and acceleration when driving - gives better control and faster convergence.

### 3. Parallel Tree Construction

- Each tree is built using all CPU cores
- Much faster training
- Better hardware utilization

### 4. Smart Handling of Missing Values

- Automatically learns best direction for missing values
- No need to impute or remove missing data
- Works efficiently with sparse data

### 5. Efficient Split Finding

- Uses approximate algorithm for large datasets
- Weighted quantile sketch
- Faster than exact methods

## How Does XGBoost Regression Work?

Let's understand step by step.

### Step 1: Initialize with Mean

Start with the average of all target values:

```
Initial prediction = mean(y_train)
```

Example:
If predicting ages [20, 30, 40, 50]:
```
Initial prediction = (20 + 30 + 40 + 50) / 4 = 35
```

### Step 2: Calculate Gradients and Hessians

For each training example, calculate:

**Gradient (first derivative)**:
```
g_i = ∂Loss/∂prediction = prediction - true_value
```

**Hessian (second derivative)**:
```
h_i = ∂²Loss/∂prediction² = 1 (for squared error)
```

Example:
```
True value = 25
Current prediction = 35
g = 35 - 25 = 10
h = 1
```

The gradient tells us we predicted too high by 10 units.

### Step 3: Build Tree Using Gradients and Hessians

XGBoost builds a tree to minimize:

```
Objective = Σ[g_i × prediction + (1/2) × h_i × prediction²] + Ω(tree)
```

For each potential split, it calculates gain:

```
Gain = (Left_G)² / (Left_H + λ) + (Right_G)² / (Right_H + λ) - (Parent_G)² / (Parent_H + λ) - γ
```

Where:
- Left_G = sum of gradients in left child
- Right_G = sum of gradients in right child
- Left_H = sum of hessians in left child
- Right_H = sum of hessians in right child
- Parent_G = sum of gradients before split
- Parent_H = sum of hessians before split
- λ (lambda) = L2 regularization parameter (prevents overfitting)
- γ (gamma) = complexity penalty (prevents too many splits)

The split with the highest positive gain is chosen. If all gains are negative, no split is made.

### Step 4: Calculate Leaf Weights

For each leaf in the tree:

```
Leaf weight = - (Sum of gradients) / (Sum of hessians + λ)
```

Example:
```
Sum of gradients in leaf = -20
Sum of hessians in leaf = 5
λ = 1

Leaf weight = -(-20) / (5 + 1) = 20 / 6 = 3.33
```

This weight is what the tree predicts for samples in this leaf.

### Step 5: Update Predictions

Add the tree's predictions to current predictions:

```
New prediction = Old prediction + (learning_rate × tree_prediction)
```

Example:
```
Old prediction = 35
Tree predicts = 3.33
Learning rate = 0.1

New prediction = 35 + (0.1 × 3.33) = 35 + 0.333 = 35.333
```

### Step 6: Repeat

Repeat steps 2-5 many times (100-1000 iterations). Each iteration:
- Calculates new gradients and hessians
- Builds a tree to correct remaining errors
- Updates predictions

### Step 7: Final Prediction

```
Final prediction = Initial + learning_rate × (Tree₁ + Tree₂ + ... + TreeN)
```

Example with 3 trees:
```
Initial = 35
Tree 1 predicts: 3.33
Tree 2 predicts: 1.50
Tree 3 predicts: 0.67
Learning rate = 0.1

Final = 35 + 0.1×3.33 + 0.1×1.50 + 0.1×0.67
Final = 35 + 0.333 + 0.150 + 0.067
Final = 35.55
```

## Mathematical Formulas Explained

### Formula 1: Initial Prediction

```
F₀(x) = mean(y)
```

Or alternatively:
```
F₀(x) = argmin_c Σ L(y_i, c)
```

This finds the constant that minimizes loss.

### Formula 2: Loss Function (Squared Error)

```
L(y, ŷ) = (y - ŷ)² / 2
```

We divide by 2 to simplify derivatives.

### Formula 3: Gradient

First derivative of loss:
```
g_i = ∂L/∂ŷ = ŷ - y
```

This tells us the direction and magnitude of error.

### Formula 4: Hessian

Second derivative of loss:
```
h_i = ∂²L/∂ŷ² = 1 (for squared error)
```

For other loss functions, this can be different.

### Formula 5: Second-Order Approximation

Taylor expansion to second order:
```
L(y, ŷ + Δ) ≈ L(y, ŷ) + g × Δ + (1/2) × h × Δ²
```

This approximates the loss for a small change Δ in prediction.

### Formula 6: Regularization Term

```
Ω(tree) = γ × T + (λ/2) × Σ(w_j²) + α × Σ|w_j|
```

Where:
- γ = penalty for each leaf (complexity penalty)
- T = number of leaves
- λ = L2 regularization (weight decay)
- α = L1 regularization (feature selection)
- w_j = weight of leaf j

### Formula 7: Optimal Leaf Weight

```
w_j* = - (Σ g_i) / (Σ h_i + λ)
```

Where sums are over all samples in leaf j.

This minimizes the objective function for that leaf.

### Formula 8: Optimal Objective Value for a Leaf

After finding optimal weight:
```
Obj* = - (1/2) × (Σ g_i)² / (Σ h_i + λ) + γ
```

### Formula 9: Split Gain

```
Gain = (G_L)² / (H_L + λ) + (G_R)² / (H_R + λ) - (G)² / (H + λ) - γ
```

This measures the improvement from splitting a node.
- If Gain > 0: Split improves the model
- If Gain ≤ 0: Don't split (pre-pruning)

### Formula 10: Model Update

```
F_m(x) = F_{m-1}(x) + η × f_m(x)
```

Where:
- η (eta) = learning rate
- f_m(x) = prediction from new tree

## Different Loss Functions

### 1. Squared Error (Default)

Formula:
```
L = (y - ŷ)² / 2
```

Gradient:
```
g = ŷ - y
```

Hessian:
```
h = 1
```

Use when: Standard regression, normal data

### 2. Squared Log Error

Formula:
```
L = (log(y+1) - log(ŷ+1))² / 2
```

Use when: Target has exponential distribution

### 3. Absolute Error (MAE)

Formula:
```
L = |y - ŷ|
```

Gradient:
```
g = sign(ŷ - y)
```

Use when: Data has outliers, want robust predictions

### 4. Huber Loss

Combines squared and absolute:
```
If |y - ŷ| ≤ δ:
    L = (y - ŷ)² / 2
Else:
    L = δ × |y - ŷ| - δ²/2
```

Use when: Want balance of squared and absolute

### 5. Quantile Loss

For predicting specific percentiles:
```
If y ≥ ŷ:
    L = α × (y - ŷ)
Else:
    L = (1 - α) × (ŷ - y)
```

Use when: Want to predict median or percentiles

## Python Libraries Used

### 1. XGBoost

Main library.

To install:
```
pip install xgboost
```

To import:
```
import xgboost as xgb
from xgboost import XGBRegressor
```

### 2. NumPy

For numerical operations.

To install:
```
pip install numpy
```

To import:
```
import numpy as np
```

### 3. Pandas

For data handling.

To install:
```
pip install pandas
```

To import:
```
import pandas as pd
```

### 4. Scikit-learn

For utilities and metrics.

To install:
```
pip install scikit-learn
```

To import:
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

### 5. Matplotlib

For visualizations.

To install:
```
pip install matplotlib
```

To import:
```
import matplotlib.pyplot as plt
```

## Complete Python Code Example

Here is a comprehensive example:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Step 1: Create or load data
from sklearn.datasets import make_regression

# Create sample regression data
X, y = make_regression(
    n_samples=1000,       # 1000 examples
    n_features=20,        # 20 features
    n_informative=15,     # 15 useful features
    noise=10,             # Add some noise
    random_state=42
)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Step 3: Create DMatrix (XGBoost's data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Step 4: Set parameters
params = {
    # Task parameters
    'objective': 'reg:squarederror',    # Regression with squared error
    'eval_metric': 'rmse',              # Evaluation metric
    
    # Tree parameters
    'max_depth': 6,                     # Maximum depth of trees
    'min_child_weight': 3,              # Minimum sum of weights in child
    'eta': 0.1,                         # Learning rate
    
    # Regularization parameters
    'gamma': 0.1,                       # Minimum loss reduction for split
    'subsample': 0.8,                   # Fraction of samples per tree
    'colsample_bytree': 0.8,            # Fraction of features per tree
    'lambda': 1.0,                      # L2 regularization
    'alpha': 0.1,                       # L1 regularization
    
    # System parameters
    'tree_method': 'hist',              # Fast histogram algorithm
    'random_state': 42,
    'verbosity': 1
}

# Step 5: Train the model
print("Training XGBoost Regressor...")
evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=200,                # Number of trees
    evals=evals,
    early_stopping_rounds=20,           # Stop if no improvement
    verbose_eval=50                     # Print every 50 rounds
)
print("Training complete!")

# Get best iteration
best_iteration = model.best_iteration
print(f"Best iteration: {best_iteration}")

# Step 6: Make predictions
y_pred_train = model.predict(dtrain)
y_pred_val = model.predict(dval)
y_pred_test = model.predict(dtest)

# Step 7: Evaluate the model

# Mean Squared Error
mse_train = mean_squared_error(y_train, y_pred_train)
mse_val = mean_squared_error(y_val, y_pred_val)
mse_test = mean_squared_error(y_test, y_pred_test)

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)
rmse_val = np.sqrt(mse_val)
rmse_test = np.sqrt(mse_test)

# Mean Absolute Error
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_val = mean_absolute_error(y_val, y_pred_val)
mae_test = mean_absolute_error(y_test, y_pred_test)

# R-squared Score
r2_train = r2_score(y_train, y_pred_train)
r2_val = r2_score(y_val, y_pred_val)
r2_test = r2_score(y_test, y_pred_test)

# Print results
print("\n=== Training Results ===")
print(f"MSE:  {mse_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"MAE:  {mae_train:.2f}")
print(f"R²:   {r2_train:.4f}")

print("\n=== Validation Results ===")
print(f"MSE:  {mse_val:.2f}")
print(f"RMSE: {rmse_val:.2f}")
print(f"MAE:  {mae_val:.2f}")
print(f"R²:   {r2_val:.4f}")

print("\n=== Testing Results ===")
print(f"MSE:  {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MAE:  {mae_test:.2f}")
print(f"R²:   {r2_test:.4f}")

# Step 8: Visualize predictions vs actual
plt.figure(figsize=(15, 5))

# Training data
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5, s=10)
plt.plot([y_train.min(), y_train.max()], 
         [y_train.min(), y_train.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Training (R² = {r2_train:.3f})')

# Validation data
plt.subplot(1, 3, 2)
plt.scatter(y_val, y_pred_val, alpha=0.5, s=10)
plt.plot([y_val.min(), y_val.max()], 
         [y_val.min(), y_val.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Validation (R² = {r2_val:.3f})')

# Testing data
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_test, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Testing (R² = {r2_test:.3f})')

plt.tight_layout()
plt.show()

# Step 9: Learning curves
results = model.evals_result()
epochs = len(results['train']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, results['train']['rmse'], label='Train')
plt.plot(x_axis, results['val']['rmse'], label='Validation')
plt.xlabel('Boosting Round')
plt.ylabel('RMSE')
plt.title('XGBoost Learning Curve')
plt.legend()
plt.axvline(x=best_iteration, color='r', linestyle='--', label='Best Iteration')
plt.show()

# Step 10: Feature importance
importance_dict = model.get_score(importance_type='gain')

# Sort by importance
importance_sorted = sorted(importance_dict.items(), 
                          key=lambda x: x[1], 
                          reverse=True)

print("\nTop 10 Feature Importances:")
for i, (feature, importance) in enumerate(importance_sorted[:10]):
    print(f"{i+1}. {feature}: {importance:.2f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
features = [item[0] for item in importance_sorted[:10]]
importances = [item[1] for item in importance_sorted[:10]]
plt.barh(features, importances)
plt.xlabel('Importance (Gain)')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

# Step 11: Residual plot
residuals = y_test - y_pred_test

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (should be random around 0)')
plt.show()

# Step 12: Save and load model
model.save_model('xgboost_regressor.json')
print("\nModel saved to xgboost_regressor.json")

# Load model
loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_regressor.json')

# Step 13: Predict for new example
new_example = X_test[0].reshape(1, -1)
dnew = xgb.DMatrix(new_example)
prediction = loaded_model.predict(dnew)

print(f"\nNew example prediction: {prediction[0]:.2f}")
print(f"Actual value: {y_test[0]:.2f}")
print(f"Error: {abs(prediction[0] - y_test[0]):.2f}")
```

## Using Scikit-learn API

XGBoost also provides scikit-learn compatible API:

```python
from xgboost import XGBRegressor

# Create regressor
xgb_reg = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,              # Use all CPU cores
    verbosity=1
)

# Train with early stopping
xgb_reg.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=20,
    verbose=True
)

# Predict
predictions = xgb_reg.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# Feature importance
importance = xgb_reg.feature_importances_
print("Feature importances:", importance)
```

## Understanding XGBoost Parameters

### Task Parameters:

1. **objective**: What to optimize
   - 'reg:squarederror': Squared error (default)
   - 'reg:squaredlogerror': Squared log error
   - 'reg:logistic': Logistic regression
   - 'reg:pseudohubererror': Pseudo-Huber loss

2. **eval_metric**: How to measure performance
   - 'rmse': Root mean squared error
   - 'mae': Mean absolute error
   - 'rmsle': Root mean squared log error
   - 'mape': Mean absolute percentage error

### Tree Parameters:

3. **max_depth**: Maximum depth of trees
   - Default: 6
   - Range: 3 to 10
   - Higher = more complex
   - Typical: 4 to 8

4. **min_child_weight**: Minimum sum of weights
   - Default: 1
   - Higher = more conservative
   - Prevents overfitting
   - Typical: 1 to 10

5. **eta** (or learning_rate): Learning rate
   - Default: 0.3
   - Range: 0.01 to 0.3
   - Lower = more accurate but slower
   - Typical: 0.01, 0.05, 0.1

### Regularization Parameters:

6. **gamma**: Minimum loss reduction
   - Default: 0
   - Range: 0 to 5
   - Higher = more conservative
   - Prevents overfitting

7. **subsample**: Fraction of samples per tree
   - Default: 1.0
   - Range: 0.5 to 1.0
   - Lower = less overfitting
   - Typical: 0.8

8. **colsample_bytree**: Fraction of features per tree
   - Default: 1.0
   - Range: 0.5 to 1.0
   - Lower = less overfitting
   - Typical: 0.8

9. **lambda** (or reg_lambda): L2 regularization
   - Default: 1
   - Higher = more regularization
   - Typical: 0.1 to 2

10. **alpha** (or reg_alpha): L1 regularization
    - Default: 0
    - Higher = more feature selection
    - Typical: 0 to 1

### System Parameters:

11. **tree_method**: Algorithm
    - 'auto': Automatic
    - 'exact': Exact greedy
    - 'approx': Approximate
    - 'hist': Histogram (fastest, recommended)

12. **n_jobs**: Number of threads
    - -1: Use all CPU cores
    - Positive: Use that many cores

13. **random_state**: Seed for reproducibility

## When to Use XGBoost Regression

### Perfect for:
- Predicting continuous values
- Tabular/structured data
- Medium to large datasets
- When accuracy is critical
- Production systems
- Competitions
- Complex non-linear relationships

### Not ideal for:
- Very small datasets (<100 samples)
- Simple linear relationships (use linear regression)
- Image or text data (use deep learning)
- When simplicity is more important
- Real-time with strict latency

## Advantages of XGBoost Regression

1. **Highest accuracy**: Best for tabular data
2. **Very fast**: 10x faster than standard gradient boosting
3. **Parallel processing**: Uses all CPU cores
4. **Built-in regularization**: L1, L2, gamma
5. **Handles missing values**: Automatically
6. **Feature importance**: Multiple types
7. **Cross-validation**: Built-in
8. **Early stopping**: Prevents overfitting
9. **Sparse data**: Efficient
10. **Production ready**: Easy to deploy
11. **Memory efficient**: Optimized algorithms
12. **Flexible**: Many loss functions

## Disadvantages of XGBoost Regression

1. **Complex**: Many hyperparameters
2. **Memory intensive**: Can use lots of RAM
3. **Less interpretable**: Black box
4. **Installation**: Can be tricky
5. **Overkill**: For simple linear problems
6. **Tuning required**: For optimal performance

## Tips for Better Results

1. **Start with defaults**: Use recommended parameters
2. **Lower learning rate**: 0.01-0.1 is best
3. **Use early stopping**: Always
4. **Cross-validation**: Built-in CV
5. **Feature engineering**: Create meaningful features
6. **Monitor metrics**: Plot learning curves
7. **Regularization**: Tune gamma, lambda, alpha
8. **Handle outliers**: Consider robust loss
9. **Feature selection**: Use importance
10. **Ensemble**: Combine multiple models

## Hyperparameter Tuning Strategy

### Step 1: Fix eta = 0.1, num_boost_round = 100

### Step 2: Tune max_depth and min_child_weight
- max_depth: 3, 5, 7, 9
- min_child_weight: 1, 3, 5, 7

### Step 3: Tune gamma
- Try: 0, 0.1, 0.2, 0.5, 1.0

### Step 4: Tune subsample and colsample_bytree
- Both: 0.6, 0.7, 0.8, 0.9, 1.0

### Step 5: Tune regularization
- lambda: 0.1, 0.5, 1.0, 2.0, 5.0
- alpha: 0, 0.1, 0.5, 1.0

### Step 6: Lower learning rate
- eta: 0.01, 0.05, 0.1
- Increase num_boost_round accordingly

## Common Mistakes to Avoid

1. **Too high learning rate**: Use 0.01-0.1
2. **Too many trees**: Use early stopping
3. **Not using regularization**: Always tune
4. **Default tree_method**: Use 'hist'
5. **Not using all cores**: Set n_jobs=-1
6. **Not validating**: Always use validation set
7. **Ignoring outliers**: Consider robust loss
8. **Not saving model**: Save trained models
9. **Overfitting**: Monitor validation metrics
10. **Wrong objective**: Match your problem

## Practical Example: Age Prediction

```python
# Age prediction from speech and facial features
# Features: 70 total (20 speech + 50 facial)
# Target: Age (18 to 80 years)

import xgboost as xgb

# Create DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Parameters optimized for age prediction
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 5,
    'min_child_weight': 3,
    'eta': 0.1,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.1,
    'tree_method': 'hist',
    'random_state': 42
}

# Train
model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=30,
    verbose_eval=50
)

# Predict age for new person
def predict_age(audio_file, image_file):
    # Extract features
    features = extract_features(audio_file, image_file)
    
    # Create DMatrix
    dfeatures = xgb.DMatrix(features.reshape(1, -1))
    
    # Predict
    predicted_age = model.predict(dfeatures)[0]
    
    print(f"Predicted age: {predicted_age:.1f} years")
    
    # Prediction interval (train separate quantile models)
    # Lower bound (10th percentile)
    params_lower = params.copy()
    params_lower['objective'] = 'reg:quantileerror'
    params_lower['quantile_alpha'] = 0.1
    
    model_lower = xgb.train(params_lower, dtrain, num_boost_round=200)
    lower = model_lower.predict(dfeatures)[0]
    
    # Upper bound (90th percentile)
    params_upper = params.copy()
    params_upper['objective'] = 'reg:quantileerror'
    params_upper['quantile_alpha'] = 0.9
    
    model_upper = xgb.train(params_upper, dtrain, num_boost_round=200)
    upper = model_upper.predict(dfeatures)[0]
    
    print(f"80% prediction interval: [{lower:.1f}, {upper:.1f}]")
    
    return predicted_age

# Use it
age = predict_age('audio.wav', 'face.jpg')
```

## Evaluation Metrics Explained

### 1. RMSE (Root Mean Squared Error)
- Square root of average squared errors
- Same units as target
- Lower is better
- Penalizes large errors

### 2. MAE (Mean Absolute Error)
- Average of absolute errors
- Same units as target
- Lower is better
- Less sensitive to outliers

### 3. R² (R-squared)
- Proportion of variance explained
- Range: -∞ to 1
- 1 = perfect, 0 = as good as mean
- Higher is better

### 4. MAPE (Mean Absolute Percentage Error)
- Average percentage error
- In percentage units
- Lower is better
- Easy to interpret

## Summary

XGBoost Regression is:
- The most powerful regression algorithm for tabular data
- Optimized for speed and accuracy
- Includes advanced regularization
- Handles missing values automatically
- Production-ready

Key features:
- Second-order approximation
- Parallel tree construction
- Built-in L1/L2 regularization
- Automatic missing value handling
- Multiple importance types
- Early stopping
- Cross-validation

For practical applications:
- Expected R² > 0.90 with tuning
- Fast training even on large datasets
- Handles both speech and facial features
- Provides interpretable importance
- Easy to deploy

XGBoost Regression is the recommended algorithm when you need the best possible accuracy for predicting continuous values from structured data.
