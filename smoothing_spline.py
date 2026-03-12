
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline, BSpline
from scipy.optimize import minimize_scalar

# Load Wage data
try:
    df = pd.read_csv('../ISLR-Datasets/Bikeshare.csv')
except FileNotFoundError:
    # Generate synthetic data if file not found (fallback)
    np.random.seed(1)
    x = np.linspace(18, 80, 100)
    y = 100 + 0.5 * x - 0.005 * (x-40)**2 + np.random.normal(0, 10, 100)
    df = pd.DataFrame({'age': x, 'wage': y})

# Extract unique X values and compute mean Y for duplicates (for efficiency)
# Smoothing splines are naturally defined on unique knots
# If we have multiple Y for same X, we can use the mean Y and weights = count
# Convert temperature from normalized scale to Celsius (assuming 0-1 scale maps to -8 to 39°C)
df['temp_celsius'] = df['temp'] * (39 - (-8)) + (-8)
# Convert "feels like" temperature from normalized scale to Celsius (assuming 0-1 scale maps to -16 to 50°C)
df['atemp_celsius'] = df['atemp'] * (50 - (-16)) + (-16)

# Round temperature to 2 decimals to ensure unique values are well-separated for spline knots
df['temp_celsius'] = df['temp_celsius'].round(2)

# Aggregate data by day
df['day_key'] = df['day'].astype(str)
daily_data = df.groupby('day_key').agg({'temp_celsius': 'mean', 'bikers': 'sum'}).reset_index()

# Extract variables for OLS
X_daily = daily_data['temp_celsius'].values
y_daily = daily_data['bikers'].values
X_daily_np = np.array(X_daily)
y_daily_np = np.array(y_daily)

# Preprocess for Smoothing Spline: Sort by X and handle duplicates
# Smoothing splines require strictly increasing x. 
# We group by temp_celsius just in case of duplicates (though unlikely for float temp).
grouped_daily = daily_data.groupby('temp_celsius')['bikers'].agg(['mean', 'count', 'var'])
x_unique = grouped_daily.index.values
y_mean = grouped_daily['mean'].values
weights = grouped_daily['count'].values
# Sum of Squared Errors within each temperature group: sum((y_ij - y_mean_i)^2)
ss_within = grouped_daily['var'].fillna(0).values * (weights - 1)

# Function to compute LOOCV score for a given lambda
def compute_loocv(lam, x, y, w):
    # Fit smoothing spline
    # make_smoothing_spline returns a BSpline object
    spl = make_smoothing_spline(x, y, w=w, lam=lam)
    
    # Compute the hat matrix diagonal (leverage)
    # Since we have reduced to unique x (about 700 points), the O(N^2) diagonal extraction is fast enough.
    # We apply the smoother to unit vectors to extract the diagonal elements S_ii
    n = len(x)
    hat_diag = np.zeros(n)
    
    # Basis matrix approach or brute force unit vectors
    identity = np.eye(n)
    for i in range(n):
        # Apply smoother to unit vector e_i with same weights and lambda
        spl_i = make_smoothing_spline(x, identity[i], w=w, lam=lam)
        hat_diag[i] = spl_i(x[i])
        
    # Fitted values
    y_hat = spl(x)
    return hat_diag, y_hat

def objective(lam):
    # Avoid log(0) or negative
    if lam <= 1e-8: return 1e10
    
    hat_diag, y_hat_mean = compute_loocv(lam, x_unique, y_mean, weights)
    
    # Observations leverage
    h_obs = hat_diag / weights
    
    # Check for singularity (h=1) or invalid leverage (h<0)
    if np.any(h_obs >= 1.0 - 1e-5) or np.any(h_obs < -1e-5):
        return 1e10
        
    denom = (1 - h_obs)**2
    
    # Group contribution
    # RSS_CV based on formula (7.11) logic for grouped data
    numerator = ss_within + weights * (y_mean - y_hat_mean)**2
    
    loocv_score = np.sum(numerator / denom)
    return loocv_score

# Optimize lambda
# We search for lambda in a reasonable range. If lambda is very large, the fit becomes linear.
res = minimize_scalar(objective, bounds=(1e-15, 100), method='bounded')
optimal_lambda = res.x
print(f"Optimal Lambda: {optimal_lambda}")

# Get optimal fit details
opt_hat, opt_y = compute_loocv(optimal_lambda, x_unique, y_mean, weights)
df_opt = np.sum(opt_hat)
print(f"Effective Matrix Trace (df for unique): {df_opt}")

# Plot
x_grid = np.linspace(x_unique.min(), x_unique.max(), 300)
opt_spline = make_smoothing_spline(x_unique, y_mean, w=weights, lam=optimal_lambda)

plt.figure(figsize=(10, 6))
plt.scatter(daily_data['temp_celsius'], daily_data['bikers'], facecolors='none', edgecolors='grey', alpha=0.5, label='Daily Data')
plt.plot(x_grid, opt_spline(x_grid), 'b-', linewidth=2, label=f'Smoothing Spline (df={df_opt:.2f}, $\lambda$={optimal_lambda:.2g})')
plt.legend()
plt.title('Smoothing Spline on Bike Sharing Data')
plt.xlabel('Temperature (Celsius)')
plt.ylabel('Total Daily Bikers')
plt.savefig('smoothing_spline_bike.png')
print("Plot saved as smoothing_spline_bike.png")
