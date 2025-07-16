import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import portfolio_functions  # Custom module with portfolio calculation functions

# Suppress scientific notation for numpy print output for better readability
np.set_printoptions(suppress=True)

# List of stock tickers in the portfolio
tickers = ["FRHC", "DLR", "PLTR", "XIACY", "COIN", "IBM"]

initial_price = 1000  # Starting price for the simulated portfolio value
T = 10  # Total time horizon in years for the simulation
N = 252  # Number of trading days per year (typical)
dt = T / N  # Time step in years (daily)
simulations = 500  # Number of Monte Carlo simulation paths

# Calculate expected portfolio return (mu) and portfolio volatility (sigma) for year 2021 using custom functions
mu = portfolio_functions.mu(tickers, 2021)
sigma = portfolio_functions.portfolio_volatility(tickers, 2021)

# Set random seed for reproducibility
np.random.seed(42)

# Initialize array to store simulated portfolio prices
# Rows = time steps, Columns = individual simulation paths
stock_paths = np.zeros((N, simulations))
stock_paths[0] = initial_price  # Set initial price for all simulations

# Create date index for the simulated time series with business day frequency
date_index = pd.date_range(start=pd.Timestamp.today().normalize(), periods=N, freq="B")

# Simulate the portfolio price paths using geometric Brownian motion model
for t in range(1, N):
    random_shock = np.random.normal(0, 1, simulations)  # Random shocks for each simulation at time t
    # GBM formula: S(t) = S(t-1) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    stock_paths[t] = stock_paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + np.sqrt(dt) * random_shock)

    # Add a positive jump of 100 every 30 days (to model a discrete event or dividend-like effect)
    if t % 30 == 0:
        stock_paths[t] += 100

# Convert simulated paths to a DataFrame for easier handling, using the date index
stock_paths = pd.DataFrame(stock_paths, index=date_index)

# Print the entire simulation results (all paths and dates)
print(stock_paths.to_string())

# Plot all simulation paths to visualize portfolio growth scenarios
plt.figure(figsize=(10, 6))
plt.plot(stock_paths)
plt.title("Simulated Portfolio Price Paths")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.show()

# Separate final simulated portfolio values into those above or equal to initial price, and those below
over = []
under = []

for final_price in stock_paths.iloc[-1]:
    if final_price >= initial_price:
        over.append(final_price)
    else:
        under.append(final_price)

# Print statistics about outcomes where portfolio value ended above or below initial price
print("Max portfolio value (over initial):", max(over))
print("Average portfolio value (over initial):", sum(over) / len(over))
print("Min portfolio value (over initial):", min(over))
print("Average portfolio value (under initial):", sum(under) / len(under))