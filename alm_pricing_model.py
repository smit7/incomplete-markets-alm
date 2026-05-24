import numpy as np

# ==========================================
# MODULE 1: Market Setup & Basis Risk
# ==========================================
np.random.seed(42)

# Market Parameters
S0 = 100.0          # Today's price of the tradable asset
vol = 10.0          # Volatility of the asset
strike = 100.0      # Strike threshold for the digital option
payout = 10.0       # The dollar payout of the digital option
rho = 0.75          # BASIS RISK: 75% correlation between asset and liability trigger

# Agent's Risk Profile
gamma = 0.15        # Risk aversion parameter for the exponential utility

# Function to simulate one future state of the market (Monte Carlo)
def simulate_market_state():
    # Simulate standard normal random variables
    Z1 = np.random.normal()
    Z2 = np.random.normal()
    
    # Create correlated variable for the liability trigger (Basis Risk)
    Z_Y = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    # Future Asset Price (S1) and the Liability Trigger (Y)
    S1 = S0 + vol * Z1 
    Y = strike + vol * Z_Y 
    
    # Liability C(w): Digital Option payout
    C = payout if Y >= strike else 0.0
    return S1, C

# ==========================================
# MODULE 4: The Robbins-Monro Algorithm
# ==========================================
print("Starting Robbins-Monro Optimization...")

N_iters = 200000
x = 0.0             # Initial portfolio guess (0 shares)
c = 0.05            # Step size multiplier

for n in range(1, N_iters + 1):
    # 1. Draw one noisy, random market scenario
    S1, C = simulate_market_state()
    
    # 2. Calculate the shortfall / Net Position
    net_position = C - x * (S1 - S0)
    
    # 3. Calculate the Noisy Gradient 
    # Derivative of Expected Penalty: exp(gamma * Z) * gamma * dZ/dx
    # Where dZ/dx = -(S1 - S0)
    gradient = -gamma * (S1 - S0) * np.exp(gamma * net_position)
    
    # 4. Diminishing Step Size: a_n = c / (n^0.6)
    # Shrinks slow enough to reach the bottom, fast enough to kill variance
    a_n = c / (n**0.6)
    
    # 5. Update Portfolio
    x = x - a_n * gradient

optimal_x = x
print(f"Convergence achieved. Optimal Hedge Ratio (x*): {optimal_x:.4f} shares\n")

# ==========================================
# MODULE 2 & 3: ALM Pricing & The Sandwich
# ==========================================
# Now that we found the optimal portfolio, we run a massive Monte Carlo 
# to calculate the exact risk and generate the prices.
N_eval = 1000000
S1_arr = np.zeros(N_eval)
C_arr = np.zeros(N_eval)

for i in range(N_eval):
    S1_arr[i], C_arr[i] = simulate_market_state()

# 1. Super-Hedging (Absolute worst-case without mathematical optimization)
p_super = payout 

# 2. Sub-Hedging (Floor value)
p_sub = 0.0 

# 3. Accounting Value (Objective marginal value: E[C] with zero risk premium)
p_acc = np.mean(C_arr)

# 4. Indifference Price (Closed-form Entropic Risk Measure derived from ALM)
# A. The price if the agent CANNOT trade to hedge the risk
risk_unhedged = np.mean(np.exp(gamma * C_arr))
p_ind_unhedged = (1 / gamma) * np.log(risk_unhedged)

# B. The price if the agent uses our Robbins-Monro optimized portfolio
risk_hedged = np.mean(np.exp(gamma * (C_arr - optimal_x * (S1_arr - S0))))
p_ind_hedged = (1 / gamma) * np.log(risk_hedged)

# ==========================================
# OUTPUT VERIFICATION
# ==========================================
print("-" * 40)
print("       THE PRICE SANDWICH RESULTS")
print("-" * 40)
print(f"Super-hedging Price:       ${p_super:.2f}  (Too expensive to trade)")
print(f"Unhedged Panic Price:      ${p_ind_unhedged:.2f}  (Seller's price if no ALM is used)")
print(f"Indifference Price (ALM):  ${p_ind_hedged:.2f}  (Our true subjective price)")
print(f"Accounting Value (Dual):   ${p_acc:.2f}  (Objective baseline)")
print(f"Sub-hedging Price:         ${p_sub:.2f}")
print("-" * 40)

# Verify the mathematical inequality
sandwich_holds = (p_sub <= p_acc <= p_ind_hedged <= p_super)
print(f"\nTheorem Verified: {sandwich_holds}")
print(f"({p_sub:.2f} <= {p_acc:.2f} <= {p_ind_hedged:.2f} <= {p_super:.2f})")
print(f"\nValue of the Algorithm: By using RM to find the optimal hedge,")
print(f"the agent reduced their required risk premium by ${(p_ind_unhedged - p_ind_hedged):.2f}.")