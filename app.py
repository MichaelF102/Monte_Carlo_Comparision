import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import qmc, norm, multivariate_normal
import time

# Attempt to import Numba for Section 10 (Hardware Acceleration)
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if Numba is not installed
    def jit(nopython=True):
        def decorator(func):
            return func
        return decorator

# Attempt to import yfinance for Section 11 (Real-World Data)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Set Plotting Style
plt.style.use('seaborn-v0_8-darkgrid')

# Page Config
st.set_page_config(
    page_title="Monte Carlo Simulation Compendium",
    page_icon="🎲",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def black_scholes_call(S, K, T, r, sigma):
    """Analytical solution for European Call."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def generate_gbm(S0, mu, sigma, T, steps, n_sims, use_sobol=False):
    """Generates Geometric Brownian Motion paths."""
    dt = T / steps
    if use_sobol:
        # Quasi-Random numbers (Sobol)
        sampler = qmc.Sobol(d=steps, scramble=True)
        Z = sampler.random(n=n_sims)
        Z = norm.ppf(Z) # Inverse transform to get Gaussian
    else:
        # Pseudo-Random numbers
        Z = np.random.normal(0, 1, (n_sims, steps))
    
    # Construct paths
    paths = np.zeros((n_sims, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])
    return paths, Z

# Helper for Heston Model (Section 9)
def generate_heston_paths(S0, v0, rho, kappa, theta, xi, T, M, N):
    """
    Generates Heston Model paths.
    S0: Spot Price, v0: Initial Volatility
    rho: Correlation, kappa: Mean reversion speed
    theta: Long run var, xi: Vol of Vol
    """
    dt = T / M
    
    # Generate Correlated Brownian Motions
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    
    # Shape: (N, M, 2)
    Z = np.random.multivariate_normal(mean, cov, (N, M))
    Z_S = Z[:, :, 0]
    Z_v = Z[:, :, 1]
    
    S = np.zeros((N, M + 1))
    v = np.zeros((N, M + 1))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    for t in range(M):
        # Euler-Maruyama for Volatility (with max(0, v) to prevent negative vol)
        v_t = v[:, t]
        v_sqrt = np.sqrt(np.maximum(v_t, 0))
        
        dv = kappa * (theta - v_t) * dt + xi * v_sqrt * np.sqrt(dt) * Z_v[:, t]
        v[:, t+1] = np.maximum(v_t + dv, 0) # Reflection or Truncation
        
        # Euler for Price
        dS = S[:, t] * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z_S[:, t]
        S[:, t+1] = S[:, t] + dS # Drifts assumed 0 for simplicity or handled elsewhere
        
    return S, v

# Helpers for Acceleration (Section 10)
def python_mc_pi(n):
    count = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            count += 1
    return 4.0 * count / n

def numpy_mc_pi(n):
    x = np.random.random(n)
    y = np.random.random(n)
    return 4.0 * np.sum(x**2 + y**2 <= 1.0) / n

@jit(nopython=True)
def numba_mc_pi(n):
    count = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            count += 1
    return 4.0 * count / n

# -----------------------------------------------------------------------------
# SECTION 1: STANDARD MONTE CARLO
# -----------------------------------------------------------------------------
def section_monte_carlo():
    st.header("1. Standard Monte Carlo (Crude MC)")
    st.markdown("""
    **The Baseline:** Uses pseudo-random numbers to simulate thousands of possible futures.
    We simulate stock price paths using **Geometric Brownian Motion (GBM)** to price a **European Call Option**.
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parameters")
        S0 = st.number_input("Spot Price ($)", 100.0, value=100.0, key="mc_s0")
        K = st.number_input("Strike Price ($)", 100.0, value=105.0, key="mc_k")
        T = st.number_input("Time to Maturity (Years)", 0.1, 5.0, 1.0, key="mc_t")
        sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2, key="mc_sigma")
        n_sims = st.slider("Simulations (N)", 100, 10000, 1000, key="mc_n")
    
    if st.button("Run Crude MC", key="btn_mc"):
        start = time.time()
        paths, _ = generate_gbm(S0, 0.05, sigma, T, 252, n_sims) # r=0.05 assumed
        terminal_prices = paths[:, -1]
        payoffs = np.maximum(terminal_prices - K, 0)
        price_est = np.exp(-0.05 * T) * np.mean(payoffs)
        duration = time.time() - start
        
        true_price = black_scholes_call(S0, K, T, 0.05, sigma)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MC Price", f"${price_est:.2f}")
        m2.metric("True Price (BS)", f"${true_price:.2f}")
        m3.metric("Error", f"${abs(price_est - true_price):.4f}")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(paths[:50, :].T, alpha=0.3, lw=0.5)
        ax[0].set_title(f"First 50 Sim Paths")
        ax[0].set_xlabel("Time Steps")
        ax[0].set_ylabel("Price")
        
        cumulative_avg = np.cumsum(payoffs * np.exp(-0.05 * T)) / np.arange(1, n_sims + 1)
        ax[1].plot(cumulative_avg, label="MC Estimate")
        ax[1].axhline(true_price, color='r', linestyle='--', label="True Price")
        ax[1].set_title("Convergence of Price")
        ax[1].set_xlabel("Iterations")
        ax[1].legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 2: QUASI MONTE CARLO
# -----------------------------------------------------------------------------
def section_quasi_mc():
    st.header("2. Quasi-Monte Carlo (QMC)")
    st.markdown("""
    **The Upgrade:** Uses **Low-Discrepancy Sequences (Sobol)** instead of random numbers. 
    Sobol sequences fill space more evenly, avoiding "clumping," leading to much faster convergence.
    """)
    n_points = st.slider("Number of Points", 100, 2000, 500, key="qmc_n")
    
    col1, col2 = st.columns(2)
    random_pts = np.random.rand(n_points, 2)
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_pts = sampler.random(n_points)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(random_pts[:,0], random_pts[:,1], alpha=0.6, c='blue', s=15)
    ax[0].set_title("Pseudo-Random (Clumpy)")
    ax[0].set_aspect('equal')
    ax[1].scatter(sobol_pts[:,0], sobol_pts[:,1], alpha=0.6, c='orange', s=15)
    ax[1].set_title("Sobol Sequence (Uniform)")
    ax[1].set_aspect('equal')
    st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 3: LEAST SQUARES MONTE CARLO (LSMC)
# -----------------------------------------------------------------------------
def section_lsmc():
    st.header("3. Least Squares Monte Carlo (LSMC)")
    st.markdown("""
    **The Problem:** Standard MC goes forward in time. American Options can be exercised *early*.
    **The Solution:** Simulate forward, then step **backwards** using **Regression**.
    """)
    S0 = 100; K = 100; r = 0.05; sigma = 0.2; T = 1.0
    N = st.slider("Simulations", 100, 5000, 1000, key="lsmc_n")
    M = st.slider("Time Steps", 10, 100, 50, key="lsmc_m")
    
    if st.button("Run LSMC"):
        dt = T / M
        df = np.exp(-r * dt)
        paths, _ = generate_gbm(S0, r, sigma, T, M, N)
        payoff = np.maximum(K - paths, 0)
        V = np.zeros_like(payoff)
        V[:, -1] = payoff[:, -1]
        exercise_decisions = np.zeros_like(paths)
        
        for t in range(M - 1, 0, -1):
            itm_mask = payoff[:, t] > 0
            if np.sum(itm_mask) > 0:
                X = paths[itm_mask, t]
                Y = V[itm_mask, t+1] * df
                coeffs = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(coeffs, X)
                exercise = payoff[itm_mask, t]
                do_exercise = exercise > continuation_value
                full_indices = np.where(itm_mask)[0]
                V[:, t] = V[:, t+1] * df
                V[full_indices[do_exercise], t] = exercise[do_exercise]
                exercise_decisions[full_indices[do_exercise], t] = 1
            else:
                 V[:, t] = V[:, t+1] * df
        price = np.mean(V[:, 1] * df)
        st.success(f"American Put Price: ${price:.4f}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        subset = 100
        for i in range(subset):
            ax.plot(paths[i, :], color='gray', alpha=0.1)
        ex_idx = np.where(exercise_decisions[:subset, :] == 1)
        if len(ex_idx[0]) > 0:
            ax.scatter(ex_idx[1], paths[ex_idx[0], ex_idx[1]], color='red', s=20, zorder=3, label="Early Exercise")
        ax.set_title("Stock Paths & Early Exercise Events (Red Dots)")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 4: MARKOV CHAIN MONTE CARLO (MCMC)
# -----------------------------------------------------------------------------
def section_mcmc():
    st.header("4. Markov Chain Monte Carlo (MCMC)")
    st.markdown("**The Algorithm:** Metropolis-Hastings to recover unknown parameters.")
    true_mu = st.number_input("True Mean", -10.0, 10.0, 5.0)
    data = np.random.normal(true_mu, 2.0, 100)
    st.scatter_chart(data)
    
    if st.button("Run MCMC Sampler"):
        mu_current = 0.0
        samples = []
        for i in range(5000):
            mu_proposal = mu_current + np.random.normal(0, 0.5)
            # Simplified Likelihood Ratio
            ll_curr = np.sum(norm.logpdf(data, mu_current, 2.0))
            ll_prop = np.sum(norm.logpdf(data, mu_proposal, 2.0))
            if np.log(np.random.rand()) < (ll_prop - ll_curr):
                mu_current = mu_proposal
            samples.append(mu_current)
            
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        ax[0].plot(samples, lw=0.5)
        ax[0].set_title("Trace Plot")
        sns.histplot(samples[1000:], kde=True, ax=ax[1], color='green')
        ax[1].set_title("Posterior Distribution")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 5: SEQUENTIAL MONTE CARLO (SMC)
# -----------------------------------------------------------------------------
def section_smc():
    st.header("5. Sequential Monte Carlo (Particle Filters)")
    st.markdown("Tracking a robot using noisy measurements.")
    if st.button("Run Particle Filter"):
        T_steps = 50; n_particles = 100
        true_position = np.zeros(T_steps)
        measurements = np.zeros(T_steps)
        curr_pos = 0
        for t in range(T_steps):
            curr_pos += 1.0 + np.random.normal(0, 0.5)
            true_position[t] = curr_pos
            measurements[t] = curr_pos + np.random.normal(0, 2.0)
            
        particles = np.zeros(n_particles)
        weights = np.ones(n_particles) / n_particles
        estimates = []
        
        for t in range(T_steps):
            particles += 1.0 + np.random.normal(0, 0.5, n_particles)
            dist = measurements[t] - particles
            weights = weights * np.exp(-0.5 * (dist**2) / 4.0)
            weights += 1.e-300
            weights /= np.sum(weights)
            estimates.append(np.sum(particles * weights))
            indices = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
            particles = particles[indices]
            weights.fill(1.0/n_particles)
            
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(true_position, 'k-', label="True")
        ax.plot(measurements, 'gx', alpha=0.5, label="Noisy GPS")
        ax.plot(estimates, 'r--', label="SMC Est")
        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 6: QUANTUM MONTE CARLO (VMC)
# -----------------------------------------------------------------------------
def section_qmc():
    st.header("6. Quantum Monte Carlo (VMC)")
    st.markdown("Finding Ground State Energy of Harmonic Oscillator.")
    alpha = st.slider("Alpha", 0.1, 2.0, 0.8, 0.1)
    if st.button("Run VMC"):
        x = 0.0; positions = []; energies = []
        for i in range(5000):
            x_new = x + np.random.uniform(-0.5, 0.5)
            prob = np.exp(-2 * alpha * (x_new**2 - x**2))
            if np.random.rand() < prob: x = x_new
            if i > 500:
                positions.append(x)
                energies.append(alpha + (x**2) * (0.5 - 2 * alpha**2))
        
        st.metric("Estimated Energy", f"{np.mean(energies):.4f}")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(positions, kde=True, ax=ax, color='purple')
        ax.set_title("Wavefunction Density")
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 7: COMPARISON
# -----------------------------------------------------------------------------
def section_comparison():
    st.header("7. Comparison & Summary")
    comparison_data = {
        "Method": ["Standard MC", "Quasi-MC", "LSMC", "MCMC", "SMC", "QMC"],
        "Convergence": ["$O(1/\sqrt{N})$", "$O(1/N)$", "Slow", "Variable", "$O(1/\sqrt{N})$", "$O(1/\sqrt{N})$"]
    }
    st.table(pd.DataFrame(comparison_data))

# -----------------------------------------------------------------------------
# SECTION 8: VARIANCE REDUCTION (ANTITHETIC VARIATES)
# -----------------------------------------------------------------------------
def section_variance_reduction():
    st.header("8. Variance Reduction: Antithetic Variates")
    st.markdown("""
    **The Concept:** Instead of generating $N$ independent paths, we generate $N/2$ pairs.
    For every random draw $Z$, we also use $-Z$.
    
    * Path A driven by $Z$
    * Path B driven by $-Z$
    
    Since $Z$ and $-Z$ are perfectly negatively correlated, the average of the two paths has significantly lower variance than two independent paths.
    """)
    
    S0 = 100; K = 100; T = 1.0; r = 0.05; sigma = 0.2
    n_sims = st.slider("Simulations (N)", 100, 5000, 1000, key="vr_n")
    
    if st.button("Compare Standard vs. Antithetic"):
        # 1. Standard MC
        # Generate N independent paths
        paths_std, _ = generate_gbm(S0, r, sigma, T, 1, n_sims)
        payoffs_std = np.maximum(paths_std[:,-1] - K, 0) * np.exp(-r*T)
        
        # 2. Antithetic MC
        # Generate N/2 random draws
        n_half = n_sims // 2
        Z = np.random.normal(0, 1, (n_half, 1))
        
        # Path 1 using Z
        ST_1 = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoff_1 = np.maximum(ST_1 - K, 0)
        
        # Path 2 using -Z
        ST_2 = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*(-Z))
        payoff_2 = np.maximum(ST_2 - K, 0)
        
        # Average the pair
        payoffs_pair = 0.5 * (payoff_1 + payoff_2) * np.exp(-r*T)
        
        # Convergence calc
        conv_std = np.cumsum(payoffs_std) / np.arange(1, n_sims + 1)
        conv_anti = np.cumsum(payoffs_pair) / np.arange(1, n_half + 1)
        
        true_price = black_scholes_call(S0, K, T, r, sigma)
        
        # Metrics
        col1, col2 = st.columns(2)
        var_std = np.var(payoffs_std)
        var_anti = np.var(payoffs_pair) # Variance of the *pair* average
        
        col1.metric("Standard Variance", f"{var_std:.4f}")
        col2.metric("Antithetic Variance", f"{var_anti:.4f}", delta=f"-{(1 - var_anti/var_std)*100:.1f}% reduction")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(conv_std, label="Standard MC", alpha=0.6)
        # Scale x-axis for antithetic to match N samples (since we did N/2 pairs)
        ax.plot(np.linspace(0, n_sims, n_half), conv_anti, label="Antithetic MC", color='green')
        ax.axhline(true_price, color='r', linestyle='--', label="True Price")
        ax.set_title("Convergence Speed Comparison")
        ax.set_xlabel("Iterations")
        ax.legend()
        st.pyplot(fig)

# -----------------------------------------------------------------------------
# SECTION 9: ADVANCED STOCHASTIC PROCESSES (HESTON)
# -----------------------------------------------------------------------------
def section_heston():
    st.header("9. Heston Stochastic Volatility Model")
    st.markdown("""
    **The Reality:** In real markets, volatility is NOT constant (as assumed in Black-Scholes).
    The **Heston Model** assumes volatility itself follows a random process (CIR process) and correlates with the stock price.
    
    $$dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S$$
    $$dv_t = \kappa(\\theta - v_t)dt + \\xi \sqrt{v_t} dW_t^v$$
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        rho = st.slider("Correlation (Price vs Vol)", -1.0, 1.0, -0.7, help="Negative correlation creates the 'leverage effect'.")
        xi = st.slider("Vol of Vol (ξ)", 0.1, 1.0, 0.3, help="How shaky is the volatility?")
    with col2:
        kappa = st.number_input("Mean Reversion Speed (κ)", 0.1, 10.0, 2.0)
        theta = st.number_input("Long Term Var (θ)", 0.01, 0.5, 0.04)
        
    if st.button("Simulate Heston Paths"):
        S, v = generate_heston_paths(S0=100, v0=0.04, rho=rho, kappa=kappa, theta=theta, xi=xi, T=1.0, M=252, N=10)
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Price Paths
        ax[0].plot(S.T, alpha=0.4)
        ax[0].set_title("Stock Price Paths ($S_t$)")
        ax[0].set_ylabel("Price")
        
        # Volatility Paths
        ax[1].plot(np.sqrt(v.T), alpha=0.4, color='orange') # Plotting sqrt(v) which is volatility
        ax[1].set_title("Stochastic Volatility Paths ($\sqrt{v_t}$)")
        ax[1].set_ylabel("Volatility")
        ax[1].set_xlabel("Time Steps")
        
        st.pyplot(fig)
        st.info("Notice: When Price drops (top), Volatility often spikes (bottom) due to negative correlation.")

# -----------------------------------------------------------------------------
# SECTION 10: HARDWARE ACCELERATION (NUMBA)
# -----------------------------------------------------------------------------
def section_acceleration():
    st.header("10. Hardware Acceleration Benchmark")
    st.markdown("""
    Monte Carlo is computationally expensive. Python loops are slow.
    We compare:
    1.  **Pure Python:** Standard for-loops (Slowest).
    2.  **NumPy Vectorization:** Array operations (Fast).
    3.  **Numba (JIT):** Just-In-Time compilation to Machine Code (Fastest).
    """)
    
    if not HAS_NUMBA:
        st.warning("⚠️ Numba is not installed in this environment. The JIT test will be a simulation.")
    
    n_sims = st.selectbox("Simulations (N)", [10000, 100000, 1000000, 5000000])
    
    if st.button("Run Benchmark"):
        results = {}
        
        # 1. Pure Python (Limit to 100k to avoid timeout)
        if n_sims <= 100000:
            start = time.time()
            python_mc_pi(n_sims)
            results["Pure Python"] = time.time() - start
        else:
            results["Pure Python"] = np.nan # Too slow
            
        # 2. NumPy
        start = time.time()
        numpy_mc_pi(n_sims)
        results["NumPy Vectorized"] = time.time() - start
        
        # 3. Numba
        # Run once to compile (warmup) then measure
        numba_mc_pi(10) 
        start = time.time()
        numba_mc_pi(n_sims)
        results["Numba JIT"] = time.time() - start
        
        # Visualization
        st.subheader(f"Execution Time for N={n_sims:,}")
        
        times = list(results.values())
        labels = list(results.keys())
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, times, color=['red', 'blue', 'green'])
        ax.set_ylabel("Time (Seconds)")
        ax.set_title("Lower is Better")
        
        # Annotate
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}s', ha='center', va='bottom')
            else:
                 ax.text(bar.get_x() + bar.get_width()/2., 0,
                        'Timed Out', ha='center', va='bottom', color='red')
        
        st.pyplot(fig)
        
        if not np.isnan(results["Pure Python"]) and results["Numba JIT"] > 0:
            speedup = results["Pure Python"] / results["Numba JIT"]
            st.success(f"🚀 Numba was {speedup:.1f}x faster than Pure Python!")

# -----------------------------------------------------------------------------
# SECTION 11: REAL-WORLD DATA (YAHOO FINANCE)
# -----------------------------------------------------------------------------
def section_real_world():
    st.header("11. Real-World Data (Yahoo Finance)")
    st.markdown("""
    **From Theory to Practice:** Connect to Yahoo Finance to fetch real historical data, estimate actual volatility, and run a simulation.
    """)
    
    if not HAS_YFINANCE:
        st.error("⚠️ `yfinance` library is not installed. Please install it to use this module.")
        st.code("pip install yfinance")
        return

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL")
    with col2:
        forecast_days = st.slider("Days to Forecast", 10, 90, 30)

    if st.button("Fetch Data & Simulate"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Get 1 year of data
                data = yf.download(ticker, period="1y", progress=False)
                
                if data.empty:
                    st.error("No data found. Check ticker symbol.")
                    return
                
                # Check for MultiIndex columns (yfinance > 0.2) and flatten if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # Calculate Log Returns
                data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
                
                # Estimate Volatility (Annualized)
                sigma = data['Log Returns'].std() * np.sqrt(252)
                S0 = float(data['Close'].iloc[-1])
                last_date = data.index[-1]
                
                st.success(f"Data Fetched! Last Price: ${S0:.2f}, Annual Volatility: {sigma*100:.1f}%")
                
                # Run Simulation
                T_years = forecast_days / 252.0
                n_sims = 1000
                paths, _ = generate_gbm(S0, 0.05, sigma, T_years, forecast_days, n_sims)
                
                # Create Forecast Dates
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
                
                # Visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot Historical (Last 90 days for clarity)
                hist_data = data['Close'].tail(90)
                ax.plot(hist_data.index, hist_data.values, 'k-', lw=2, label="Historical Data")
                
                # Plot Simulations
                # We need to map integer steps to dates for the plot
                # paths shape is (1000, 31) -> 0 to 30. Index 0 is S0.
                # We only plot the future part (1 to 30)
                for i in range(min(100, n_sims)): # Plot 100 paths max
                    ax.plot(future_dates, paths[i, 1:], color='blue', alpha=0.05)
                    
                # Plot Mean Path
                mean_path = np.mean(paths, axis=0)
                ax.plot(future_dates, mean_path[1:], 'r--', lw=2, label="Mean Forecast")
                
                ax.set_title(f"Monte Carlo Forecast for {ticker} ({forecast_days} Days)")
                ax.set_ylabel("Price ($)")
                ax.legend()
                
                st.pyplot(fig)
                
                # Cone of Uncertainty
                final_prices = paths[:, -1]
                lower_bound = np.percentile(final_prices, 5)
                upper_bound = np.percentile(final_prices, 95)
                
                st.metric("95% Confidence Interval", f"${lower_bound:.2f} - ${upper_bound:.2f}")
                
            except Exception as e:
                st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# MAIN APP LOGIC
# -----------------------------------------------------------------------------

def main():
    st.sidebar.title("Monte Carlo Lab 🧪")
    
    options = [
        "Standard Monte Carlo", 
        "Quasi Monte Carlo", 
        "Least Squares MC (American)", 
        "Markov Chain MC (Bayesian)",
        "Sequential MC (Particle Filter)",
        "Quantum MC (Variational)",
        "Variance Reduction (Antithetic)",
        "Heston Model (Stochastic Vol)",
        "Hardware Acceleration (JIT)",
        "Real-World Data (Yahoo Finance)",
        "Comparison & Summary"
    ]
    
    selection = st.sidebar.radio("Select Method", options)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Select a module to view the simulation logic and visualization.")
    
    if selection == "Standard Monte Carlo":
        section_monte_carlo()
    elif selection == "Quasi Monte Carlo":
        section_quasi_mc()
    elif selection == "Least Squares MC (American)":
        section_lsmc()
    elif selection == "Markov Chain MC (Bayesian)":
        section_mcmc()
    elif selection == "Sequential MC (Particle Filter)":
        section_smc()
    elif selection == "Quantum MC (Variational)":
        section_qmc()
    elif selection == "Variance Reduction (Antithetic)":
        section_variance_reduction()
    elif selection == "Heston Model (Stochastic Vol)":
        section_heston()
    elif selection == "Hardware Acceleration (JIT)":
        section_acceleration()
    elif selection == "Real-World Data (Yahoo Finance)":
        section_real_world()
    elif selection == "Comparison & Summary":
        section_comparison()

if __name__ == "__main__":
    main()
