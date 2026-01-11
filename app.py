import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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

# Attempt to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Page Config
st.set_page_config(
    page_title="Monte Carlo Simulation Compendium",
    page_icon="🎲",
    layout="wide"
)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

@st.cache_data
def fetch_stock_data(ticker):
    """Fetches stock data and returns S0, Sigma, and Returns."""
    if not HAS_YFINANCE:
        return None, None, None
    try:
        data = yf.download(ticker, period="1y", progress=False)
        if data.empty:
            return None, None, None
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        S0 = float(data['Close'].iloc[-1])
        data['Log Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        sigma = data['Log Returns'].std() * np.sqrt(252)
        returns = data['Log Returns'].dropna().values
        return S0, sigma, returns
    except Exception as e:
        return None, None, None

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

def generate_heston_paths(S0, v0, rho, kappa, theta, xi, T, M, N):
    """Generates Heston Model paths."""
    dt = T / M
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    Z = np.random.multivariate_normal(mean, cov, (N, M))
    Z_S = Z[:, :, 0]
    Z_v = Z[:, :, 1]
    
    S = np.zeros((N, M + 1))
    v = np.zeros((N, M + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    
    for t in range(M):
        v_t = v[:, t]
        v_sqrt = np.sqrt(np.maximum(v_t, 0))
        dv = kappa * (theta - v_t) * dt + xi * v_sqrt * np.sqrt(dt) * Z_v[:, t]
        v[:, t+1] = np.maximum(v_t + dv, 0)
        dS = S[:, t] * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z_S[:, t]
        S[:, t+1] = S[:, t] + dS
        
    return S, v

# Acceleration Helpers
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
        
        # Data Source Switch
        data_source = "Manual"
        if HAS_YFINANCE:
            data_source = st.radio("Source", ["Manual", "Yahoo Finance"], horizontal=True, key="mc_src")
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Ticker", "SPY", key="mc_ticker")
            S0_real, sigma_real, _ = fetch_stock_data(ticker)
            if S0_real:
                st.success(f"Last Price: ${S0_real:.2f}, Vol: {sigma_real:.2%}")
                S0 = S0_real
                sigma = sigma_real
                K = st.number_input("Strike Price ($)", value=float(S0), key="mc_k_yf")
            else:
                st.error("Failed to fetch data.")
                S0 = 100.0; sigma = 0.2; K = 105.0
        else:
            S0 = st.number_input("Spot Price ($)", 100.0, value=100.0, key="mc_s0")
            K = st.number_input("Strike Price ($)", 100.0, value=105.0, key="mc_k")
            sigma = st.slider("Volatility (σ)", 0.1, 1.0, 0.2, key="mc_sigma")
            
        T = st.number_input("Time to Maturity (Years)", 0.1, 5.0, 1.0, key="mc_t")
        n_sims = st.slider("Simulations (N)", 100, 10000, 1000, key="mc_n")
    
    if st.button("Run Crude MC", key="btn_mc"):
        start = time.time()
        paths, _ = generate_gbm(S0, 0.05, sigma, T, 252, n_sims)
        terminal_prices = paths[:, -1]
        payoffs = np.maximum(terminal_prices - K, 0)
        price_est = np.exp(-0.05 * T) * np.mean(payoffs)
        duration = time.time() - start
        true_price = black_scholes_call(S0, K, T, 0.05, sigma)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("MC Price", f"${price_est:.2f}")
        m2.metric("True Price (BS)", f"${true_price:.2f}")
        m3.metric("Error", f"${abs(price_est - true_price):.4f}")
        
        # Plotly
        fig = make_subplots(rows=1, cols=2, subplot_titles=("First 50 Sim Paths", "Convergence of Price"))
        
        # 1. Path Plot
        for i in range(min(50, n_sims)):
            fig.add_trace(go.Scatter(y=paths[i, :], mode='lines', 
                                     line=dict(width=1, color='rgba(31, 119, 180, 0.3)'), 
                                     showlegend=False, hoverinfo='skip'), row=1, col=1)
        
        # 2. Convergence
        cumulative_avg = np.cumsum(payoffs * np.exp(-0.05 * T)) / np.arange(1, n_sims + 1)
        fig.add_trace(go.Scatter(y=cumulative_avg, mode='lines', name='MC Estimate', line=dict(color='blue')), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, n_sims], y=[true_price, true_price], mode='lines', 
                                 name='True Price', line=dict(color='red', dash='dash')), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 2: QUASI MONTE CARLO
# -----------------------------------------------------------------------------
def section_quasi_mc():
    st.header("2. Quasi-Monte Carlo (QMC)")
    st.markdown("""
    **The Upgrade:** Uses **Low-Discrepancy Sequences (Sobol)** instead of random numbers. 
    """)
    n_points = st.slider("Number of Points", 100, 2000, 500, key="qmc_n")
    
    # Visualization Part
    random_pts = np.random.rand(n_points, 2)
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_pts = sampler.random(n_points)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pseudo-Random (Clumpy)", "Sobol Sequence (Uniform)"))
    fig.add_trace(go.Scatter(x=random_pts[:,0], y=random_pts[:,1], mode='markers', name='Random', marker=dict(size=5, color='blue', opacity=0.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sobol_pts[:,0], y=sobol_pts[:,1], mode='markers', name='Sobol', marker=dict(size=5, color='orange', opacity=0.6)), row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison Part
    st.subheader("Pricing Convergence Comparison")
    
    # Data Source Switch
    col1, col2 = st.columns(2)
    with col1:
        data_source = "Manual"
        if HAS_YFINANCE:
            data_source = st.radio("Pricing Source", ["Manual", "Yahoo Finance"], horizontal=True, key="qmc_src")
            
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Ticker", "NVDA", key="qmc_ticker")
            S0_real, sigma_real, _ = fetch_stock_data(ticker)
            if S0_real:
                S0 = S0_real; sigma = sigma_real
                st.caption(f"Using {ticker}: ${S0:.2f}, Vol: {sigma:.2%}")
            else:
                S0 = 100; sigma = 0.2
        else:
            S0 = 100; sigma = 0.2
            
    if st.button("Compare Convergence"):
        st.write("Running convergence test...")
        true_price = black_scholes_call(S0, S0, 1.0, 0.05, sigma) # ATM Option
        N_range = np.arange(100, 3000, 100)
        err_mc = []; err_qmc = []
        
        for n in N_range:
            # MC
            paths_mc, _ = generate_gbm(S0, 0.05, sigma, 1.0, 1, n, use_sobol=False)
            p_mc = np.exp(-0.05) * np.mean(np.maximum(paths_mc[:, -1] - S0, 0))
            err_mc.append(abs(p_mc - true_price))
            # QMC
            # Sobol needs 2^k ideally, but we simulate raw usage here
            sampler = qmc.Sobol(d=1, scramble=True)
            z = norm.ppf(sampler.random(n))
            st_q = S0 * np.exp((0.05 - 0.5*sigma**2) + sigma * z)
            p_qmc = np.exp(-0.05) * np.mean(np.maximum(st_q - S0, 0))
            err_qmc.append(abs(p_qmc - true_price))
            
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=N_range, y=err_mc, mode='lines', name='Crude MC'))
        fig2.add_trace(go.Scatter(x=N_range, y=err_qmc, mode='lines', name='Quasi MC', line=dict(color='orange')))
        fig2.update_layout(title="Error Convergence (Log Scale)", yaxis_type="log", xaxis_title="Simulations", yaxis_title="Abs Error")
        st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 3: LEAST SQUARES MONTE CARLO (LSMC)
# -----------------------------------------------------------------------------
def section_lsmc():
    st.header("3. Least Squares Monte Carlo (LSMC)")
    st.markdown("""
    **The Problem:** Pricing American Options (Early Exercise).
    **The Solution:** Longstaff-Schwartz Algorithm (Regression on backward induction).
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # Data Source Switch
        data_source = "Manual"
        if HAS_YFINANCE:
            data_source = st.radio("Source", ["Manual", "Yahoo Finance"], horizontal=True, key="lsmc_src")
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Ticker", "AMZN", key="lsmc_ticker")
            S0_real, sigma_real, _ = fetch_stock_data(ticker)
            if S0_real:
                st.success(f"Last: ${S0_real:.2f}, Vol: {sigma_real:.2%}")
                S0 = S0_real; sigma = sigma_real
                K = st.number_input("Strike ($)", value=float(S0), key="lsmc_k_yf")
            else:
                S0 = 100.0; sigma = 0.2; K = 100.0
        else:
            S0 = 100.0; K = 100.0; sigma = 0.2
            
        N = st.slider("Simulations", 100, 5000, 1000, key="lsmc_n")
        M = st.slider("Time Steps", 10, 100, 50, key="lsmc_m")
        T = 1.0; r = 0.05
    
    if st.button("Run LSMC"):
        dt = T / M
        df = np.exp(-r * dt)
        paths, _ = generate_gbm(S0, r, sigma, T, M, N)
        payoff = np.maximum(K - paths, 0) # Put Option
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
        
        fig = go.Figure()
        subset = 100
        for i in range(min(subset, N)):
            fig.add_trace(go.Scatter(y=paths[i, :], mode='lines', 
                                     line=dict(color='gray', width=0.5), opacity=0.2, showlegend=False, hoverinfo='skip'))
        ex_idx = np.where(exercise_decisions[:subset, :] == 1)
        if len(ex_idx[0]) > 0:
            fig.add_trace(go.Scatter(x=ex_idx[1], y=paths[ex_idx[0], ex_idx[1]], 
                                     mode='markers', name='Early Exercise',
                                     marker=dict(color='red', size=6, symbol='x')))
        fig.update_layout(title="Optimal Early Exercise Points", xaxis_title="Time Step", yaxis_title="Stock Price")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 4: MARKOV CHAIN MONTE CARLO (MCMC)
# -----------------------------------------------------------------------------
def section_mcmc():
    st.header("4. Markov Chain Monte Carlo (MCMC)")
    st.markdown("**The Algorithm:** Metropolis-Hastings.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        data_mode = "Synthetic"
        if HAS_YFINANCE:
            data_mode = st.radio("Target Distribution", ["Synthetic Data", "Real Stock Returns"], key="mcmc_mode")
            
        if data_mode == "Real Stock Returns":
            ticker = st.text_input("Ticker", "MSFT", key="mcmc_ticker")
            _, _, returns = fetch_stock_data(ticker)
            if returns is not None:
                data = returns * 100 # Convert to percentage for easier viewing
                st.caption(f"Fitting Normal Dist to {ticker} Daily Returns (%)")
            else:
                st.error("Error fetching data.")
                data = np.random.normal(5.0, 2.0, 100)
        else:
            true_mu = st.number_input("True Mean", -10.0, 10.0, 5.0)
            data = np.random.normal(true_mu, 2.0, 100)
    
    # Plot Data
    fig_data = go.Figure(data=go.Histogram(x=data, nbinsx=30, name='Data'))
    fig_data.update_layout(title="Observed Data Distribution", height=300)
    st.plotly_chart(fig_data, use_container_width=True)
    
    if st.button("Run MCMC Sampler"):
        # MCMC to find Mean of data
        mu_current = 0.0 # Random start
        samples = []
        fixed_sigma = np.std(data) # Simplify by assuming sigma is known/fixed to sample std
        
        for i in range(5000):
            mu_proposal = mu_current + np.random.normal(0, 0.5)
            # Log Likelihoods
            ll_curr = np.sum(norm.logpdf(data, mu_current, fixed_sigma))
            ll_prop = np.sum(norm.logpdf(data, mu_proposal, fixed_sigma))
            
            if np.log(np.random.rand()) < (ll_prop - ll_curr):
                mu_current = mu_proposal
            samples.append(mu_current)
            
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Trace Plot (Walker)", "Posterior Distribution of Mean"))
        fig.add_trace(go.Scatter(y=samples, mode='lines', name='Walker', line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Histogram(x=samples[1000:], name='Posterior', marker_color='green', opacity=0.7), row=2, col=1)
        
        estimated_mean = np.mean(samples[1000:])
        fig.add_vline(x=estimated_mean, line_dash="dash", line_color="black", annotation_text=f"Est: {estimated_mean:.2f}", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"MCMC Estimated Mean: {estimated_mean:.4f}")

# -----------------------------------------------------------------------------
# SECTION 5: SEQUENTIAL MONTE CARLO (SMC)
# -----------------------------------------------------------------------------
def section_smc():
    st.header("5. Sequential Monte Carlo (Particle Filters)")
    st.markdown("Tracking a robot using noisy measurements. (Robotics Context)")
    st.caption("Note: This module uses synthetic robot data as it simulates a physical tracking problem.")
    
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
        uncertainty = []
        
        for t in range(T_steps):
            particles += 1.0 + np.random.normal(0, 0.5, n_particles)
            dist = measurements[t] - particles
            weights = weights * np.exp(-0.5 * (dist**2) / 4.0)
            weights += 1.e-300
            weights /= np.sum(weights)
            est = np.sum(particles * weights)
            estimates.append(est)
            uncertainty.append(np.std(particles))
            indices = np.random.choice(np.arange(n_particles), size=n_particles, p=weights)
            particles = particles[indices]
            weights.fill(1.0/n_particles)
            
        fig = go.Figure()
        est_arr = np.array(estimates); unc_arr = np.array(uncertainty)
        x_axis = np.arange(T_steps)
        
        fig.add_trace(go.Scatter(x=x_axis, y=est_arr+unc_arr, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x_axis, y=est_arr-unc_arr, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', name='Uncertainty'))
        fig.add_trace(go.Scatter(x=x_axis, y=true_position, mode='lines', name='True Path', line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=x_axis, y=measurements, mode='markers', name='Noisy GPS', marker=dict(color='green', symbol='x')))
        fig.add_trace(go.Scatter(x=x_axis, y=estimates, mode='lines', name='SMC Estimate', line=dict(color='red', dash='dash')))
        
        fig.update_layout(title="Particle Filter Tracking", xaxis_title="Time Step", yaxis_title="Position")
        st.plotly_chart(fig, use_container_width=True)

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
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Wavefunction Density", "Energy Convergence"))
        fig.add_trace(go.Histogram(x=positions, histnorm='probability density', name='Density', marker_color='purple'), row=1, col=1)
        
        x_grid = np.linspace(-3, 3, 100)
        psi_squared = np.sqrt(2*alpha/np.pi) * np.exp(-2*alpha*x_grid**2)
        fig.add_trace(go.Scatter(x=x_grid, y=psi_squared, mode='lines', name='Analytic', line=dict(color='black', dash='dash')), row=1, col=1)
        
        cumulative_energy = np.cumsum(energies) / np.arange(1, len(energies)+1)
        fig.add_trace(go.Scatter(y=cumulative_energy, mode='lines', name='Est. Energy'), row=1, col=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Ground State (0.5)", row=1, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

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
    **The Concept:** Reduce variance by averaging paths driven by $Z$ and $-Z$.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        data_source = "Manual"
        if HAS_YFINANCE:
            data_source = st.radio("Source", ["Manual", "Yahoo Finance"], horizontal=True, key="vr_src")
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Ticker", "GOOGL", key="vr_ticker")
            S0_real, sigma_real, _ = fetch_stock_data(ticker)
            if S0_real:
                S0 = S0_real; sigma = sigma_real
                K = st.number_input("Strike ($)", value=float(S0), key="vr_k_yf")
            else:
                S0 = 100.0; sigma = 0.2; K = 100.0
        else:
            S0 = 100.0; K = 100.0; sigma = 0.2
            
        T = 1.0; r = 0.05
        n_sims = st.slider("Simulations (N)", 100, 5000, 1000, key="vr_n")
    
    if st.button("Compare Standard vs. Antithetic"):
        # Standard
        paths_std, _ = generate_gbm(S0, r, sigma, T, 1, n_sims)
        payoffs_std = np.maximum(paths_std[:,-1] - K, 0) * np.exp(-r*T)
        
        # Antithetic
        n_half = n_sims // 2
        Z = np.random.normal(0, 1, (n_half, 1))
        ST_1 = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        ST_2 = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*(-Z))
        payoffs_pair = 0.5 * (np.maximum(ST_1 - K, 0) + np.maximum(ST_2 - K, 0)) * np.exp(-r*T)
        
        conv_std = np.cumsum(payoffs_std) / np.arange(1, n_sims + 1)
        conv_anti = np.cumsum(payoffs_pair) / np.arange(1, n_half + 1)
        true_price = black_scholes_call(S0, K, T, r, sigma)
        
        var_std = np.var(payoffs_std)
        var_anti = np.var(payoffs_pair)
        
        col1, col2 = st.columns(2)
        col1.metric("Standard Variance", f"{var_std:.4f}")
        col2.metric("Antithetic Variance", f"{var_anti:.4f}", delta=f"-{(1 - var_anti/var_std)*100:.1f}%")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=conv_std, mode='lines', name='Standard MC', opacity=0.7))
        fig.add_trace(go.Scatter(x=np.linspace(0, n_sims, n_half), y=conv_anti, mode='lines', name='Antithetic MC', line=dict(color='green')))
        fig.add_hline(y=true_price, line_dash="dash", line_color="red", annotation_text="True Price")
        fig.update_layout(title="Convergence Comparison", xaxis_title="Iterations", yaxis_title="Price Est")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# SECTION 9: ADVANCED STOCHASTIC PROCESSES (HESTON)
# -----------------------------------------------------------------------------
def section_heston():
    st.header("9. Heston Stochastic Volatility Model")
    st.markdown("Simulating with stochastic volatility.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        data_source = "Manual"
        if HAS_YFINANCE:
            data_source = st.radio("Source", ["Manual", "Yahoo Finance"], horizontal=True, key="heston_src")
        
        if data_source == "Yahoo Finance":
            ticker = st.text_input("Ticker", "TSLA", key="heston_ticker")
            S0_real, sigma_real, _ = fetch_stock_data(ticker)
            if S0_real:
                st.caption(f"Using {ticker} (S0=${S0_real:.2f}, Hist Vol={sigma_real:.2%})")
                S0 = S0_real
                # Use Historical Vol as proxy for initial Heston parameters
                theta = st.number_input("Long Term Var (θ)", 0.01, 1.0, float(sigma_real**2))
            else:
                S0 = 100; theta = 0.04
        else:
            S0 = 100
            theta = st.number_input("Long Term Var (θ)", 0.01, 0.5, 0.04)
        
        rho = st.slider("Correlation", -1.0, 1.0, -0.7)
        xi = st.slider("Vol of Vol", 0.1, 1.0, 0.3)
        kappa = 2.0
        
    if st.button("Simulate Heston Paths"):
        # Using theta as initial variance proxy as well
        S, v = generate_heston_paths(S0, theta, rho, kappa, theta, xi, 1.0, 252, 10)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, 
                            subplot_titles=("Stock Price Paths", "Stochastic Volatility Paths"))
        for i in range(S.shape[0]):
            fig.add_trace(go.Scatter(y=S[i, :], mode='lines', line=dict(color='rgba(31, 119, 180, 0.5)')), row=1, col=1)
            fig.add_trace(go.Scatter(y=np.sqrt(v[i, :]), mode='lines', line=dict(color='rgba(255, 127, 14, 0.5)')), row=2, col=1)
            
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# SECTION 11: REAL-WORLD DATA (YAHOO FINANCE)
# -----------------------------------------------------------------------------

def section_real_world():
    st.header("11. Real-World Data (Yahoo Finance)")
    
    if not HAS_YFINANCE: 
        st.error("⚠️ `yfinance` library is not installed."); return

    # --- Controls ---
    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        ticker = st.text_input("Ticker", "NVDA", key="rw_ticker").upper()
    with col2: 
        forecast_days = st.slider("Days", 10, 252, 60)
    with col3:
        drift_type = st.selectbox("Drift", ["Risk-Free (5%)", "Historical Mean"])
    with col4:
        n_sims = st.selectbox("Simulations", [500, 1000, 2000], index=1)

    if st.button("Run Quant Analysis", key="btn_rw", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # --- 1. Data Fetching & Cleaning ---
                data = yf.download(ticker, period="2y", progress=False)
                if data.empty: st.error("No data."); return
                
                # Fix MultiIndex
                if isinstance(data.columns, pd.MultiIndex):
                    try: data = data.xs(ticker, axis=1, level=0)
                    except: data.columns = data.columns.get_level_values(0)
                
                prices = data['Close']
                if len(prices) < 60: st.error("Not enough history."); return
                
                # --- 2. Parameter Estimation ---
                S0 = float(prices.iloc[-1])
                log_returns = np.log(prices / prices.shift(1)).dropna()
                
                # Annualized Volatility
                sigma = log_returns.std() * np.sqrt(252)
                
                # Drift Calculation
                if drift_type == "Historical Mean":
                    mu = log_returns.mean() * 252 + 0.5 * sigma**2
                else:
                    mu = 0.05

                # --- 3. Run Monte Carlo ---
                T_years = forecast_days / 252.0
                # generate_gbm(S0, mu, sigma, T, steps, n_sims)
                paths, _ = generate_gbm(S0, mu, sigma, T_years, forecast_days, n_sims)
                
                # --- 4. Main Fan Chart ---
                future_dates = pd.date_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
                
                # Quantiles
                sim_data = paths[:, 1:] # Exclude S0
                p05 = np.percentile(sim_data, 5, axis=0)
                p50 = np.percentile(sim_data, 50, axis=0)
                p95 = np.percentile(sim_data, 95, axis=0)

                # Summary Metrics
                final_prices = paths[:, -1]
                exp_return = (np.mean(final_prices) / S0) - 1
                prob_profit = np.mean(final_prices > S0)

                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Current Price", f"${S0:.2f}")
                m2.metric("Volatility (Ann.)", f"{sigma*100:.1f}%")
                m3.metric("Win Probability", f"{prob_profit*100:.0f}%", help="Prob. price > Current Price")
                m4.metric("Exp. Return", f"{exp_return*100:.1f}%", help=f"Over {forecast_days} days")

                # Plot Main Chart
                fig_main = go.Figure()
                # Historical
                hist_show = prices.tail(90)
                fig_main.add_trace(go.Scatter(x=hist_show.index, y=hist_show.values, mode='lines', name='Historical', line=dict(color='black')))
                # Fan Chart
                fig_main.add_trace(go.Scatter(x=future_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False))
                fig_main.add_trace(go.Scatter(x=future_dates, y=p05, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.2)', name='90% Conf. Interval'))
                fig_main.add_trace(go.Scatter(x=future_dates, y=p50, mode='lines', name='Median Forecast', line=dict(color='blue')))
                
                fig_main.update_layout(title=f"Price Forecast: {ticker}", height=400, yaxis_title="Price", hovermode="x unified")
                st.plotly_chart(fig_main, use_container_width=True)

                # --- 5. DEEP DIVE TABS ---
                tab1, tab2, tab3 = st.tabs(["📊 Risk Distribution", "📉 Drawdown Analysis", "⏳ Volatility Regime"])

                with tab1:
                    st.markdown("##### Distribution of Final Prices")
                    # Calculate VaR
                    sorted_returns = np.sort((final_prices - S0) / S0)
                    var_95 = np.percentile(sorted_returns, 5)
                    cvar_95 = sorted_returns[sorted_returns <= var_95].mean()

                    col_var1, col_var2 = st.columns(2)
                    col_var1.error(f"VaR (95%): {var_95*100:.1f}%")
                    col_var2.error(f"CVaR (Expected Shortfall): {cvar_95*100:.1f}%")

                    fig_hist = px.histogram(final_prices, nbins=50, title="Final Price Distribution", labels={'value': 'Price'})
                    fig_hist.add_vline(x=S0, line_dash="dash", line_color="black", annotation_text="Start Price")
                    fig_hist.add_vline(x=np.percentile(final_prices, 5), line_color="red", annotation_text="95% VaR")
                    fig_hist.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_hist, use_container_width=True)

                with tab2:
                    st.markdown("##### Simulated Maximum Drawdowns")
                    st.caption("This answers: *'Even if the trend is up, how much might I lose along the way?'*")
                    
                    # Vectorized Drawdown Calculation for all paths
                    # accumulated max along time axis
                    cum_max = np.maximum.accumulate(paths, axis=1)
                    drawdowns = (paths - cum_max) / cum_max
                    max_drawdowns = np.min(drawdowns, axis=1) # Min because DD is negative

                    avg_dd = np.mean(max_drawdowns)
                    worst_dd = np.min(max_drawdowns)

                    st.write(f"**Average Max Drawdown:** {avg_dd*100:.1f}% | **Worst Case Scenario:** {worst_dd*100:.1f}%")

                    fig_dd = px.histogram(max_drawdowns, nbins=50, title="Distribution of Max Drawdowns", labels={'value': 'Max Drawdown'})
                    fig_dd.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_dd, use_container_width=True)

                with tab3:
                    st.markdown("##### Historical Volatility Regime")
                    st.caption("Is the current market calm or chaotic compared to the past?")
                    
                    # Rolling 30-day Volatility
                    rolling_vol = log_returns.rolling(window=30).std() * np.sqrt(252)
                    
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values, name="30D Rolling Vol"))
                    fig_vol.add_hline(y=sigma, line_dash="dash", line_color="red", annotation_text="Current Sim Input")
                    fig_vol.update_layout(title="Historical 30-Day Annualized Volatility", height=350, yaxis_title="Volatility")
                    st.plotly_chart(fig_vol, use_container_width=True)

            except Exception as e:
                st.error(f"Quant Analysis Failed: {e}")

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
        "Real-World Data (Yahoo Finance)",
        "Comparison & Summary"
    ]
    
    selection = st.sidebar.selectbox("Select Method", options)
    
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
