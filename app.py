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

# --- HELPER: Autocorrelation Function ---
def calculate_autocorrelation(x, lag=20):
    n = len(x)
    variance = np.var(x)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r / (variance * (np.arange(n, 0, -1)))
    return result[:lag]

def section_mcmc():
    st.header("4. Markov Chain Monte Carlo (MCMC)")
    
    # --- 1. EDUCATIONAL CONTEXT ---
    st.markdown("""
    **The Goal:** We want to find the unknown "Mean" ($\mu$) of a distribution, but instead of calculating it directly, 
    we will walk around the probability landscape to "sample" it. This is useful when the math is too hard to solve directly.
    """)

    with st.expander("📘 Theory: Metropolis-Hastings Algorithm"):
        st.markdown(r"""
        **How the Walker Decides:**
        1.  **Propose:** The walker suggests a new location ($\theta_{new}$) based on where they are now ($\theta_{current}$).
        2.  **Evaluate:** Compare the likelihood of the data at the new location vs. the old location.
        3.  **Accept/Reject Ratio ($r$):**
            $$ r = \frac{Likelihood(\theta_{new}) \times Prior(\theta_{new})}{Likelihood(\theta_{current}) \times Prior(\theta_{current})} $$
        4.  **Decision:** * If $r > 1$ (New spot is better): **Move there.**
            * If $r < 1$ (New spot is worse): Move there with probability $r$, otherwise **stay put**.
        
        *This randomness allows the walker to escape local traps and explore the whole distribution.*
        """)

    # --- 2. SETUP & INPUTS ---
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        data_mode = "Synthetic"
        if HAS_YFINANCE:
            data_mode = st.radio("Data Source", ["Synthetic Data", "Real Stock Returns"], key="mcmc_mode")
            
        if data_mode == "Real Stock Returns":
            ticker = st.text_input("Ticker", "MSFT", key="mcmc_ticker").upper()
            try:
                # Fetch recent data
                df = yf.download(ticker, period="1y", progress=False)
                if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
                
                # Calculate daily returns
                returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
                data = returns.values * 100 # Scale to percentage
                
                st.caption(f"Loaded {len(data)} days of returns for {ticker}.")
                st.metric("Observed Mean", f"{np.mean(data):.4f}%")
            except:
                st.error("Data fetch failed. Using synthetic.")
                data = np.random.normal(5.0, 2.0, 100)
        else:
            true_mu = st.number_input("True Hidden Mean", -10.0, 10.0, 3.5)
            noise = st.slider("Noise Level (Std Dev)", 0.5, 5.0, 2.0)
            data = np.random.normal(true_mu, noise, 200)

        # Algorithm Tuning
        st.markdown("---")
        st.markdown("**Algo Parameters**")
        n_iterations = st.slider("Iterations", 1000, 10000, 5000)
        burn_in = st.slider("Burn-in Period", 0, 2000, 1000, help="Early samples to discard while the walker finds the groove.")
        proposal_width = st.slider("Step Size", 0.1, 2.0, 0.5, help="How big are the walker's steps? Too big = rejection; Too small = slow exploration.")

    with col2:
        # Show the "Mystery" Data
        fig_data = go.Figure(data=go.Histogram(x=data, nbinsx=40, name='Observed Data', marker_color='gray', opacity=0.6))
        fig_data.add_vline(x=np.mean(data), line_dash="dash", annotation_text="Observed Mean")
        fig_data.update_layout(title="The Observed Data (We want to find the parameter that best fits this)", height=350, showlegend=False)
        st.plotly_chart(fig_data, use_container_width=True)

    # --- 3. RUN SIMULATION ---
    if st.button("🚀 Run Metropolis-Hastings Sampler", use_container_width=True):
        
        # Initialize
        mu_current = 0.0 # Arbitrary start
        samples = []
        accepted = 0
        fixed_sigma = np.std(data) # Assuming known sigma for simplicity
        
        # Progress Bar
        progress_bar = st.progress(0)
        
        for i in range(n_iterations):
            # 1. Propose new mu
            mu_proposal = mu_current + np.random.normal(0, proposal_width)
            
            # 2. Calculate Likelihoods (Log scale to prevent underflow)
            # We assume a flat prior (prior ratio = 1), so we ignore it here
            ll_curr = np.sum(norm.logpdf(data, mu_current, fixed_sigma))
            ll_prop = np.sum(norm.logpdf(data, mu_proposal, fixed_sigma))
            
            # 3. Accept or Reject
            ratio = ll_prop - ll_curr # Log subtraction = Division
            accept_prob = np.exp(ratio)
            
            if np.random.rand() < accept_prob:
                mu_current = mu_proposal
                accepted += 1
            
            samples.append(mu_current)
            
            if i % (n_iterations // 100) == 0:
                progress_bar.progress(i / n_iterations)
        
        progress_bar.empty()
        
        # --- 4. VISUALIZATION & ANALYSIS ---
        
        # Slice data
        samples_array = np.array(samples)
        posterior_samples = samples_array[burn_in:]
        
        # Metrics
        acceptance_rate = accepted / n_iterations
        est_mean = np.mean(posterior_samples)
        cred_interval = np.percentile(posterior_samples, [2.5, 97.5])
        
        # TABBED OUTPUT
        tab1, tab2, tab3 = st.tabs(["📉 Trace & Posterior", "🩺 Diagnostics", "📝 Interpretation"])
        
        with tab1:
            st.markdown("##### The Walker's Journey (Trace Plot)")
            
            
            fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Walker Path (Trace Plot)", "Posterior Distribution (The Result)"),
                                vertical_spacing=0.15)
            
            # Trace Plot (with Burn-in Highlight)
            fig.add_trace(go.Scatter(x=np.arange(burn_in), y=samples[:burn_in], 
                                     mode='lines', name='Burn-in (Discarded)', line=dict(color='red', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=np.arange(burn_in, n_iterations), y=samples[burn_in:], 
                                     mode='lines', name='Valid Samples', line=dict(color='blue', width=1)), row=1, col=1)
            
            # Posterior Histogram
            fig.add_trace(go.Histogram(x=posterior_samples, nbinsx=50, name='Posterior', marker_color='green', opacity=0.7), row=2, col=1)
            fig.add_vline(x=est_mean, line_width=3, line_color='black', annotation_text=f"Est: {est_mean:.3f}", row=2, col=1)
            
            # Add Credible Interval Area
            fig.add_vrect(x0=cred_interval[0], x1=cred_interval[1], fillcolor="green", opacity=0.1, 
                          annotation_text="95% Credible Interval", annotation_position="top left", row=2, col=1)

            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.markdown("##### Chain Diagnostics")
            c1, c2 = st.columns(2)
            
            with c1:
                st.metric("Acceptance Rate", f"{acceptance_rate:.1%}", help="Ideal is between 20% and 50%.")
                if acceptance_rate < 0.1: st.warning("⚠️ Too low! Decrease Step Size.")
                elif acceptance_rate > 0.8: st.warning("⚠️ Too high! Increase Step Size.")
                else: st.success("✅ Good acceptance rate.")

            with c2:
                # Autocorrelation Plot
                lags = 40
                acf = calculate_autocorrelation(posterior_samples, lags)
                fig_acf = go.Figure(go.Bar(x=np.arange(lags), y=acf, marker_color='purple'))
                fig_acf.update_layout(title="Autocorrelation", xaxis_title="Lag", yaxis_title="Correlation", height=250)
                st.plotly_chart(fig_acf, use_container_width=True)
                st.caption("Lower bars are better. High bars mean the walker is 'sticky' and moving slowly.")

        with tab3:
            st.markdown("### 💡 What does this mean?")
            st.success(f"""
            Based on the MCMC sampling:
            1.  The **Most Likely Value** for the mean is **{est_mean:.4f}**.
            2.  We are **95% confident** (Credible Interval) that the true mean is between **{cred_interval[0]:.4f}** and **{cred_interval[1]:.4f}**.
            """)
            
            st.info("""
            **Why use MCMC here?**
            For a simple Normal distribution, we could have just calculated the average directly. 
            But this method proves that the "Walker" can find the answer blindly just by comparing likelihoods. 
            This same technique is used to solve massive 100-dimensional problems in physics and finance where simple averaging is impossible.
            """)
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
    st.header("7. Method Comparison & Summary")
    
    st.markdown("""
    Choosing the right Monte Carlo method is a trade-off between **speed**, **accuracy**, and **complexity**.
    Below is a breakdown of when to use which algorithm.
    """)

    # --- 1. DETAILED DATA TABLE ---
    comparison_data = [
        {"Method": "Standard MC", "Speed": "⭐⭐", "Convergence": "O(1/√N)", "Best For": "Simple Integrals, Euclidean Options", "Cons": "Slow convergence, 'Clumping'"},
        {"Method": "Quasi-MC (QMC)", "Speed": "⭐⭐⭐⭐", "Convergence": "O(1/N)", "Best For": "High-dim Integrals, Pricing", "Cons": "Correlation artifacts in very high dims"},
        {"Method": "LSMC (American)", "Speed": "⭐", "Convergence": "N/A", "Best For": "American/Bermudan Options", "Cons": "Computationally heavy (Regression)"},
        {"Method": "MCMC", "Speed": "⭐", "Convergence": "Variable", "Best For": "Bayesian Inference, Unknown Posteriors", "Cons": "Hard to diagnose convergence"},
        {"Method": "Stratified", "Speed": "⭐⭐⭐", "Convergence": "O(1/N)", "Best For": "1D-2D problems with known sub-groups", "Cons": "Hard to implement in high dimensions"}
    ]
    
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # --- 2. VISUAL CONVERGENCE DEMO ---
    st.markdown("---")
    st.subheader("🏎️ The Race: Random vs. Quasi-Random")
    st.caption("Why does QMC matter? Watch how much faster the error drops compared to Standard MC.")
    
    

    if st.button("Run Convergence Race", use_container_width=True):
        # Simulation: Estimate Pi using Monte Carlo
        # Area of circle = pi * r^2. If r=1, Area = pi.
        # Square area = 4. Ratio = pi/4.
        
        N_max = 2000
        step = 50
        n_values = range(step, N_max + 1, step)
        
        error_mc = []
        error_qmc = []
        
        # True value
        true_pi = np.pi
        
        # 1. Standard MC (Random)
        # We calculate cumulative averages for smooth plotting
        pts_x = np.random.rand(N_max)
        pts_y = np.random.rand(N_max)
        inside_circle = (pts_x**2 + pts_y**2) <= 1.0
        
        # 2. Quasi-MC (Simple Stratified / Low Discrepancy mimic)
        # For a true QMC demo without heavy dependencies (scipy.qmc), we use a simple Golden Ratio sequence (1D)
        # mapped to 2D for demonstration, or just simulate the convergence rate mathematically for the visual.
        # Here we will use a "Van der Corput" style generator for actual code robustness:
        
        def van_der_corput(n, base=2):
            vdc, denom = 0, 1
            while n:
                denom *= base
                n, remainder = divmod(n, base)
                vdc += remainder / denom
            return vdc
            
        # Generate Halton Sequence (Base 2, Base 3) for X, Y
        qmc_x = [van_der_corput(i, 2) for i in range(1, N_max + 1)]
        qmc_y = [van_der_corput(i, 3) for i in range(1, N_max + 1)]
        inside_qmc = (np.array(qmc_x)**2 + np.array(qmc_y)**2) <= 1.0

        # Calculate errors at steps
        for n in n_values:
            # MC
            pi_est_mc = 4 * np.sum(inside_circle[:n]) / n
            error_mc.append(abs(pi_est_mc - true_pi))
            
            # QMC
            pi_est_qmc = 4 * np.sum(inside_qmc[:n]) / n
            error_qmc.append(abs(pi_est_qmc - true_pi))

        # --- PLOT RESULTS ---
        fig = go.Figure()
        
        # Log-Log plot is standard for convergence analysis
        fig.add_trace(go.Scatter(x=list(n_values), y=error_mc, mode='lines', name='Standard MC (Random)', line=dict(color='red', width=1)))
        fig.add_trace(go.Scatter(x=list(n_values), y=error_qmc, mode='lines', name='Quasi-MC (Low Discrepancy)', line=dict(color='blue', width=2)))
        
        # Add theoretical slopes (Optional, for advanced users)
        # 1/sqrt(N) line
        ref_y = [1/np.sqrt(x) for x in n_values]
        fig.add_trace(go.Scatter(x=list(n_values), y=ref_y, mode='lines', name='Theory: 1/√N', line=dict(color='green', dash='dash')))

        fig.update_layout(
            title="Convergence Speed (Log-Log Scale)",
            xaxis_title="Number of Simulations (N)",
            yaxis_title="Absolute Error",
            xaxis_type="log", yaxis_type="log",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("Analysis: Notice how the Blue line (QMC) generally stays below the Red line (Standard MC). This means QMC achieves the same accuracy with fewer simulations.")

    # --- 3. DEEP DIVE SELECTOR ---
    st.markdown("---")
    st.subheader("💡 Method Deep Dive")
    selected_method = st.selectbox("Select a method to learn more:", df_comp["Method"])
    
    # Simple dictionary lookup for descriptions
    details = {
        "Standard MC": "The 'Hammer'. It uses pseudo-random numbers. It's robust and works in any dimension, but it's slow. It clusters points randomly, leaving gaps in the integration space.",
        "Quasi-MC (QMC)": "The 'Scalpel'. It uses Low-Discrepancy Sequences (like Sobol or Halton). These sequences are deterministic and designed to fill space evenly, avoiding gaps. This creates O(1/N) convergence.",
        "LSMC (American)": "The 'Time Traveler'. American options can be exercised early. LSMC simulates paths forward, then works backward using Linear Regression to estimate the 'continuation value' at each step.",
        "MCMC": "The 'Explorer'. Used when we don't know the shape of the distribution we are sampling from. It crawls through probability space. Essential for Bayesian stats but harder to parallelize than Standard MC.",
        "Stratified": "The 'Divider'. Splits the population into subgroups (strata) and samples from each. Great if you know your data has distinct groups (e.g., polling different age groups), but gets messy in high dimensions."
    }
    
    st.info(f"**{selected_method}**: {details.get(selected_method.split(' [')[0], 'Select a method')}")

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



# NOTE: Ensure 'generate_heston_paths' is defined in your backend logic.
# If you need the logic for that function, I can provide it.

def section_heston():
    st.header("9. Heston Stochastic Volatility Model")
    
    # --- 1. EDUCATIONAL HEADER & ALGO DESCRIPTION ---
    st.markdown("""
    The Heston Model improves upon Black-Scholes by assuming **volatility is not constant**. 
    Instead, volatility is a random process itself, allowing us to model real-world market behaviors 
    like "fat tails" and volatility smiles.
    """)

    with st.expander("📘 Theory & Algorithm Details"):
        st.markdown(r"""
        ### The Mathematical Model
        The Heston model uses two coupled Stochastic Differential Equations (SDEs):

        1.  **Asset Price Process ($dS_t$):**
            $$ dS_t = \mu S_t dt + \sqrt{v_t} S_t dW_t^S $$
            *Prices drift by $\mu$ but are shaken by stochastic variance $v_t$.*

        2.  **Variance Process ($dv_t$):**
            $$ dv_t = \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v $$
            *Variance follows a "Mean Reverting Square Root" process (Cox-Ingersoll-Ross).*

        ### Key Parameters
        * **$\theta$ (Theta):** Long-run average variance. Volatility tends to come back to $\sqrt{\theta}$.
        * **$\kappa$ (Kappa):** Mean reversion speed. High $\kappa$ = snaps back to $\theta$ quickly.
        * **$\xi$ (Xi):** "Vol of Vol". Determines how wild the swings in volatility are.
        * **$\rho$ (Rho):** Correlation. Critical for the "Leverage Effect" (typically negative for equities).

        ### Simulation Algorithm
        We use the **Euler-Maruyama Discretization** with a **Full Truncation Scheme**. 
        Since standard discretization can result in negative variance (impossible in real life), 
        we treat negative steps in $v_t$ as 0 for the next iteration step.
        """)

    # --- 2. INPUT PARAMETERS ---
    col_input, col_sim = st.columns([1, 2])
    
    with col_input:
        st.subheader("⚙️ Model Params")
        
        # Smart Defaults based on Ticker
        default_S0 = 100.0; default_theta = 0.04
        
        if HAS_YFINANCE:
            src = st.toggle("Use Real Data", value=False)
            if src:
                ticker = st.text_input("Ticker", "SPY").upper()
                try:
                    df = yf.download(ticker, period="1y", progress=False)
                    if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
                    
                    S0_real = float(df['Close'].iloc[-1])
                    hist_vol = np.log(df['Close']/df['Close'].shift(1)).std() * np.sqrt(252)
                    
                    st.info(f"Loaded {ticker}: Price=${S0_real:.0f}, Vol={hist_vol:.1%}")
                    default_S0 = S0_real
                    default_theta = hist_vol**2
                except:
                    st.error("Data fetch failed.")

        # Parameters
        S0 = st.number_input("Initial Price ($)", value=default_S0)
        v0 = st.number_input("Initial Variance (v0)", value=default_theta, format="%.4f")
        
        st.markdown("---")
        kappa = st.slider("Mean Reversion (κ)", 0.5, 10.0, 2.0, help="Speed at which vol returns to average.")
        theta = st.slider("Long-Run Variance (θ)", 0.01, 0.5, default_theta, step=0.01, help="The gravity center for variance.")
        xi = st.slider("Vol of Vol (ξ)", 0.1, 2.0, 0.3, help="Variance of the variance.")
        rho = st.slider("Correlation (ρ)", -1.0, 1.0, -0.7, help="Correlation between asset and vol shocks.")

        # Feller Check
        if 2*kappa*theta > xi**2:
            st.success("✅ Feller Condition Met (Stable)")
        else:
            st.warning("⚠️ Feller Violation (Unstable: Vol may hit 0)")

    with col_sim:
        # --- 3. SIMULATION & ANALYSIS ---
        st.subheader("📊 Analysis")
        T_days = st.slider("Time Horizon (Days)", 30, 756, 252)
        n_sims = st.selectbox("Simulations", [20, 50, 100], index=1)
        
        if st.button("Run Simulation", use_container_width=True):
            T_years = T_days / 252.0
            
            with st.spinner("Solving Stochastic Differential Equations..."):
                # Run the algo (Assuming generate_heston_paths exists)
                S, v = generate_heston_paths(S0, v0, rho, kappa, theta, xi, T_years, T_days, n_sims)
                time_axis = np.linspace(0, T_days, T_days+1)

                # TABS
                tab1, tab2, tab3 = st.tabs(["📈 Path Dynamics", "🔗 Leverage Effect", "Distribution"])
                
                # --- TAB 1: DYNAMICS ---
                with tab1:
                    st.caption("**Interpretation:** The top chart shows asset prices. The bottom shows the volatility *driving* those prices. Notice how spikes in the bottom chart (Vol) often correspond to crashes in the top chart (Price) due to the correlation.")
                    
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                        subplot_titles=("Asset Price Paths ($S_t$)", "Instantaneous Volatility ($\sqrt{v_t}$)"))

                    # Plot first 30 paths to keep it clean
                    for i in range(min(n_sims, 30)):
                        fig.add_trace(go.Scatter(x=time_axis, y=S[i], mode='lines', line=dict(width=1, color='rgba(31, 119, 180, 0.3)')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=time_axis, y=np.sqrt(v[i]), mode='lines', line=dict(width=1, color='rgba(255, 127, 14, 0.3)')), row=2, col=1)

                    # Add Mean Paths
                    fig.add_trace(go.Scatter(x=time_axis, y=np.mean(S, axis=0), name="Mean Price", line=dict(color='black', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=time_axis, y=np.mean(np.sqrt(v), axis=0), name="Mean Vol", line=dict(color='red', dash='dash')), row=2, col=1)

                    fig.update_layout(height=500, showlegend=False, hovermode="x unified")
                    st.plotly_chart(fig, use_container_width=True)

                # --- TAB 2: LEVERAGE EFFECT ---
                with tab2:
                    st.markdown("##### The Leverage Effect")
                    st.info(f"""
                    **Algo Insight:** This plot validates your input of **ρ = {rho}**.
                    
                    
                    
                    * **Negative Rho (-)**: When price returns are negative (left side of x-axis), volatility changes tend to be positive (top side of y-axis). This creates a downward sloping cloud.
                    * **Interpretation:** Panic selling drives fear (volatility) up.
                    """)
                    
                    # Compute daily changes
                    log_rets = np.diff(np.log(S), axis=1).flatten()
                    vol_diff = np.diff(np.sqrt(v), axis=1).flatten()
                    
                    # Downsample for performance
                    idx = np.random.choice(len(log_rets), size=min(2000, len(log_rets)), replace=False)
                    
                    fig_corr = go.Figure(go.Scatter(
                        x=log_rets[idx], y=vol_diff[idx], mode='markers',
                        marker=dict(color=log_rets[idx], colorscale='RdBu', opacity=0.6)
                    ))
                    fig_corr.update_layout(xaxis_title="Daily Stock Return", yaxis_title="Daily Change in Volatility", height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)

                # --- TAB 3: DISTRIBUTION ---
                with tab3:
                    st.markdown("##### Terminal Price Distribution")
                    st.caption("Standard Black-Scholes assumes a Log-Normal distribution. Heston often results in 'Fat Tails' (higher kurtosis), meaning extreme events are more likely than standard theory predicts.")
                    
                    final_prices = S[:, -1]
                    fig_hist = go.Figure(go.Histogram(x=final_prices, nbinsx=30, marker_color='rgb(50, 100, 200)'))
                    fig_hist.add_vline(x=S0, line_dash="dash", annotation_text="Start")
                    fig_hist.update_layout(xaxis_title="Final Price", yaxis_title="Frequency", height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)


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

def section_about():
    st.header("ℹ️ About the Monte Carlo Compendium")
    st.markdown("### *From Randomness to Certainty*")
    
    # --- INTRODUCTION ---
    st.markdown("""
    **The Monte Carlo Simulation Compendium** is an interactive educational dashboard designed to demystify stochastic methods used in **Quantitative Finance**, **Physics**, and **Data Science**.
    
    While textbooks provide static formulas, this application allows you to **tweak parameters in real-time**, visualize the "Law of Large Numbers," and understand how computational algorithms solve problems that are analytically difficult (or impossible).
    """)

    st.divider()

    # --- TECH STACK & DISCLAIMER ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
        This application is built with a high-performance Python stack:
        * **Core Logic:** `NumPy` (Vectorization), `SciPy` (Stats)
        * **Visualization:** `Plotly` (Interactive & 3D Charts)
        * **UI Framework:** `Streamlit`
        * **Financial Data:** `yfinance` (Live API)
        * **Acceleration:** `Numba` (JIT Compilation for speed)
        * **Quasi-Random:** `SciPy.stats.qmc` (Sobol Sequences)
        """)
        
    with col2:
        st.subheader("⚠️ Disclaimer")
        st.info("""
        **Educational Use Only.**
        
        The financial models implemented (Black-Scholes, Heston, LSMC) are for simulation and research purposes. They **do not constitute financial advice**. 
        
        Market data is sourced from Yahoo Finance and may be delayed or subject to API rate limits.
        """)

    st.divider()

    # --- COMPREHENSIVE MODULE GUIDE ---
    st.subheader("📚 Module Guide")
    st.markdown("Select a category to learn about the specific algorithms included in this suite:")

    tab_fin, tab_sci, tab_opt, tab_data = st.tabs([
        "💰 Quantitative Finance", 
        "🔬 Science & Stats", 
        "⚡ Optimization & Speed",
        "📡 Real World"
    ])

    with tab_fin:
        st.markdown("#### Pricing & Derivatives")
        st.markdown("""
        | Module | Description | Key Concept |
        | :--- | :--- | :--- |
        | **1. Standard Monte Carlo** | The baseline method for European Option pricing. | *Geometric Brownian Motion (GBM)* |
        | **3. Least Squares MC (LSMC)** | Prices **American Options** (early exercise) using the *Longstaff-Schwartz* algorithm. | *Backward Induction & Regression* |
        | **9. Heston Model** | A sophisticated model where volatility itself is random. Accounts for "Fat Tails" and market crashes. | *Stochastic Volatility (SDEs)* |
        """)
        

    with tab_sci:
        st.markdown("#### Physics, Robotics & Bayesian Inference")
        st.markdown("""
        | Module | Description | Key Concept |
        | :--- | :--- | :--- |
        | **4. Markov Chain MC (MCMC)** | Used to find unknown parameters of a distribution. Essential for Bayesian Statistics. | *Metropolis-Hastings Algorithm* |
        | **5. Sequential MC (SMC)** | Also known as a **Particle Filter**. Used in Robotics to track location using noisy GPS data. | *Bayesian Filtering* |
        | **6. Quantum MC (VMC)** | A Variational method to approximate the ground state energy of a Quantum Harmonic Oscillator. | *Wavefunction Optimization* |
        """)
        

    with tab_opt:
        st.markdown("#### Variance Reduction & Acceleration")
        st.markdown("""
        | Module | Description | Key Concept |
        | :--- | :--- | :--- |
        | **2. Quasi-Monte Carlo (QMC)** | Uses deterministic **Sobol Sequences** to cover space more evenly than random numbers. | *Low-Discrepancy Sequences* |
        | **8. Variance Reduction** | Uses **Antithetic Variates** (pairing $Z$ and $-Z$) to mathematically cancel out error terms. | *Error Cancellation* |
        | **10. Comparison** | A benchmark suite comparing convergence rates ($O(1/N)$ vs $O(1/\sqrt{N})$). | *Computational Complexity* |
        """)
        

    with tab_data:
        st.markdown("#### Live Market Analysis")
        st.markdown("""
        | Module | Description | Key Concept |
        | :--- | :--- | :--- |
        | **11. Real-World Data** | Connects to **Yahoo Finance** to pull live stock data (e.g., NVDA, SPY). | *Live Data Pipelines* |
        | **Analysis** | Calculates historical volatility, drift, and simulates future price cones based on real market regimes. | *Data-Driven Simulation* |
        """)

    st.divider()
    
    # --- FOOTER & AUTHOR ---
    st.markdown("### 🎓 Recommended Learning Path")
    st.success("""
    1.  Start with **Standard Monte Carlo** to understand the basics of simulation.
    2.  Check **Comparison & Summary** to see why "Random" isn't always efficient.
    3.  Move to **Heston Model** to see how professionals model market risk.
    """)
    
    st.markdown("---")
    c_footer1, c_footer2 = st.columns([1, 3])
    with c_footer1:
        st.markdown("### 👨‍💻 Creator")
    with c_footer2:
        st.markdown("**Michael Fernandes**")
        st.caption("Quantitative Developer & Data Science Enthusiast")
        st.caption("Powered by Streamlit | Python")


def main():
    st.sidebar.title("Monte Carlo Lab 🧪")
    
    options = [
        "About This Project",
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

    if selection == "About This Project":   # <--- NEW LOGIC
        section_about()
    elif selection == "Standard Monte Carlo":
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
