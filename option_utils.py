import numpy as np
# from scipy.stats import norm
from scipy.special import ndtr
from scipy.optimize import brentq
import pandas as pd

#garman kohlanger
def d1(F, K,sigma,tau):
    return (np.log(F/K) + 0.5*sigma**2*tau)/(sigma*np.sqrt(tau))

def d2(F, K,sigma,tau):
    return d1(F, K,sigma,tau) - sigma*np.sqrt(tau)

def gk_call(F, K,rd,sigma,tau):
    d_1 = d1(F, K,sigma,tau)
    d_2 = d_1-sigma*np.sqrt(tau)
    return np.exp(-rd*tau)*(F*ndtr(d_1) - K*ndtr(d_2))

def gk_put(F, K,rd,sigma,tau):
    d_1 = d1(F, K,sigma,tau)
    d_2 = d_1-sigma*np.sqrt(tau)
    return np.exp(-rd*tau)*(K*ndtr(-d_2) - F*ndtr(-d_1))

def fwd_delta_call(F, K,sigma,tau):
    return ndtr(d1(F, K,sigma,tau))

def frw_delta_put(F, K,sigma,tau):
    return ndtr(d1(F, K,sigma,tau))-1


#reiswich wystup
def spot_delta(F,K,rf,sigma,tau,phi):
    #phi=1 for call, phi=-1 for put
    d_1 = d1(F, K, sigma, tau)
    return phi*np.exp(-rf*tau)*ndtr(phi*d_1)

def atm_dns_strike(F,sigma_atm,tau):
    #ATM delta-neutral straddle strike
    #DNS: d1=0
    return F*np.exp(0.5*sigma_atm**2*tau) 

def delta_atm_spot(F,rf,sigma_atm,tau): #later used in theorem 1
    #ATM call spot delta (phi=1) under DNS convention
    K_atm=atm_dns_strike(F,sigma_atm,tau)
    return spot_delta(F,K_atm,rf,sigma_atm,tau,1)

#R&W theorem 1
def rw_a_spot(rf, tau):
    #a=call delta-put delta
    return np.exp(-rf*tau)

def rw_coefficients_spot(F,sigma_atm,rr,sigma_smile,rf,tau,delta_bar=0.25):
    delta_atm = delta_atm_spot(F, rf, sigma_atm, tau)
    a = rw_a_spot(rf, tau)
    # Common positive call-delta axis
    delta_call = delta_bar
    delta_put = a - delta_bar 
    x_call = delta_call - delta_atm
    x_put = delta_put - delta_atm
    b0 = sigma_smile + rr / 2
    b1 = sigma_smile - rr / 2
    det = x_call * x_put**2 - x_put * x_call**2
    c1 = (b0*x_put**2 - b1*x_call**2) / det
    c2 = (x_call*b1 - x_put*b0) / det

    return c1, c2, delta_atm, a


def rw_smile_delta_spot(F, delta, sigma_atm, rr,sigma_smile, rf,tau, delta_quote = 0.25):
    c1,c2,delta_atm,a=rw_coefficients_spot(F, sigma_atm, rr, sigma_smile, rf, tau, delta_quote)
    return sigma_atm + c1*(delta-delta_atm) + c2*(delta-delta_atm)**2
#define market strangle vol = atm vol+quoted strangle aka bf
def rw_market_strangle_vol(sigma_atm,bf):
    return sigma_atm+bf #equation 4


def strike_from_spot_delta(F,rf,sigma,tau,delta_target,phi):
    """find strike K s.t. the non-premiunm -adjusted pot delta euqalts delta target"""
    def objective(K):
        return spot_delta(F,K,rf,sigma,tau,phi)-delta_target
    return brentq(objective, F*0.2, F*5)

def rw_market_strangle_price(F,rd,sigma_atm,br,rf,tau,delta_bar=0.25):
    sigma_ms=rw_market_strangle_vol(sigma_atm,br)
    K_call=strike_from_spot_delta(F,rf,sigma_ms,tau,delta_bar,1)
    K_put=strike_from_spot_delta(F,rf,sigma_ms,tau,-delta_bar,-1)
    strangle_price=(gk_call(F,K_call,rd,sigma_ms,tau)+gk_put(F,K_put,rd,sigma_ms,tau)) 
    return strangle_price, K_call, K_put, sigma_ms

def rw_vol_for_strike(
    F, K, sigma_atm, rr, sigma_smile, rf, tau,
    delta_quote=0.25
):
    def objective(sigma):
        delta = spot_delta(F, K, rf, sigma, tau, 1)

        smile_vol = rw_smile_delta_spot(F, delta, sigma_atm, rr, sigma_smile, rf, tau, delta_quote)
        return sigma - smile_vol

    lower = 1e-4
    upper = 5.0
    f_lower = objective(lower)
    f_upper = objective(upper)
    if f_lower * f_upper > 0:
        raise ValueError(
            "INNER_VOL_ROOT: "
            f"K={K}, sigma_smile={sigma_smile}, "
            f"f_lower={f_lower}, f_upper={f_upper}"
        )
    return brentq(objective, lower, upper, xtol=1e-8, rtol=1e-10)

def calibrate_smile_strangle(F, rd, rf, sigma_atm, rr, bf, tau,delta_bar=0.25):
    market_price, K_call, K_put, sigma_ms = rw_market_strangle_price(F, rd, sigma_atm, bf, rf, tau, delta_bar)

    def objective(sigma_smile):
        vol_call = rw_vol_for_strike(F, K_call, sigma_atm, rr,sigma_smile, rf, tau, delta_bar)
        vol_put = rw_vol_for_strike(F, K_put, sigma_atm, rr,sigma_smile, rf, tau, delta_bar)

        model_price = (gk_call(F, K_call, rd, vol_call, tau)+ gk_put(F, K_put, rd, vol_put, tau))
        return model_price - market_price
    candidates = np.linspace(
        1e-8,
        max(0.05, bf + 0.03),
        80
    )
    valid_points = []
    for candidate in candidates:
        try:
            value = objective(candidate)

            if np.isfinite(value):
                valid_points.append((candidate, value))
        except ValueError:
            continue
    bracket = None
    for (x1, f1), (x2, f2) in zip(valid_points[:-1],valid_points[1:]):
        if f1 == 0:
            bracket = (x1, x1)
            break
        if f1 * f2 < 0:
            bracket = (x1, x2)
            break
    if bracket is None:
        raise ValueError("OUTER_SMILE_ROOT: no feasible sign-changing bracket; "
            f"ATM={sigma_atm}, RR={rr}, BF={bf}")
    if bracket[0] == bracket[1]:
        sigma_smile_calibrated = bracket[0]
    else:
        sigma_smile_calibrated = brentq(objective,bracket[0],bracket[1])
    return (sigma_smile_calibrated,market_price,K_call,K_put,sigma_ms)


#breeden litzenberger
def recover_rnd(F,rd,rf,sigma_atm,rr,bf,tau,grid_low=0.5,grid_high=1.3,n_grid=801):
    sigma_s,market_price,Kc,Kp,sigma_ms=calibrate_smile_strangle(F,rd,rf,sigma_atm,rr,bf,tau)
    K_grid = np.linspace(F*grid_low, F*grid_high, n_grid)
    vol_grid = np.array([rw_vol_for_strike(F,K,sigma_atm, rr,sigma_s, rf,tau) for K in K_grid])
    call_grid = np.array([gk_call(F, K,rd,vol,tau) for K, vol in zip(K_grid, vol_grid)])
    dC_dK=np.gradient(call_grid, K_grid)
    d2C_dK2=np.gradient(dC_dK, K_grid)
    rnd=np.exp(rd*tau)*d2C_dK2
    return {"K": K_grid,"vol": vol_grid,"call": call_grid,"rnd": rnd,
        "sigma_s": sigma_s,"market_price": market_price,"Kc": Kc,"Kp": Kp,"sigma_ms": sigma_ms}

from scipy.integrate import cumulative_trapezoid
#tail risk measure
def rnd_tail_measure(out, F):
    K=np.asarray(out["K"],dtype=float)
    raw_rnd = np.asarray(out['rnd'],dtype=float)
    raw_min_density = raw_rnd.min()
    rnd=np.maximum(raw_rnd,0.0)
    # Calculate tail risk measures here
    area=np.trapz(rnd, K)
    rnd = rnd/area #normalise density
    x = np.log(K/F) #ensure assymetry around K
    left_tail_5=np.trapz(rnd[K<=F*0.95], K[K<=F*0.95])
    left_tail_10=np.trapz(rnd[K<=F*0.90], K[K<=F*0.90])
    log_mean = np.trapz(x*rnd, K)
    log_var = np.trapz((x-log_mean)**2*rnd, K)
    log_std = np.sqrt(log_var)
    log_skew = np.trapz(((x-log_mean)/log_std)**3*rnd, K)
    log_kurt = np.trapz(((x-log_mean)/log_std)**4*rnd, K)

    cdf = cumulative_trapezoid(rnd, K, initial =0.0) #integrate pdf to obtain cdf (no gaussioan assumptions, nothing)
    cdf = cdf/cdf[-1] #rescales the cds so the final value is 1
    q01=np.interp(0.01,cdf,K) # finds strike level where probability reaches 0.01 (S<q_0.01)=0.01
    q05=np.interp(0.05,cdf,K) #so it is basically returns a quantile value
    #cummulative integral needed for 
    cum_first = cumulative_trapezoid(K*rnd, K,initial=0) #first cumulative moment
    es01_level=np.interp(q01,K,cum_first)/0.01
    es05_level=np.interp(q05, K, cum_first)/0.05
    VaR01=1-q01/F
    VaR05=1-q05/F
    ES01=1-es01_level/F
    ES05 = 1-es05_level/F
    return {
        "area": area, "mean":log_mean, 'std':log_std, "skew":log_skew,'kurt':log_kurt, 'excess_kurt':log_kurt - 3, 
        "min_density":raw_min_density,'q01':q01,'q05':q05,'VaR01':VaR01,'VaR05':VaR05,'ES01':ES01,'ES05':ES05,
        'ProbBelow95F':left_tail_5,'ProbBelow90F':left_tail_10}

def no_arbitrage_checks(out, tol_density = -1e-7, tol_area = 0.02,tol_convex = -1e-7):
    K=out["K"]
    vol = out["vol"]
    call = out["call"]
    rnd=out["rnd"]
    dC_dK = np.gradient(call, K)
    d2C_dK2 = np.gradient(dC_dK, K)
    area=np.trapz(rnd, K)
    checks={"vol_positive":np.all(vol>0),
            "call_decreasing":np.all(np.diff(call)<=1e-8),
            "call_convex":np.all(d2C_dK2>=tol_convex),
            "rnd_positive":np.min(rnd)>=tol_density,
            "density_area_ok":np.abs(area-1)<=tol_area,
            "area":area,
            "min_density":np.min(rnd),
            "min_call_second_derivative":np.min(d2C_dK2)}
    checks["passes_no_arb"]=(
        checks["vol_positive"] and checks["call_decreasing"] and checks["call_convex"] and checks["rnd_positive"] 
        and checks["density_area_ok"]
    )
    if checks["passes_no_arb"]:
        quality = "PASS"
    elif checks["area"] > 0.98 and checks["area"] < 1.02:
        quality = "WARNING"
    else:
        quality = "FAIL"
    checks["quality"] = quality
    return checks

# frunciton taht directly retirns the left-tail probabilities from the first derivaties of call price
def recover_left_tails(F, rd, rf, sigma_atm, rr, bf, tau, delta_bar = 0.25, n_grid =101, 
                       grid_low=0.8, grid_high = 1.02):
    sigma_s, market_price, Kc, Kp, sigma_ms = calibrate_smile_strangle(F, rd, rf, sigma_atm, 
                                                                       rr, bf, tau,delta_bar)
    K_grid = np.linspace(grid_low*F, grid_high*F, n_grid)
    vol_grid = np.array([rw_vol_for_strike(F, K, sigma_atm, rr, sigma_s, rf, tau,delta_bar)
                          for K in K_grid])
    call_grid = np.array([gk_call(F, K, rd, vol, tau) for K, vol in zip(K_grid, vol_grid)])
    dC_dK = np.gradient(call_grid, K_grid)
    cdf_grid = 1+np.exp(rd*tau)*dC_dK
    cdf_grid = np.clip(cdf_grid, 0, 1)
    left_tail_5=np.interp(0.95*F, K_grid, cdf_grid)
    left_tail_10=np.interp(0.90*F, K_grid, cdf_grid)
    return {
        "left_tail_5": left_tail_5,
        "left_tail_10": left_tail_10,
        "sigma_s":sigma_s
    }

def recover_left_tails_fast(F,rd,rf,sigma_atm,rr,bf,tau,delta_bar=0.25,relative_step=1e-4):
    sigma_s, market_price, Kc, Kp, sigma_ms = calibrate_smile_strangle(
        F, rd, rf, sigma_atm,rr, bf, tau, delta_bar)

    def cdf_at_strike(K):
        h = relative_step * F
        K_minus = K - h
        K_plus = K + h
        vol_minus = rw_vol_for_strike(F, K_minus, sigma_atm, rr,sigma_s, rf, tau, delta_bar)
        vol_plus = rw_vol_for_strike(F, K_plus, sigma_atm, rr,sigma_s, rf, tau, delta_bar)
        call_minus = gk_call(F, K_minus, rd, vol_minus, tau)
        call_plus = gk_call(F, K_plus, rd, vol_plus, tau)
        dC_dK = (call_plus - call_minus) / (2 * h)
        probability = 1 + np.exp(rd * tau) * dC_dK
        return float(np.clip(probability, 0, 1))
    return {"left_tail_5": cdf_at_strike(0.95 * F),"left_tail_10": cdf_at_strike(0.90 * F),
        "sigma_s": sigma_s,"Kc": Kc,"Kp": Kp}

def implied_log_tail_measure(moneyness, rnd):
    m = np.asarray(moneyness, dtype=float)
    density = np.asarray(rnd, dtype=float)
    #valid observations
    valid = (np.isfinite(m) & np.isfinite(density) & (m>0))
    m = m[valid]
    density = density[valid]
    #order from lowest FX outcome to highest
    order = np.argsort(m)
    m = m[order]
    density = density[order]
    density = density/np.trapz(density,m) #integral = 1
    cdf = cumulative_trapezoid(density, m, initial=0)
    q01_m = np.interp(0.01,cdf,m) #moneynes quantiles
    q05_m = np.interp(0.05, cdf,m)
    q99_m=np.interp(0.99,cdf,m)
    q95_m=np.interp(0.95,cdf,m)
    #transform to log forward-relative log return
    q01_log = np.log(q01_m)
    q05_log = np.log(q05_m)
    q99_log = np.log(q99_m)
    q95_log = np.log(q95_m)
    VaR01_L_log = -q01_log
    VaR05_L_log = -q05_log
    VaR01_R_log = q99_log
    VaR05_R_log = q95_log
    #log return corresponding to every grid point:
    x = np.log(m)
    #cumulative integral
    cumulative_tail_mean = cumulative_trapezoid(x*density,m,initial=0)
    cumulative_log_expectation = cumulative_trapezoid(x*density, m, initial = 0)
    total_log_expectation = cumulative_log_expectation[-1]    
    tail01 = np.interp(q01_m,m,cumulative_log_expectation) #weighted sum contained in 1%/5% quantiles
    tail05 = np.interp(q05_m,m,cumulative_log_expectation)
    tail99 = (total_log_expectation - np.interp(q99_m,m, cumulative_log_expectation))
    tail95 = (total_log_expectation - np.interp(q95_m,m,cumulative_log_expectation))
    #convert weighted sum to conditional tail averages
    ES01_L_log = -(tail01/0.01)
    ES05_L_log = -(tail05/0.05)
    ES01_R_log = tail99/0.01
    ES05_R_log = tail95/0.05
    return (VaR01_L_log, VaR05_L_log, VaR01_R_log, VaR05_R_log, ES01_L_log, ES05_L_log, 
            ES01_R_log, ES05_R_log)

def empirical_risk_measures(x):
    x = x.dropna()
    q01=x.quantile(0.01)
    q05=x.quantile(0.05)
    q99 = x.quantile(0.99)
    q95 = x.quantile(0.95)
    return pd.Series({'VaR01_L_P':-q01,'VaR05_L_P':-q05,'VaR01_R_P': q99, 'VaR05_R_P': q95, 
                      'ES01_L_P':-x[x<=q01].mean(), 'ES05_L_P':-x[x<=q05].mean(),
                        'ES01_R_P': x[x>=q99].mean(), 'ES05_R_P':x[x>=q95].mean(),
                          'skew_P':x.skew(),'excess_kurt_P':x.kurt(),'n_obs':len(x)})