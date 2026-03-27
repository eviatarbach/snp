import numpy as np
#import properscoring
import scipy
import scoringrules
import scipy.stats
import scipy.optimize
import jax
jax.config.update('jax_platform_name', 'cpu')
from tqdm import tqdm

scoringrules.backends.set_active("jax")

n_years = 29
n_ens = 25

seas5 = np.fromfile(open("SEAS5_NAO_Nov1981-2009_Rea5m.ascii", 'r'), sep=' ', count=n_years*(n_ens + 2)).reshape(n_years, -1)
years = seas5[:, 0].astype(int)
seas5 = seas5[:, 1:-1]  # Exclude year column (first) and ensemble mean column (last)

era5 = np.fromfile(open("ERA5_NAO_DJF1981-2009.ascii", 'r'), sep=' ', count=n_years*2).reshape(n_years, -1)
era5 = era5[:, 1:].flatten()

varY = era5.var()
era5_norm = era5/np.sqrt(varY)
seas5_norm = seas5/np.sqrt(varY)

era5_bin = (era5_norm > 0).astype(int)
seas5_bin = (seas5_norm > 0).mean(axis=1)

# seas5_mean = seas5_norm.mean(axis=1)
# varX = seas5_norm.mean(axis=1).var()
# varXmY = (era5_norm - seas5_norm.mean(axis=1)).var()
# rho = np.corrcoef(seas5_mean, era5_norm)[1, 0]
# rho_f = np.sqrt(varX/(varX + varXmY))

# print("Var(X) =", varX)
# print("Var(X - Y) =", varXmY)
# print("sigma^2 =", (seas5_norm - seas5_mean.reshape(-1, 1)).var())
# print("rho^2 =", rho**2)
# print("rho =", rho)
# print("RPC =", rho/rho_f)

def normalize(forecasts, validation):
    varY = validation.var()
    forecasts_norm = (forecasts - forecasts.mean())/np.sqrt(varY)
    validation_norm = (validation - validation.mean())/np.sqrt(varY)
    return forecasts_norm, validation_norm

def bootstrap(f, forecasts, validation, n, **kwargs):
    n_t = len(validation)
    bootstrap_idx = np.random.choice(np.arange(n_t), (n_t, n))
    return np.array([f(forecasts[bootstrap_idx[:, i], :], validation[bootstrap_idx[:, i]], **kwargs) for i in tqdm(range(n))])

def bootstrap_bin(f, forecasts, validation, n, **kwargs):
    n_t = len(validation)
    bootstrap_idx = np.random.choice(np.arange(n_t), (n_t, n))
    return np.array([f(forecasts[bootstrap_idx[:, i]], validation[bootstrap_idx[:, i]], **kwargs) for i in tqdm(range(n))])

def classic_RPC(forecasts, validation):
    fcst_mean = forecasts.mean(axis=1)

    varX = forecasts.mean(axis=1).var()
    varXmY = (validation - forecasts.mean(axis=1)).var()
    rho = np.corrcoef(fcst_mean, validation)[1, 0]
    rho_f = np.sqrt(varX/(varX + varXmY))

    return rho/rho_f

#crps_mo = properscoring.crps_ensemble(era5_norm, seas5_norm).mean()
#crps_mm = properscoring.crps_ensemble(seas5_mean, seas5_norm).mean()

def expected_score(mu, nu, scoring_rule):
    return np.mean([scoring_rule(y, mu) for y in nu])

def expected_score_bin(mu, nu, scoring_rule):
    return nu*scoring_rule(1.0, mu) + (1 - nu)*scoring_rule(0.0, mu)

def entropy(mu, scoring_rule):
    return expected_score(mu, mu, scoring_rule)

def entropy_bin(mu, scoring_rule):
    return expected_score_bin(mu, mu, scoring_rule)

def divergence(mu, nu, scoring_rule):
    return expected_score(mu, nu, scoring_rule) - entropy(nu, scoring_rule)

def divergence_bin(mu, nu, scoring_rule):
    return expected_score_bin(mu, nu, scoring_rule) - entropy_bin(nu, scoring_rule)

def shift_mean(forecasts, a, b):
    return (forecasts - forecasts.mean(axis=1).reshape(-1, 1)) + a + b*forecasts.mean(axis=1).reshape(-1, 1)

def score_SSCrat(forecasts, validation, scoring_rule):
    f_bar = forecasts.flatten()
    entropy_fbar = entropy(f_bar, scoring_rule=scoring_rule)
    ssc_f = np.mean([entropy(forecasts[i, :], scoring_rule=scoring_rule) for i in range(len(validation))])/entropy_fbar

    score_pi = lambda ab: sum(scoring_rule(validation, shift_mean(forecasts, ab[0], ab[1])))
    #print(score_pi([0.0, 1.0]))
    score_pi_grad = jax.grad(score_pi, argnums=0)

    a, b = scipy.optimize.minimize(score_pi, [0.0, 1.0], method='BFGS', jac=score_pi_grad).x

    pi = shift_mean(forecasts, a, b)

    pi_bar = pi.flatten()
    entropy_pi_bar = entropy(pi_bar, scoring_rule=scoring_rule)

    ssc_pi = np.mean([entropy(pi[i, :], scoring_rule=scoring_rule) for i in range(len(validation))])/entropy_pi_bar
    #ssc_pi = np.mean([scoring_rule(validation[i], pi[i, :]) for i in range(len(validation))])/entropy_pi_bar
    SSCrat = ssc_f/ssc_pi

    return ssc_f, ssc_pi

def score_SSCrat_bin(forecasts, validation, scoring_rule):
    f_bar = forecasts.mean()
    entropy_fbar = entropy_bin(f_bar, scoring_rule=scoring_rule)

    ssc_f = np.mean([entropy_bin(f, scoring_rule=scoring_rule) for f in forecasts])/entropy_fbar

    score_pi = lambda ab: sum(scoring_rule(validation, jax.scipy.special.expit(ab[0] + ab[1]*jax.scipy.special.logit(forecasts))))
    score_pi_grad = jax.grad(score_pi, argnums=0)

    a, b = scipy.optimize.minimize(score_pi, [0.0, 1.0], method='BFGS', jac=score_pi_grad).x

    pi = jax.scipy.special.expit(a + b*jax.scipy.special.logit(forecasts))
    pi_bar = pi.mean()
    entropy_pi_bar = entropy_bin(pi_bar, scoring_rule=scoring_rule)

    ssc_pi = np.mean([entropy_bin(f, scoring_rule=scoring_rule) for f in pi])/entropy_pi_bar

    SSCrat = ssc_f/ssc_pi

    return ssc_f, ssc_pi