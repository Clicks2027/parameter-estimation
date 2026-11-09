# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Regression: a reversible reaction in a batch reactor
#
# **CHME 5137, homework after Lectures 12 and 13 (October 20 and 23, 2026).** Section 5 of
# the Lecture 13 notebook started this. Here it is in full.
#
# *Written for 2026 by Claude Code (Claude Opus 5.5) from Prof. West's A ⇌ B fitting
# exercise of earlier years, and checked by Prof. West.*
#
# ---
#
# You run a reversible, first-order reaction in a batch reactor:
# $$\mathrm{A} \rightleftharpoons \mathrm{B}, \qquad
# \frac{dC_A}{dt} = k_r C_B - k_f C_A, \qquad C_B = C_{A,0} - C_A$$
# Every run starts with $C_{A,0} = 10$ mol/L and no B. You measured $C_A$ every 10 s for
# 100 s, at three temperatures.
#
# **Kinetics.** The forward rate constant follows the Arrhenius expression,
# $k_f = A \exp(-E_a/RT)$.
#
# **Thermodynamics.** At equilibrium the forward and reverse rates are equal, so the
# equilibrium constant is $K = k_f/k_r$. (Assume an ideal solution, so this
# concentration-based $K$ is the thermodynamic one.) From thermodynamics,
# $$\Delta G^\circ = -RT\ln K \qquad\text{and}\qquad \Delta G^\circ = \Delta H - T\,\Delta S$$
# A and B have very similar heat capacities, so take $\Delta H$ and $\Delta S$ to be
# independent of temperature. Then $k_r = k_f/K$, and the whole model has **four
# parameters**: $\log_{10} A$, $E_a$, $\Delta H$ and $\Delta S$.
#
# Before the experiments, from the literature and a quantum chemistry calculation, your
# estimates were $\log_{10}A \approx 6$ (A in s⁻¹), $E_a \approx 45$ kJ/mol,
# $\Delta H \approx -10$ kJ/mol and $\Delta S \approx -50$ J/mol/K. Let's see what the data say.
#
# ### How to work
#
# - **Commit as you go.** After each numbered task works, Restart & Run All, then commit
#   with a message that says what you did, and push. At least one commit per task.
#   **Commit the `.ipynb` and its paired `.py` together.**
# - **Answer the questions in words**, in the markdown cells marked *Answer*.
# - **Say what your ± means.** One standard error, or a 95% interval? Either is fine. Say which (Lecture 13).
# - **Let bugs shout.** If `solve_ivp` fails, `raise` an error with its `message`.
#   Don't let a half-finished solution reach `curve_fit` without anyone noticing.

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize
import scipy.stats

plt.rcParams['figure.figsize'] = [5, 3]
R = 8.314  # J/mol/K

# %% [markdown]
# The data, the same as in class:

# %%
cA0 = 10.0   # mol/L
times = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100.])   # s
cA_data = {   # mol/L, keyed by temperature in K
    298.15: np.array([8.649, 7.441, 7.141, 6.366, 6.215, 5.990, 5.852, 5.615, 5.481, 5.644]),
    308.15: np.array([7.230, 6.073, 5.452, 5.317, 5.121, 4.998, 4.951, 4.978, 5.015, 5.036]),
    323.15: np.array([5.137, 4.568, 4.548, 4.461, 4.382, 4.525, 4.483, 4.565, 4.459, 4.635]),
}
temperatures = np.array(list(cA_data))

for T, cA in cA_data.items():
    plt.plot(times, cA, 'o:', label=f'{T:.0f} K')
plt.plot(0, cA0, 'ko')
plt.xlabel('time (s)')
plt.ylabel('$C_A$ (mol/L)')
plt.ylim(0, None)
plt.legend()
plt.show()


# %% [markdown]
# ## 1. One temperature
#
# **(a) The model.** Write `cA_model(times, kf, kr)`, which solves the ODE with
# `scipy.integrate.solve_ivp` and returns $C_A$ at `times` (use `t_eval`). In Lecture 13 we fitted
# functions that were a formula. `curve_fit` doesn't care: any function that takes the
# independent variable first and then the parameters, and returns predictions, will do. Even
# one that runs a whole simulation.
#
# Then **test it** (Lecture 11). This ODE has an exact solution,
# $$C_A(t) = C_{A,\mathrm{eq}} + (C_{A,0} - C_{A,\mathrm{eq}})\,e^{-(k_f+k_r)t},
# \qquad C_{A,\mathrm{eq}} = C_{A,0}\frac{k_r}{k_f+k_r}$$
# Check that your function agrees with it, for some reasonable $k_f$ and $k_r$, with an
# `assert`. (We use `solve_ivp` anyway because your project's model won't have an exact
# solution.)

# %%
def cA_model(times, kf, kr):
    """C_A (mol/L) at each of `times` (s), for rate constants kf and kr (1/s)."""
    def rate(t, y):
        cA = y[0]
        cB = cA0 - cA
        return [kr * cB - kf * cA]

    solution = scipy.integrate.solve_ivp(
        rate,
        (0, max(times)),
        [cA0],
        t_eval=times,
    )
    if not solution.success:
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.y[0]


test_times = np.array([0.0, 10.0, 50.0, 100.0])
kf_test, kr_test = 0.04, 0.02
cA_eq = cA0 * kr_test / (kf_test + kr_test)
cA_exact = cA_eq + (cA0 - cA_eq) * np.exp(
    -(kf_test + kr_test) * test_times
)
assert np.allclose(
    cA_model(test_times, kf_test, kr_test), cA_exact, rtol=1e-3, atol=1e-8
)


# %% [markdown]
# **(b) Fit it at 298.15 K**, with `scipy.optimize.curve_fit`. **Try it without `p0` first.**
# It may fail with an error from deep inside SciPy, or it may appear to succeed. Either way,
# find out what happened: print and plot what came back, or what `cA_model` returns
# at `curve_fit`'s default starting point. Then give it a starting guess you can justify
# from the data. Roughly how long does $C_A$ take to get halfway to where it ends up?

# %%
print(cA_model(times, 1, 1))

popt_default, pcov_default = scipy.optimize.curve_fit(
    cA_model, times, cA_data[298.15]
)
print(popt_default)


# %%
popt, pcov = scipy.optimize.curve_fit(
    cA_model, times, cA_data[298.15], p0=[0.02, 0.02]
)
print(popt)

# %%

# %% [markdown]
# **(c) Look at it.** Plot the data and the fitted curve, and below it the residuals
# (data minus model). Report $k_f$ and $k_r$ with their uncertainties.
#
# What happened without `p0`, and why? What starting guess did you use, and how did you
# choose it?

# %%
perr = np.sqrt(np.diag(pcov))
t_fit = np.linspace(0, max(times), 200)
cA_fit = cA_model(t_fit, *popt)
residuals = cA_data[298.15] - cA_model(times, *popt)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(times, cA_data[298.15], 'o', label='data')
ax1.plot(t_fit, cA_fit, label='fit')
ax1.set_ylabel('$C_A$ (mol/L)')
ax1.legend()

ax2.plot(times, residuals, 'o')
ax2.axhline(0, color='k')
ax2.set_xlabel('time (s)')
ax2.set_ylabel('residual')
plt.tight_layout()
plt.show()

print(f"kf = {popt[0]:.5f} +/- {perr[0]:.5f} 1/s")
print(f"kr = {popt[1]:.5f} +/- {perr[1]:.5f} 1/s")


# %% [markdown] vscode={"languageId": "raw"}
# It takes C_A 19 seconds to get halfway to where it ends up. Without p0, curve_fit started with kf=kr=1 and the model predicted C_A was already about 5 mol/L at the first measurement time, 10 seconds. But the measured value there was 8.649 mol/L, so that starting guess made the reaction look much faster than the data suggests. I useda starting guess of p0 = 0.02, 0.02, rates at hundreths per second because the data changes over a few tens of seconds. The reported uncertainties (±) for (k_f) and (k_r) are one standard error.

# %%

# %% [markdown]
# ## 2. Three temperatures, in two steps
#
# **(a)** Fit $k_f$ and $k_r$ separately at each temperature, and make a small table of
# $k_f$, $k_r$ (each with its ±) and $K = k_f/k_r$.
#
# **Before fitting anything else:** from your $K$ values alone, is the reaction endothermic
# or exothermic? How can you tell?

# %%
fit_results = {}

for T in temperatures:
    popt_T, pcov_T = scipy.optimize.curve_fit(
        cA_model, times, cA_data[T], p0=[0.02, 0.02]
    )
    perr_T = np.sqrt(np.diag(pcov_T))
    K_T = popt_T[0] / popt_T[1]
    fit_results[T] = (popt_T, perr_T, K_T)
    print(
        f"{T:.2f} K: kf = {popt_T[0]:.5f} +/- {perr_T[0]:.5f} 1/s, "
        f"kr = {popt_T[1]:.5f} +/- {perr_T[1]:.5f} 1/s, K = {K_T:.4f}"
    )


# %% [markdown]
# K increases as temperature increases, so the reaction is endothermic.

# %% [markdown]
# **(b) Arrhenius.** As in Lecture 13: `scipy.stats.linregress` on $\ln k_f$ against
# $1/T$ gives $E_a$ from the slope and $\ln A$ from the intercept. Convert $\ln A$ to
# $\log_{10} A$.
#
# **(c) van 't Hoff.** Put the two expressions for $\Delta G^\circ$ together and divide by $-RT$:
# $$\ln K = -\frac{\Delta H}{R}\,\frac{1}{T} + \frac{\Delta S}{R}$$
# So $\ln K$ against $1/T$ is a straight line too, with slope $-\Delta H/R$ and intercept
# $\Delta S/R$. Fit it, and get $\Delta H$ and $\Delta S$, with their ±. Does the sign of
# $\Delta H$ agree with what you said in (a)?

# %%
kf_values = np.array([fit_results[T][0][0] for T in temperatures])
arrhenius_fit = scipy.stats.linregress(1 / temperatures, np.log(kf_values))

Ea = -arrhenius_fit.slope * R / 1000
log10A = arrhenius_fit.intercept / np.log(10)
Ea_se = arrhenius_fit.stderr * R / 1000
log10A_se = arrhenius_fit.intercept_stderr / np.log(10)

print(f"Ea = {Ea:.2f} +/- {Ea_se:.2f} kJ/mol")
print(f"log10(A) = {log10A:.2f} +/- {log10A_se:.2f}")

K_values = np.array([fit_results[T][2] for T in temperatures])
vant_hoff_fit = scipy.stats.linregress(1 / temperatures, np.log(K_values))

dH = -vant_hoff_fit.slope * R / 1000
dS = vant_hoff_fit.intercept * R
dH_se = vant_hoff_fit.stderr * R / 1000
dS_se = vant_hoff_fit.intercept_stderr * R

print(f"dH = {dH:.2f} +/- {dH_se:.2f} kJ/mol")
print(f"dS = {dS:.2f} +/- {dS_se:.2f} J/mol/K")


# %% [markdown]
# **(d)** `linregress` gave you a ± on each of these. What does that ± know about,
# and what has it left out? Think about how many points each straight line had, and about
# the ± you got in (a).

# %% [markdown]
# The +/- from lingress reflect how closely the three fitted points follow a straight line. They leave out the uncertainties in the fitted rate constants and equilibrium constants from 2a, because those values are treated as exact in the regressions. With only three temperatures, the standard errors are also based on very few points.

# %% [markdown]
# ## 3. All the data at once
#
# Step 2 fitted 3 × 2 rate constants, then fitted lines through three points each. Better:
# fit the four parameters you actually want **directly to all 30 measurements**, so that
# every measurement counts and the uncertainties carry all the way through.
#
# `curve_fit` wants one `xdata` and one `ydata`. With several experiments, stack them:
# `ydata` is all 30 concentrations in one array, and `xdata` carries **both** the
# temperature and the time of each one. `curve_fit` never looks inside `xdata`; it just passes it
# to your function. So it can be a tuple of two arrays:

# %%
T_all = np.repeat(temperatures, len(times))     # 298.15 ten times, then 308.15 ten times, ...
t_all = np.tile(times, len(temperatures))        # 10, 20, ..., 100, three times over
cA_all = np.concatenate([cA_data[T] for T in temperatures])
x_all = (T_all, t_all)
print(T_all[:12], t_all[:12], cA_all[:12], sep='\n')


# %% [markdown]
# **(a)** Write `cA_model_all(x, log10A, Ea, dH, dS)`. Unpack `x` into temperatures and
# times. For each temperature, work out $k_f$, $K$ and $k_r$, then call your `cA_model` for
# that temperature's times, and put the results in the right places of the output array.
# Solve one ODE per temperature, not one per data point. A boolean mask such as
# `T_col == T` picks out the rows.
#
# Notice the **units**: $E_a$ and $\Delta H$ in kJ/mol, $\Delta S$ in J/mol/K, and $A$ as
# $\log_{10}A$. That keeps all four parameters between about 1 and 100. An optimizer
# has an easier time when its parameters are all a similar size than when one is
# $10^{9}$ and another is $10^{-2}$. So convert carefully inside the function.
#
# Then fit it, **starting from your answers to task 2**, and report the four parameters with
# their ±.

# %%
def cA_model_all(x, log10A, Ea, dH, dS):
    """C_A (mol/L) for every (T, t) pair in x. Ea and dH in kJ/mol, dS in J/mol/K."""
    T_col, t_col = x
    cA = np.empty_like(t_col)

    for T in np.unique(T_col):
        mask = T_col == T
        kf = 10**log10A * np.exp(-Ea * 1000 / (R * T))
        K = np.exp(-(dH * 1000 - T * dS) / (R * T))
        kr = kf / K
        cA[mask] = cA_model(t_col[mask], kf, kr)

    return cA



# %%
p0 = [log10A, Ea, dH, dS]
popt, pcov = scipy.optimize.curve_fit(cA_model_all, x_all, cA_all, p0=p0)
perr = np.sqrt(np.diag(pcov))

print(f"log10(A) = {popt[0]:.2f} +/- {perr[0]:.2f}")
print(f"Ea = {popt[1]:.2f} +/- {perr[1]:.2f} kJ/mol")
print(f"dH = {popt[2]:.2f} +/- {perr[2]:.2f} kJ/mol")
print(f"dS = {popt[3]:.2f} +/- {perr[3]:.2f} J/mol/K")


# %% [markdown]
# **(b)** Now start from your estimates from before the experiments instead (6, 45, −10, −50).
# Does it find the same answer? How far did the data move you from what you expected?

# %%
p0_prior = [6, 45, -10, -50]
popt_prior, pcov_prior = scipy.optimize.curve_fit(
    cA_model_all, x_all, cA_all, p0=p0_prior
)
perr_prior = np.sqrt(np.diag(pcov_prior))

print(popt_prior)
print(popt_prior - popt)


# %% [markdown]
# **(c)** Compare with task 2. Are the centres the same? Are the error bars? Which set of
# error bars would you believe, and why?
#
# The centres are similar and overlap within one standard error. 
#
# The centres are similar and overlap within one standard error. Task 2 gives log10(A) = 9.23 +/- 0.16, Ea = 62.77 +/- 0.95 kJ/mol, dH = 11.97 +/- 0.74 kJ/mol, and dS = 38.72 +/- 2.38 J/mol/K. Task 3 gives 9.48 +/- 0.37, 64.28 +/- 2.14 kJ/mol, 11.53 +/- 0.79 kJ/mol, and 37.33 +/- 2.51 J/mol/K, respectively. Task 3's error bars are larger for log10(A), Ea, dH, and dS. I would trust Task 3's error bars more because it fits all 30 measurements at once; Task 2's regression uses only three points and does not include uncertainty in the fitted rate constants and K values.
#
# **(d) The correlation matrix.** The off-diagonal elements of `pcov` say how the errors in
# the parameters go together (Lecture 13). Turn the covariance into correlations:
# `pcov / np.outer(perr, perr)`. Which pairs are strongly correlated? Why those pairs?
# (Think about Lecture 13's ridge, and about the range of temperatures you measured over.)
# If someone quotes only your four ± values, what are they missing?

# %%
correlation = pcov / np.outer(perr, perr)
print(correlation)

# %% [markdown]
# log_10 A and Ea, and dH and dS, are strongly correlated. Over the limited temperature range, changes in each pair can compensate for one another. The four separate +/- values don’t show those relationships.

# %% [markdown]
# ## 4. Look at the residuals
#
# **Always look at the residuals** (Lecture 13). Plot the residuals of the task 3 fit against
# time, one colour per temperature, with a line at zero.
#
# - Do you see a pattern, or does it look like noise?
# - Is the scatter about the same width at all three temperatures? Least squares assumed it is.
# - Nobody told you the measurement uncertainty, so `curve_fit` (with
#   `absolute_sigma=False`, the default) estimated it from the scatter of the residuals.
#   Work out that estimate yourself:
#   $s = \sqrt{\sum_i r_i^2 / (N - p)}$, with $N$ data points and $p$ parameters. What does
#   it say about how precisely $C_A$ was measured?
# - Suppose the experimentalist now tells you their instrument is good to ±0.01 mol/L.
#   What would that tell you about your model, or about the experiment?

# %%
predicted_cA = cA_model_all(x_all, *popt)
residuals = cA_all - predicted_cA

for T in temperatures:
    mask = T_all == T
    plt.plot(t_all[mask], residuals[mask], 'o', label=f'{T:.0f} K')

plt.axhline(0, color='black', linestyle='--')
plt.xlabel('time (s)')
plt.ylabel('residual (mol/L)')
plt.legend()
plt.show()

s = np.sqrt(np.sum(residuals**2) / (len(cA_all) - len(popt)))
print(f's = {s:.3f} mol/L')

# %% [markdown]
# The residuals are scattered around zero without a clear pattern, although the 298 K data appears more spread out. The estimated residual standard deviation is 0.089 mol/L, much larger than the instrument's +/- 0.01 mol/L precision. This suggests model mismatch or other experimental variability.

# %% [markdown]
# ## 5. Your project
#
# One paragraph. Which parameters in your project's model will come from fitting data, and
# where will the data come from: your own simulations, the literature, a research group? What
# can you say about the errors in those data? Are they the same absolute size, the same
# relative size, or known point by point? And are any of your parameters likely to be
# correlated, the way $E_a$ and $A$ are here?

# %% [markdown]
# *Answer:*

# %% [markdown]
# ## Stretch problems (optional)
#
# **(a) Untangle the ridge.** Write $k_f$ in terms of its value at a reference temperature in
# the middle of your data, say $T_\mathrm{ref} = 310$ K:
# $$k_f(T) = k_\mathrm{ref}\exp\left[-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_\mathrm{ref}}\right)\right]$$
# It's the same model, written differently. Fit $\log_{10}k_\mathrm{ref}$, $E_a$, $\Delta H$
# and $\Delta S$, and look at the correlation matrix again. What changed, and why? Can you do
# the same for $\Delta H$ and $\Delta S$?
#
# **(b) Draw the posterior.** At 298.15 K only, put $k_f$ and $k_r$ on a 2-D grid and draw
# their posterior, as in Lecture 13, part 5, with a flat prior and Gaussian errors of width
# $s$ from task 4. Do its 95% credible intervals (summed over the other parameter) agree with
# `curve_fit`'s 95% confidence intervals? Why should they, or why not?

# %%

# %%

# %% [markdown]
# ---
#
# ### When you're done
#
# Restart & Run All. Check that every cell ran, in order, with no errors, and that the plots
# are there. Commit and push. Then check on GitHub that your fork shows your latest commit.
