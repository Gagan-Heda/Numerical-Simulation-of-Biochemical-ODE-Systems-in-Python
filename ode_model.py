from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class ParamsABC:
    """
    Parameters for the sequential reaction A -> B -> C.

    Attributes
    ----------
    k1_per_s: float
        Rate constant k1 [1/s] for A -> B.
    k2_per_s: float
        Rate constant k2 [1/s] for B -> C.
    tol_mass: float
        Tolerance for mass-conservation checks.
    tol_neg: float
        Tolerance for negativity checks.

    Notes
    -----
    Units: time [s], concentrations [arbitrary or µM], rates [1/s].
    """

    k1_per_s: float = 100.0
    k2_per_s: float = 0.1
    tol_mass: float = 1e-9
    tol_neg: float = 1e-12


def dxdt(state, params):
    """
    Compute the derivatives [dA/dt, dB/dt, dC/dt] for the sequential reaction A -> B -> C.

    Equations
    ---------
        dA/dt = -k1 * A
        dB/dt =  k1 * A - k2 * B
        dC/dt =  k2 * B

    Parameters
    ----------
    state : array-like, shape (3,)
        Current concentrations [A, B, C] in µM (or arbitrary units).
    params : ParamsABC
        Contains reaction rate constants k1_per_s and k2_per_s [1/s].

    Returns
    -------
    np.ndarray, shape (3,)
        Derivatives [dA/dt, dB/dt, dC/dt].

    Units
    -----
    A, B, C : µM (or arbitrary concentration units)
    t, dt   : seconds [s]
    k1, k2  : 1/seconds [1/s]
    """
    A, B, C = state
    dA = -params.k1_per_s * A
    dB = params.k1_per_s * A - params.k2_per_s * B
    dC = params.k2_per_s * B
    return np.array([dA, dB, dC])
    

def euler_step(state, params, dt):
    """
    Explicit Euler step for the sequential reaction A -> B -> C.

    Equations
    ---------
        new_state = state + dt * [dA/dt, dB/dt, dC/dt]

    Parameters
    ----------
    state : array-like, shape (3,)
        Current concentrations [A, B, C] in µM.
    params : ParamsABC
        Reaction parameters (rate constants [1/s]).
    dt : float
        Time step [s].

    Returns
    -------
    np.ndarray, shape (3,)
        Updated concentrations [A, B, C] at t + dt.

    Units
    -----
    A, B, C : µM
    dt      : seconds [s]
    k1, k2  : 1/seconds [1/s]
    """
    return state + dt * dxdt(state, params)


def simulate(x0, t_end_s, dt, params, checks=True):
    """
    Simulate the sequential reaction A -> B -> C using explicit Euler integration.
    Equations
    ---------
        dA/dt = -k1 * A
        dB/dt =  k1 * A - k2 * B
        dC/dt =  k2 * B

    Parameters
    ----------
    x0 : array-like, shape (3,)
        Initial concentrations [A0, B0, C0] in µM.
    t_end_s : float
        Final simulation time [s].
    dt : float
        Time step size [s].
    params : ParamsABC
        Reaction parameters (rate constants [1/s]).
    checks : bool, default=True
        If True, enforce:
          - Non-negativity check (no concentration < -tol_neg).
          - Mass conservation check (total mass ~ constant within tol_mass).
    Returns
    -------
    t : np.ndarray
        Time points [s].
    X : np.ndarray, shape (len(t), 3)
        Concentrations [A, B, C] at each time point.

    Units
    -----
    A, B, C : µM
    t, dt   : seconds [s]
    k1, k2  : 1/seconds [1/s]
    """
    t = np.arange(0, t_end_s + dt, dt)
    X = np.zeros((len(t), 3))
    X[0] = x0
    mass0 = np.sum(x0)

    for i in range(1, len(t)):
        X[i] = euler_step(X[i-1], params, dt)

        if checks:
            if np.any(X[i] < -params.tol_neg):
                raise ValueError(f"Negative concentration at t={t[i]}s: {X[i]}")
            if abs(np.sum(X[i]) - mass0) > params.tol_mass:
                raise ValueError(f"Mass not conserved at t={t[i]}s: total={np.sum(X[i])}, expected={mass0}")

    return t, X

# ---------- Reference (analytic) solutions ----------

def analytic_A(t, A0, k1):
    """
    Analytical solution for A(t) in the reaction A -> B -> C.

    Equation
    --------
    dA/dt = -k1 * A

    Solution
    --------
    A(t) = A0 * exp(-k1 * t)

    Units
    -----
    A, A0: concentration [µM]
    t: time [s]
    k1: rate constant [1/s]

    Returns
    -------
    float
        Concentration of A at time t [µM].
    """
    A_exact = A0 * math.exp(-k1 * t)
    return A_exact


def analytic_B(t, A0, k1, k2):
    """
    Analytical solution for B(t) in the reaction A -> B -> C.

    Equations
    ---------
    dA/dt = -k1 * A
    dB/dt = k1 * A - k2 * B

    Solution
    --------
    B(t) = (A0 * k1 / (k2 - k1)) * (exp(-k1 * t) - exp(-k2 * t))

    Units
    -----
    A0: initial concentration of A [µM]
    t: time [s]
    k1, k2: rate constants [1/s]

    Returns
    -------
    float
        Concentration of B at time t [µM].
    """
    B_exact = ((A0 * k1) / (k2 - k1)) * (math.exp(-k1 * t) - math.exp(-k2 * t))
    return B_exact


def analytic_C(t, A0, k1, k2):
    """
    Analytical solution for C(t) in the reaction A -> B -> C.

    Equations
    ---------
    dA/dt = -k1 * A
    dB/dt = k1 * A - k2 * B
    dC/dt = k2 * B

    Solution
    --------
    C(t) = A0 * (1 - (k2 / (k2 - k1)) * exp(-k1 * t) + (k1 / (k2 - k1)) * exp(-k2 * t))

    Units
    -----
    A0: initial concentration of A [µM]
    t: time [s]
    k1, k2: rate constants [1/s]

    Returns
    -------
    float
        Concentration of C at time t [µM].
    """
    C_exact = A0 * (1 - (k2 / (k2 - k1)) * math.exp(-k1 * t) + (k1 / (k2 - k1)) * math.exp(-k2 * t))
    return C_exact


    