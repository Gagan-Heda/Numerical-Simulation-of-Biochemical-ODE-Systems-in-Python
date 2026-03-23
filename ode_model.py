from __future__ import annotations
from dataclasses import dataclass
import numpy as np


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
    Units: time [s], concentrations [µM or arbitrary], rates [1/s].
    """

    k1_per_s: float = 100.0
    k2_per_s: float = 0.1
    tol_mass: float = 1e-9
    tol_neg: float = 1e-12


def dxdt(state, params):
    """
    Compute the derivative [dA/dt, dB/dt, dC/dt] for the sequential reaction A -> B -> C.

    Equations:
        dA/dt = -k1 * A
        dB/dt =  k1 * A - k2 * B
        dC/dt =  k2 * B

    Units:
        - A, B, C [µM or arbitrary concentration]
        - t, dt [s]
        - k1, k2 [1/s]

    Parameters
    ----------
    state : np.ndarray
        Vector [A, B, C] of concentrations.
    params : ParamsABC
        Contains k1_per_s, k2_per_s.

    Returns
    -------
    np.ndarray
        Vector [dA/dt, dB/dt, dC/dt], units [concentration/s].
    """
    A, B, C = state
    k1, k2 = params.k1_per_s, params.k2_per_s
    dA = -k1 * A
    dB = k1 * A - k2 * B
    dC = k2 * B
    return np.array([dA, dB, dC])


def euler_step(state, params, dt):
    """
    Take one explicit Euler step for A -> B -> C.

    Formula:
        state(t + dt) = state(t) + dt * [dA/dt, dB/dt, dC/dt]

    Units:
        - A, B, C [µM or arbitrary concentration]
        - t, dt [s]
        - k1, k2 [1/s]

    Parameters
    ----------
    state : np.ndarray
        Vector [A, B, C] of concentrations.
    params : ParamsABC
        Reaction parameters.
    dt : float
        Step size [s].

    Returns
    -------
    np.ndarray
        Updated concentrations after one step.
    """
    return state + dt * dxdt(state, params)


def simulate(x0, t_end_s, dt_s, params, checks=True, clip_negative=False):
    """
    Simulate A -> B -> C with Euler integration.

    Performs checks if enabled:
        - Non-negativity: raises ValueError if any concentration < -tol_neg
        - Mass conservation: raises ValueError if |(A+B+C) - (A0+B0+C0)| > tol_mass

    Units:
        - A, B, C [µM or arbitrary concentration]
        - t, dt [s]
        - k1, k2 [1/s]

    Parameters
    ----------
    x0 : np.ndarray
        Initial concentrations [A0, B0, C0].
    t_end_s : float
        Final integration time [s].
    dt_s : float
        Step size [s].
    params : ParamsABC
        Model parameters (k1, k2, tolerances).
    checks : bool, default=True
        If True, perform the checks listed above.
    clip_negative : bool, default=False
        If True, clip concentrations < 0 to 0 instead of raising.

    Returns
    -------
    t : np.ndarray
        Time grid from 0 to t_end_s [s].
    X : np.ndarray
        Concentrations at all times, shape (len(t), 3).
    """
    t = np.arange(0, t_end_s + dt_s, dt_s)
    X = np.zeros((len(t), 3))
    X[0] = x0
    total0 = np.sum(x0)

    for i in range(1, len(t)):
        X[i] = euler_step(X[i - 1], params, dt_s)

        if checks:
            if np.any(X[i] < -params.tol_neg):
                if clip_negative:
                    X[i] = np.maximum(X[i], 0)
                else:
                    raise ValueError(f"Negative concentration at step {i}: {X[i]}")
            if abs(np.sum(X[i]) - total0) > params.tol_mass:
                raise ValueError(f"Mass not conserved at step {i}: {X[i]}")

    return t, X


# ---------- Reference (analytic) solutions ----------

def analytic_A(t, A0, k1):
    """
    Analytical solution for A(t).

    Equation:
        A(t) = A0 * exp(-k1 * t)

    Units:
        - A0 [µM]
        - k1 [1/s]
        - t [s]
    """
    return A0 * np.exp(-k1 * t)


def analytic_B(t, A0, k1, k2):
    """
    Analytical solution for B(t) with B0 = 0.

    Equation (for k1 != k2):
        B(t) = A0 * k1/(k2 - k1) * (exp(-k1*t) - exp(-k2*t))

    Special case:
        If k1 == k2: B(t) = A0 * k1 * t * exp(-k1*t)

    Units:
        - A0 [µM]
        - k1, k2 [1/s]
        - t [s]
    """
    if k1 == k2:
        return A0 * k1 * t * np.exp(-k1 * t)
    return A0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))


def analytic_C(t, A0, k1, k2):
    """
    Analytical solution for C(t) with C0 = 0.

    Equation:
        C(t) = A0 - A(t) - B(t)

    Units:
        - A0 [µM]
        - k1, k2 [1/s]
        - t [s]
    """
    return A0 - analytic_A(t, A0, k1) - analytic_B(t, A0, k1, k2)