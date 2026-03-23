# Numerical Simulation of Biochemical ODE Systems in Python

Python implementation of numerical and analytic solutions for a sequential biochemical reaction system (A → B → C) using Euler integration with validation checks.

---

## Project Overview

This project simulates the time evolution of concentrations in a sequential biochemical reaction using ordinary differential equations (ODEs). The system models a simple reaction pathway:

A → B → C

The workflow provides both numerical integration and analytic solutions for validation.

**Goals:**

- Implement Euler’s method for numerical integration  
- Ensure simulation correctness via physical constraints  
- Compare numerical results to exact analytic solutions  

---

## Key Features

### ODE System Implementation
- Computes derivatives with `dxdt(state, params)`  
- Encodes reaction kinetics for the A → B → C pathway  

### Euler Integration
- Implements `euler_step()` for iterative time-stepping  
- Builds a full numerical simulator  

### Simulation Engine
- Simulates concentration trajectories over time  
- Returns time grid and concentration matrix  

### Validation Checks
- Prevents negative concentrations  
- Verifies conservation of total mass (A + B + C)  
- Raises errors if physical constraints are violated  

### Analytical Solutions
- Closed-form solutions for:
  - A(t)
  - B(t)
  - C(t)  

- Enables direct comparison between numerical and exact solutions  

---

## Mathematical Model

For initial conditions:

A(0) = A0, B(0) = 0, C(0) = 0

Exact solutions:

- A(t) = A0 * exp(-k1 * t)  
- B(t) = A0 * k1/(k2 - k1) * (exp(-k1*t) - exp(-k2*t))  
- C(t) = A0 - A(t) - B(t)  

ODE system:

- dA/dt = -k1 * A  
- dB/dt = k1 * A - k2 * B  
- dC/dt = k2 * B  

---

## Technologies Used

- Python  
- NumPy  
- Dataclasses  

---

## File Structure

- `src/ode_models.py`  
  - `dxdt` – computes derivatives  
  - `euler_step` – performs a single Euler integration step  
  - `simulate` – runs the full numerical simulation  
  - `analytic_A`, `analytic_B`, `analytic_C` – analytic solutions  

---

## Example Usage

```python
from src.ode_models import simulate, dxdt, analytic_A, analytic_B, analytic_C

x0 = [1.0, 0.0, 0.0]
t_end = 0.05
dt = 0.01
params = {"k1": 1.0, "k2": 0.5}

t, X = simulate(x0, t_end, dt, params)

# Compare with analytic solutions
A_exact = [analytic_A(ti, x0[0], params) for ti in t]
B_exact = [analytic_B(ti, x0[0], params) for ti in t]
C_exact = [analytic_C(ti, x0[0], params) for ti in t]
