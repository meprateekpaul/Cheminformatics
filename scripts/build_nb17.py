import os
import sys
from nb_helper import create_notebook, add_md, add_code, save_notebooks

def build_nb17():
    nb = create_notebook()

    add_md(nb, """# Notebook 17:\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prateekpaul/Cheminformatics/blob/main/notebooks/nb17_scoring_functions_and_sampling.ipynb)\n\n Scoring Functions and Conformational Sampling

**Part F: Structure-Based Drug Design (SBDD)**  
**Estimated Time:** ~120 min  
**Level:** Advanced (●●●)

---

## 🎯 Learning Objectives
1. **Thermodynamics of Protein-Ligand Binding:** Relate binding free energy $\Delta G^\circ$ to the inhibition constant $K_i$ and understand the Cheng-Prusoff conversion from $IC_{50}$.
2. **Three Classes of Scoring Functions:** Classify and compare **physics-based** force fields, **empirical** regressions, and **knowledge-based** potentials of mean force (PMF).
3. **AutoDock & Vina Functional Decomposition:** Dissect empirical terms: van der Waals, directional hydrogen bonds, Coulomb electrostatics ($332.0$ conversion constant), desolvation, and torsional penalties.
4. **Glide-Style Scoring Terms:** Formulate piecewise geometric penalty functions for hydrogen-bond distances $g(\Delta r)$ and angles $h(\Delta \alpha)$.
5. **Ligand Conformational Strain Energy:** Calculate internal strain energy ($\Delta E_{\text{strain}} = E_{\text{bound}} - E_{\text{min}}$) using molecular mechanics.
6. **Conformational Sampling Algorithms:** Implement **Metropolis Monte Carlo with Simulated Annealing** to sample the conformational and positional landscape of a ligand in a binding pocket.
""")

    add_md(nb, """## 1. Prerequisites & Environment Setup""")

    add_code(nb, """import os
import sys
import random
import numpy as np
import pandas as pd

SEED = 42
FULL_RUN = False
random.seed(SEED)
np.random.seed(SEED)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path adjustments
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath("../../src"))

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import chemlib
    print("RDKit and chemlib successfully loaded.")
except ImportError:
    print("Running in standard NumPy/SciPy environment.")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 11
""")

    add_md(nb, """## 2. Physics & Thermodynamics of Molecular Recognition

The equilibrium binding of a ligand ($L$) to a macromolecular receptor ($R$) is governed by the standard Gibbs free energy of binding:
$$R + L \rightleftharpoons RL, \quad K_a = \frac{[RL]}{[R][L]}, \quad K_i = \frac{1}{K_a}$$

$$\Delta G^\circ = -RT \ln K_a = RT \ln K_i$$

At physiological temperature ($T = 298.15\ \text{K}$):
- $RT \approx 0.592\ \text{kcal/mol}$.
- A 10-fold increase in affinity (one log unit drop in $K_i$) corresponds to:
  $$\Delta \Delta G^\circ = 0.592 \times \ln(10) \approx 1.364\ \text{kcal/mol}$$
- **Cheng-Prusoff Relationship (Competitive Inhibition):**
  $$K_i = \frac{IC_{50}}{1 + \frac{[S]}{K_m}}$$
  where $[S]$ is substrate concentration and $K_m$ is the Michaelis constant. Under conditions where $[S] \ll K_m$, $K_i \approx IC_{50}$.

> **Crucial Reality Check:** Molecular docking scoring functions predict binding affinities with typical root-mean-squared errors of **$\pm 1.5 - 2.5\ \text{kcal/mol}$**. Therefore, docking scores should be used for **relative rank ordering**, not as true thermodynamic free energies!
""")

    add_code(nb, """# Interactive plot: Delta G vs Ki
R = 1.9872e-3  # kcal / (mol * K)
T = 298.15     # Kelvin

ki_values = np.logspace(-11, -3, 200)  # 10 pM to 1 mM
delta_g = R * T * np.log(ki_values)

plt.figure(figsize=(8, 4.5))
plt.semilogx(ki_values, delta_g, color="#2563eb", lw=2.5)
plt.axvline(x=1e-9, color="#e11d48", linestyle="--", label="1 nM ($K_i$) -> -12.3 kcal/mol")
plt.axvline(x=1e-6, color="#f59e0b", linestyle="--", label="1 uM ($K_i$) -> -8.2 kcal/mol")

plt.title(r"Standard Binding Free Energy vs. Inhibition Constant $K_i$", fontweight="bold")
plt.xlabel(r"Inhibition Constant $K_i$ (Molar, log scale)")
plt.ylabel(r"$\Delta G^\circ$ (kcal/mol)")
plt.legend()
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 3. The Three Classes of Scoring Functions

| Scoring Function Class | Underlying Philosophy | Mathematical Form | Examples | Strengths / Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **1. Physics-Based** | Molecular mechanics force fields (enthalpy + solvent approximations) | $E = E_{\text{vdW}} + E_{\text{Coulomb}} + \Delta G_{\text{solv}}$ | DOCK 4, CHARMM, AMBER, AutoDock 3 | Physically interpretable; slow; sensitive to coordinate jitter. |
| **2. Empirical** | Sum of parameterized descriptors weighted by regression on known $K_i$ | $\Delta G = \sum w_i f_i(\mathbf{r})$ | ChemScore, AutoDock 4, Vina, Glide | Fast; calibrated to drug-like complexes; prone to over-fitting. |
| **3. Knowledge-Based** | Statistical Potentials of Mean Force (PMF) derived from PDB frequencies | $w(r) = -k_B T \ln\left(\frac{\rho(r)}{\rho_{\text{bulk}}}\right)$ | PMF, DrugScore, BLEEP | Captures rare interactions implicitly; sensitive to PDB database bias. |

### Potentials of Mean Force (PMF)
Knowledge-based scoring utilizes the **Boltzmann inversion principle**:
$$w(r) = -k_B T \ln \left( \frac{g(r)}{g_{\text{bulk}}} \right)$$
If an atom pair $(i, j)$ occurs more frequently at distance $r$ in the PDB than expected by chance ($g(r) > g_{\text{bulk}}$), the interaction free energy is favorable ($w(r) < 0$).
""")

    add_code(nb, """# Comparing potential curves: LJ 12-6 vs Morse vs Knowledge-Based PMF
r = np.linspace(2.5, 7.0, 300)
sigma = 3.6
epsilon = 0.5  # kcal/mol

# 1. Lennard-Jones 12-6
v_lj = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# 2. Morse potential (covalent / deep basin)
r_e = sigma * (2**(1/6))
a = 1.5
v_morse = epsilon * (np.exp(-2 * a * (r - r_e)) - 2 * np.exp(-a * (r - r_e)))

# 3. Knowledge-Based PMF (with secondary desolvation hump)
v_pmf = -0.6 * np.exp(-((r - 3.8)**2) / 0.3) + 0.2 * np.exp(-((r - 5.2)**2) / 0.5)

plt.figure(figsize=(9, 5))
plt.plot(r, v_lj, label="Physics: Lennard-Jones 12-6", lw=2, color="#2563eb")
plt.plot(r, v_morse, label="Empirical: Morse Potential", lw=2, color="#10b981", linestyle="--")
plt.plot(r, v_pmf, label="Knowledge-Based: PMF w(r)", lw=2, color="#e11d48")

plt.ylim(-1.0, 1.5)
plt.axhline(0, color="gray", lw=0.8, linestyle=":")
plt.title("Comparison of Scoring Function Potentials", fontweight="bold")
plt.xlabel("Interatomic Distance $r$ (Å)")
plt.ylabel("Potential Energy / Score (kcal/mol)")
plt.legend()
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 4. AutoDock 4 & Vina Scoring Decomposition

### AutoDock 4 Master Equation:
$$\Delta G = W_{\text{vdW}} \sum_{i, j} \left( \frac{A_{ij}}{r_{ij}^{12}} - \frac{B_{ij}}{r_{ij}^6} \right) + W_{\text{hbond}} \sum_{i, j} E(t) \left( \frac{C_{ij}}{r_{ij}^{12}} - \frac{D_{ij}}{r_{ij}^{10}} \right) + W_{\text{elec}} \sum_{i, j} \frac{332.0 \cdot q_i q_j}{\epsilon(r_{ij}) r_{ij}} + W_{\text{desolv}} \Delta G_{\text{desolv}} + W_{\text{tors}} N_{\text{rot}}$$

- **The Electrostatic Constant $332.0$:**
  $$k_e = \frac{1}{4\pi\varepsilon_0} = \frac{e^2}{4\pi\varepsilon_0 \cdot 1\ \text{Å}} \approx 332.0637\ \frac{\text{kcal}\cdot\text{Å}}{\text{mol}\cdot e^2}$$
  This converts atomic partial charges (in elementary charge $e$) and distances (in Å) directly into **kcal/mol**.
- **Distance-Dependent Dielectric $\epsilon(r) = 4r$:**
  Models solvent screening implicitly without explicit water molecules.

### AutoDock Vina Scoring Terms:
AutoDock Vina replaces explicit physical formulas with conformation-dependent piecewise functions based on the surface distance $d_{ij} = r_{ij} - (R_i + R_j)$:
1. **Gauss 1:** $w_1 \cdot \exp(-(d / 0.5)^2)$ (attractive dispersion).
2. **Gauss 2:** $w_2 \cdot \exp(-((d - 3.0) / 2.0)^2)$ (secondary attractive basin).
3. **Repulsion:** $w_3 \cdot d^2$ for $d < 0$ (steric clash penalty).
4. **Hydrogen Bond:** $w_4 \cdot h(d)$ where $h(d) = 1$ for $d \le -0.7\ \text{Å}$ and decays linearly to 0 at $d = 0$.
5. **Torsional Entropy Penalty:** $\Delta G_{\text{dock}} = \frac{\Delta G_{\text{inter}}}{1 + 0.058 \cdot N_{\text{rot}}}$.
""")

    add_code(nb, """# Implementation of Vina-style distance terms
def vina_distance_terms(d: np.ndarray):
    \"\"\"Compute Vina pairwise terms given surface distance d = r - (R_i + R_j).\"\"\"
    gauss1 = np.exp(-(d / 0.5)**2)
    gauss2 = np.exp(-((d - 3.0) / 2.0)**2)
    repulsion = np.where(d < 0, d**2, 0.0)
    
    # H-bond piecewise
    hbond = np.zeros_like(d)
    hbond = np.where(d <= -0.7, 1.0, hbond)
    mask = (d > -0.7) & (d < 0.0)
    hbond[mask] = -d[mask] / 0.7
    
    return gauss1, gauss2, repulsion, hbond

d_grid = np.linspace(-1.5, 4.0, 300)
g1, g2, rep, hb = vina_distance_terms(d_grid)

plt.figure(figsize=(9, 5))
plt.plot(d_grid, g1, label="Gauss 1 (Short dispersion)", lw=2, color="#2563eb")
plt.plot(d_grid, g2, label="Gauss 2 (Long dispersion)", lw=2, color="#06b6d4")
plt.plot(d_grid, rep, label="Steric Repulsion ($d < 0$)", lw=2, color="#e11d48")
plt.plot(d_grid, hb, label="Hydrogen Bonding", lw=2, color="#10b981")

plt.axvline(0, color="black", linestyle="--", alpha=0.5, label="Contact distance ($d=0$)")
plt.title("AutoDock Vina Empirical Term Profiles", fontweight="bold")
plt.xlabel("Surface Distance $d = r_{ij} - (R_i + R_j)$ (Å)")
plt.ylabel("Term Value (unitless)")
plt.legend()
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 5. Glide-Style Piecewise Hydrogen-Bond Geometry

In Glide (Friesner et al., 2004), hydrogen bonds are scored with explicit geometric attenuation:
$$E_{\text{HB}} = W_{\text{HB}} \cdot g(\Delta r) \cdot h(\Delta \alpha)$$

1. **Distance Function $g(\Delta r)$:**
   $$\Delta r = |r_{\text{DA}} - r_{\text{ideal}}|$$
   $$g(\Delta r) = \begin{cases} 1 & \text{if } \Delta r \le 0.25\ \text{Å} \\ 1 - \frac{\Delta r - 0.25}{0.50} & \text{if } 0.25 < \Delta r \le 0.75\ \text{Å} \\ 0 & \text{if } \Delta r > 0.75\ \text{Å} \end{cases}$$
2. **Angle Function $h(\Delta \alpha)$:**
   $$\Delta \alpha = |180^\circ - \theta_{\text{D-H}\cdots\text{A}}|$$
   $$h(\Delta \alpha) = \begin{cases} 1 & \text{if } \Delta \alpha \le 20^\circ \\ 1 - \frac{\Delta \alpha - 20^\circ}{40^\circ} & \text{if } 20^\circ < \Delta \alpha \le 60^\circ \\ 0 & \text{if } \Delta \alpha > 60^\circ \end{cases}$$
""")

    add_code(nb, """def glide_hbond_terms(r_da: np.ndarray, theta: np.ndarray, r_ideal: float = 2.8):
    \"\"\"Calculate Glide-style distance and angle penalty factors.\"\"\"
    delta_r = np.abs(r_da - r_ideal)
    g_r = np.ones_like(delta_r)
    mask1 = (delta_r > 0.25) & (delta_r <= 0.75)
    g_r[mask1] = 1.0 - (delta_r[mask1] - 0.25) / 0.50
    g_r[delta_r > 0.75] = 0.0

    delta_a = np.abs(180.0 - theta)
    h_a = np.ones_like(delta_a)
    mask2 = (delta_a > 20.0) & (delta_a <= 60.0)
    h_a[mask2] = 1.0 - (delta_a[mask2] - 20.0) / 40.0
    h_a[delta_a > 60.0] = 0.0

    return g_r, h_a

r_test = np.linspace(2.0, 4.0, 200)
a_test = np.linspace(100.0, 180.0, 200)
gr, _ = glide_hbond_terms(r_test, np.full_like(r_test, 180.0))
_, ha = glide_hbond_terms(np.full_like(a_test, 2.8), a_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(r_test, gr, color="#8b5cf6", lw=2)
ax1.set_title(r"Distance Attenuation $g(\Delta r)$")
ax1.set_xlabel(r"Donor-Acceptor Distance $r_{\mathrm{DA}}$ (Å)")
ax1.set_ylabel("Attenuation Factor")

ax2.plot(a_test, ha, color="#ec4899", lw=2)
ax2.set_title(r"Angle Attenuation $h(\Delta \\alpha)$")
ax2.set_xlabel("D-H...A Angle (degrees)")
ax2.set_ylabel("Attenuation Factor")

plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 6. Conformational Sampling: Metropolis Monte Carlo Simulated Annealing

A docking program must search an immense conformational space:
- **Rigid Degrees of Freedom:** 3 translational coordinates $(x, y, z)$ + 3 rotational Euler angles $(\phi, \theta, \psi)$.
- **Flexible Degrees of Freedom:** $N_{\text{rot}}$ internal dihedral angles.
- Total search dimensions: $D = 6 + N_{\text{rot}}$.

### Metropolis Monte Carlo with Simulated Annealing (SA)
1. **Initial State:** Position ligand at $\mathbf{x}_0$ with energy $E_0$. Set high temperature $T_0$.
2. **Move Proposal:** Perturb position $\mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}$.
3. **Metropolis Criterion:**
   - If $\Delta E = E(\mathbf{x}') - E(\mathbf{x}) \le 0$: **Accept move**.
   - If $\Delta E > 0$: **Accept with probability** $P = \exp\left(-\frac{\Delta E}{k_B T}\right)$.
4. **Cooling Schedule:** Reduce temperature geometrically: $T_{k+1} = \gamma \cdot T_k$ (where $\gamma \approx 0.95 - 0.99$).
""")

    add_code(nb, """# Toy 2D Binding Pocket Energy Surface: multi-well potential
def pocket_energy_surface(x: float, y: float) -> float:
    \"\"\"A 2D energy landscape with local traps and a global binding minimum.\"\"\"
    # Global minimum near (3.0, 2.0) with depth -8.5 kcal/mol
    v_global = -8.5 * np.exp(-((x - 3.0)**2 + (y - 2.0)**2) / 1.5)
    # Local decoy minimum near (-2.5, -1.0) with depth -5.0 kcal/mol
    v_local = -5.0 * np.exp(-((x + 2.5)**2 + (y + 1.0)**2) / 2.0)
    # Steep steric walls
    v_walls = 0.08 * (x**4 + y**4)
    return float(v_global + v_local + v_walls)

# Visualize landscape
gx = np.linspace(-4, 5, 100)
gy = np.linspace(-4, 4, 100)
GX, GY = np.meshgrid(gx, gy)
GZ = np.vectorize(pocket_energy_surface)(GX, GY)

plt.figure(figsize=(7, 5.5))
cp = plt.contourf(GX, GY, GZ, levels=25, cmap="viridis")
plt.colorbar(cp, label="Energy (kcal/mol)")
plt.plot(3.0, 2.0, "r*", markersize=14, label="True Binding Pocket (-8.5 kcal/mol)")
plt.plot(-2.5, -1.0, "w^", markersize=10, label="Decoy Pocket (-5.0 kcal/mol)")
plt.title("Toy Pocket Energy Landscape", fontweight="bold")
plt.xlabel("Ligand X Position (Å)")
plt.ylabel("Ligand Y Position (Å)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
""")

    add_md(nb, """## 7. Guided Student Exercises & Self-Checks

---

### Exercise 1: Lennard-Jones 12-6 Potential Function
Implement the pairwise Lennard-Jones 12-6 potential function:
$$V_{\text{LJ}}(r, \epsilon, \sigma) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$
where:
- $r$ is the interatomic distance (in Å).
- $\epsilon$ is the potential well depth (in kcal/mol).
- $\sigma$ is the finite distance at which the inter-particle potential is zero (in Å).
""")

    add_code(nb, """# Exercise 1: Lennard-Jones Potential
def lennard_jones_potential(r: float, epsilon: float, sigma: float) -> float:
    \"\"\"Compute Lennard-Jones 12-6 potential energy (kcal/mol).\"\"\"
    # --- SOLUTION BEGIN ---
    ratio = sigma / r
    ratio6 = ratio ** 6
    ratio12 = ratio6 ** 2
    return float(4.0 * epsilon * (ratio12 - ratio6))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 1
# At r = sigma, V = 0
assert abs(lennard_jones_potential(3.5, 0.25, 3.5)) < 1e-6, "At r=sigma, potential must be 0"

# At r = sigma * 2^(1/6), V = -epsilon (minimum)
r_min = 3.5 * (2.0 ** (1.0 / 6.0))
v_min = lennard_jones_potential(r_min, 0.25, 3.5)
assert abs(v_min - (-0.25)) < 1e-4, f"At r_min, expected -0.25 kcal/mol, got {v_min}"
print("✔ Exercise 1 passed!")
""")

    add_md(nb, """---

### Exercise 2: Electrostatic Coulomb Interaction with Distance-Dependent Dielectric
Implement the electrostatic interaction between two point charges with distance-dependent dielectric $\epsilon(r) = 4r$:
$$V_{\text{elec}}(q_1, q_2, r) = \frac{332.0 \cdot q_1 \cdot q_2}{4 r^2}$$
where $q_1, q_2$ are partial charges in units of elementary charge $e$, and $r$ is distance in Å.
""")

    add_code(nb, """# Exercise 2: Electrostatic Potential
def coulomb_dielectric_potential(q1: float, q2: float, r: float) -> float:
    \"\"\"Compute electrostatic energy with distance-dependent dielectric eps(r) = 4r (kcal/mol).\"\"\"
    # --- SOLUTION BEGIN ---
    return float((332.0 * q1 * q2) / (4.0 * (r ** 2)))
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 2
# Opposite charges at 3.0 Angstroms
v_elec = coulomb_dielectric_potential(+1.0, -1.0, 3.0)
# Expected: -332.0 / (4 * 9) = -332 / 36 = -9.222 kcal/mol
assert abs(v_elec - (-9.2222)) < 1e-2, f"Expected -9.22 kcal/mol, got {v_elec}"

# Like charges at 4.0 Angstroms
v_rep = coulomb_dielectric_potential(+0.5, +0.5, 4.0)
assert v_rep > 0.0, "Like charges must repel"
print("✔ Exercise 2 passed!")
""")

    add_md(nb, """---

### Exercise 3: Metropolis Acceptance Criterion
Implement the Metropolis acceptance decision:
$$P_{\text{accept}} = \begin{cases} 1.0 & \text{if } \Delta E \le 0 \\ \exp\left(-\frac{\Delta E}{k_B T}\right) & \text{if } \Delta E > 0 \end{cases}$$
The function `metropolis_accept(delta_energy: float, temperature: float, random_uniform: float) -> bool`:
- Takes $\Delta E$ (kcal/mol), temperature $T$ (Kelvin), and a random float in $[0, 1)$.
- Uses Boltzmann constant $k_B = 1.9872 \times 10^{-3}\ \text{kcal}/(\text{mol}\cdot\text{K})$.
- Returns `True` if the proposed move is accepted, `False` otherwise.
""")

    add_code(nb, """# Exercise 3: Metropolis Acceptance
KB = 1.9872e-3  # kcal / (mol * K)

def metropolis_accept(delta_energy: float, temperature: float, rand_val: float) -> bool:
    \"\"\"Determine whether to accept a proposed Monte Carlo step.\"\"\"
    # --- SOLUTION BEGIN ---
    if delta_energy <= 0.0:
        return True
    prob = math.exp(-delta_energy / (KB * temperature))
    return rand_val < prob
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 3
# Favorable move (delta E < 0) always accepted
assert metropolis_accept(-2.0, 300.0, 0.99) is True

# High energy uphill step with high random number rejected
assert metropolis_accept(+5.0, 300.0, 0.90) is False

# Small uphill step at high temperature accepted
# At 1000 K, delta E = 0.5 kcal/mol -> prob = exp(-0.5 / (0.0019872 * 1000)) = exp(-0.2516) = 0.777
assert metropolis_accept(+0.5, 1000.0, 0.50) is True
print("✔ Exercise 3 passed!")
""")

    add_md(nb, """---

### Exercise 4: Simulated Annealing Minimization
Implement `simulated_annealing_optimize(n_steps: int = 500, t_init: float = 1000.0, cooling_rate: float = 0.98, seed: int = 42)`:
1. Start at coordinates $(x, y) = (-3.5, -2.5)$ (trapped in a suboptimal local region).
2. For each step:
   - Propose perturbation: $\Delta x, \Delta y \sim \mathcal{N}(0, 0.3^2)$.
   - Compute $\Delta E = E(x_{\text{new}}, y_{\text{new}}) - E(x_{\text{curr}}, y_{\text{curr}})$.
   - Accept or reject using `metropolis_accept`.
   - Update $T \leftarrow T \times \text{cooling\_rate}$.
3. Return the best $(x_{\text{best}}, y_{\text{best}}, E_{\text{best}})$ observed.
""")

    add_code(nb, """# Exercise 4: Simulated Annealing Minimization
def simulated_annealing_optimize(
    n_steps: int = 500,
    t_init: float = 1000.0,
    cooling_rate: float = 0.98,
    seed: int = 42
):
    \"\"\"Run Monte Carlo simulated annealing optimization on pocket_energy_surface.\"\"\"
    rng = np.random.RandomState(seed)
    # --- SOLUTION BEGIN ---
    curr_x, curr_y = -3.5, -2.5
    curr_e = pocket_energy_surface(curr_x, curr_y)
    
    best_x, best_y, best_e = curr_x, curr_y, curr_e
    temp = t_init

    for _ in range(n_steps):
        dx, dy = rng.normal(0, 0.35, size=2)
        cand_x = curr_x + dx
        cand_y = curr_y + dy
        cand_e = pocket_energy_surface(cand_x, cand_y)

        delta_e = cand_e - curr_e
        rand_u = rng.uniform(0.0, 1.0)

        if metropolis_accept(delta_e, temp, rand_u):
            curr_x, curr_y, curr_e = cand_x, cand_y, cand_e
            if curr_e < best_e:
                best_x, best_y, best_e = curr_x, curr_y, curr_e

        temp = max(temp * cooling_rate, 1e-4)

    return best_x, best_y, best_e
    # --- SOLUTION END ---
""")

    add_code(nb, """# Self-check: Exercise 4
opt_x, opt_y, opt_e = simulated_annealing_optimize(n_steps=1200, seed=42)
print(f"Annealing optimization finished at ({opt_x:.2f}, {opt_y:.2f}) with energy {opt_e:.2f} kcal/mol")

# Verify that the optimizer escapes the decoy basin and reaches the true pocket (x in [2.0, 4.0], y in [1.0, 3.0])
assert opt_e < -7.0, f"Expected optimizer to find deep pocket (< -7.0 kcal/mol), got {opt_e}"
assert 2.0 <= opt_x <= 4.0, f"Expected x in [2.0, 4.0], got {opt_x}"
assert 1.0 <= opt_y <= 3.0, f"Expected y in [1.0, 3.0], got {opt_y}"
print("✔ Exercise 4 passed!")
""")

    add_md(nb, """## Common Pitfalls

1. **Over-reliance on Raw Docking Energies**: Docking scoring functions approximate free energy of binding but omit explicit entropic penalty and solvent reorganization details.
2. **Metropolis Acceptance Trapping**: In Monte Carlo sampling, poor move proposals or suboptimal temperatures cause simulations to become trapped in local energy minima.
3. **Electrostatic Dielectric Approximations**: Distance-dependent dielectrics ($\\epsilon(r) = 4r$) are crude approximations of true aqueous continuum solvation.""")

    add_md(nb, """## Summary and Key Takeaways

- **Scoring Function Categories**: Physics-based force fields, empirical regressions, and knowledge-based potentials balance computational speed and binding affinity accuracy.
- **Conformational Search**: Monte Carlo and genetic algorithms explore high-dimensional ligand torsional degrees of freedom.
- **Scoring Decomposition**: Dissecting terms provides physical insight into electrostatic, steric, and desolvation contributions to affinity.""")

    add_code(nb, """# Final validation cell
print("✔ NB17 checks passed")
""")

    save_notebooks(
        nb,
        "notebooks/nb17_scoring_functions_and_sampling.ipynb",
        "solutions/nb17_scoring_functions_and_sampling.ipynb"
    )

if __name__ == "__main__":
    build_nb17()
