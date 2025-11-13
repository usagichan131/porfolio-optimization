# Optimization Project – Minimum Fuel Control, LP Duality & Portfolio Optimization

This repository contains the code and report material for a course project in **optimization / convex optimization**.

The project implements and analyzes several classic problems:

1. **Minimum fuel optimal control** of a discrete-time linear system, modeled and solved as a linear program (LP).
2. **Sensitivity analysis of primal–dual optimal solutions** for linear programs under basic problem transformations.
3. **LP reformulation and duality** for an \(\ell_1\)-norm–constrained problem and its \(\ell_\infty\)-regularized dual.
4. **Mean–variance portfolio optimization** (Markowitz model), including an extension with a **risk-free asset** and a numerical study of the **efficient frontier** using real market data.

The implementation uses **Python**, **NumPy**, **Matplotlib**, and **CVXPY**. In the portfolio part, **yfinance** is used to fetch historical stock data.

---

## 1. Problem 1 – Minimum Fuel Optimal Control

We consider a discrete-time linear system

\[
x(t + 1) = A x(t) + b u(t), \quad t = 0, \dots, N-1,
\]

with

- state \(x(t) \in \mathbb{R}^n\),
- scalar control input \(u(t) \in \mathbb{R}\),
- initial state \(x(0) = 0\),
- terminal constraint \(x(N) = x_{\text{des}}\).

The **total fuel** consumed is

\[
F = \sum_{t=0}^{N-1} f(u(t)),
\]

where the fuel map is the convex piecewise-linear function

\[
f(a) =
\begin{cases}
|a|, & |a| \le 1, \\
2|a| - 1, & |a| > 1.
\end{cases}
\]

The tasks implemented in the code are:

- Derive an **LP formulation** using auxiliary variables:
  - Introduce variables \(t_i\) such that \(t_i \ge f(u_i)\) and minimize \(\sum_i t_i\).
  - Use the controllability matrix \(C = [A^{N-1} b, A^{N-2} b, \dots, A b, b]\) to enforce \(Cu = x_{\text{des}}\).
- Solve the problem in **two ways** using CVXPY:
  1. Directly, via `abs`, `maximum` and dynamic constraints, letting CVXPY build the LP.
  2. Using the **explicit LP** derived from the fuel function.
- Simulate and plot:
  - The optimal control input \(u(t)\) for \(t = 0, \dots, N-1\),
  - The position and velocity trajectories (components of \(x(t)\)) for \(t = 0, \dots, N\).

An example instance included in the project:

- \(A = \begin{bmatrix} 1 & 1 \\ 0 & 0.95 \end{bmatrix}\),
- \(b = [0,\ 0.1]^T\),
- \(x(0) = (0, 0)\), \(x_{\text{des}} = (10, 0)\),
- horizon \(N = 20\).

This models a simple 1D vehicle: \(x_1(t)\) position, \(x_2(t)\) velocity, with damping in the velocity dynamics.

---

## 2. Problem 2 – LP Sensitivity (Primal & Dual)

We study the standard LP

\[
\begin{aligned}
\text{minimize} &\quad c^T x \\
\text{subject to} &\quad A x = b, \\
&\quad x \ge 0,
\end{aligned}
\]

and its dual. Assuming both are feasible, let \(x^*\) and \(y^*\) be optimal primal and dual solutions.

The report and notes analyze what happens to \(x^*\), \(y^*\), and the optimal values when we perform, **separately**:

1. Multiply the cost vector by a scalar: \(c \leftarrow \lambda c\), \(\lambda > 0\).
2. Multiply a single equality constraint by \(\lambda\).
3. Replace one equality constraint with a linear combination of it and another constraint.
4. Multiply the right-hand side by \(\lambda\): \(b \leftarrow \lambda b\).

For each case, the project:

- Uses **strong duality** to reason about changes in the optimal value.
- Identifies whether the **optimal primal solution** \(x^*\) and **optimal dual solution** \(y^*\) remain the same or change.
- Clarifies how **equivalent constraint transformations** can leave the primal optimal set unchanged while modifying the dual formulation.

This part is analytical/theoretical and is documented in the report and accompanying notes.

---

## 3. Problem 3 – \(\ell_1\)-Constraint as LP & Dual Problem

We consider the problem

\[
\begin{aligned}
\text{minimize} &\quad c^T x \\
\text{subject to} &\quad \|A x + b\|_1 \le 1,
\end{aligned}
\]

with \(A \in \mathbb{R}^{m \times n}\), \(b \in \mathbb{R}^m\), \(c \in \mathbb{R}^n\).

The implementation and write-up do the following:

1. Introduce an auxiliary variable \(y \in \mathbb{R}^m\) to represent \(|A x + b|\), and rewrite the \(\ell_1\)-constraint as
   linear inequalities, producing an **LP in inequality form** with variables \((x, y)\).
2. Derive the **dual LP**, and show it is equivalent to
   \[
   \begin{aligned}
   \text{maximize} &\quad b^T z - \|z\|_\infty \\
   \text{subject to} &\quad A^T z + c = 0,
   \end{aligned}
   \]
   for some dual vector \(z\).
3. Describe explicitly how to map between the standard dual variables and the vector \(z\).
4. Provide a direct inequality argument (without quoting LP duality) that for any primal-feasible \(x\) and dual-feasible \(z\),
   \[
   c^T x \;\ge\; b^T z - \|z\|_\infty,
   \]
   establishing **weak duality** directly from norm properties.

This section illustrates how classical LP duality and norm inequalities are connected.

---

## 4. Problem 7 – Portfolio Optimization (Markowitz Model)

The final part of the project addresses **mean–variance portfolio optimization**.

Let

- \(x \in \mathbb{R}^n\) be portfolio weights (fractions of wealth in each asset),
- \(r \in \mathbb{R}^n\) be estimated expected returns,
- \(V \in \mathbb{R}^{n \times n}\) be the covariance matrix of returns,
- \(\alpha \ge 0\) be the risk-aversion parameter.

Baseline Markowitz formulation:

\[
\begin{aligned}
\text{maximize} &\quad r^T x - \alpha x^T V x \\
\text{subject to} &\quad \sum_i x_i = 1, \\
&\quad x_i \ge 0.
\end{aligned}
\]

### 4.1 Extension with Risk-Free Asset

The report extends the model to include a **risk-free asset** with return \(r_f\). If \(\sum_i x_i\) is the total proportion
invested in risky assets, then \(1 - \sum_i x_i\) is implicitly invested in the risk-free asset, and the objective becomes

\[
\text{maximize} \quad r^T x + r_f (1 - \textstyle\sum_i x_i) - \alpha x^T V x,
\]

with the same non-negativity and budget constraints. This captures the trade-off between safe and risky investments.

### 4.2 Numerical Efficient Frontier (Dow 30 Example)

Using 25 weeks of historical data for the 30 stocks in the Dow Jones Industrial Average:

1. Download adjusted close prices with `yfinance` over a chosen 25-week window.
2. Compute weekly returns, then estimate:
   - mean return vector `r`,
   - covariance matrix `V`.
3. For several values of \(\alpha\) (e.g., 1, 2, 5, 10, 100):
   - Solve the Markowitz problem with CVXPY.
   - Record portfolio return and variance for each \(\alpha\).
4. Plot the **efficient frontier**:
   - x-axis: standard deviation \(\sqrt{x^T V x}\),
   - y-axis: expected return \(r^T x\).

This illustrates how portfolio composition shifts as risk aversion changes, and how efficient portfolios lie on the frontier.

---

## 6. Installation & Setup

### 6.1 Clone the repository

```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_FOLDER>
```

### 6.2 Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 6.3 Install dependencies

If a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

Otherwise, install the main packages manually:

```bash
pip install numpy pandas matplotlib cvxpy yfinance jupyter
```

---

## 7. Usage

### 7.1 Jupyter Notebooks

Launch Jupyter:

```bash
jupyter notebook
```

Then open and run the notebooks in each subdirectory, for example:

1. `problem1_minimum_fuel_control/minimum_fuel_control.ipynb`
2. `problem3_l1_duality/l1_duality.ipynb`
3. `portfolio_optimization/portfolio_markowitz.ipynb`

Each notebook contains explanation, model setup, solver calls, and plots.

---

## 8. References

- S. Boyd and L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004.  
- H. Markowitz, “Portfolio Selection,” *Journal of Finance*, 1952.  

These references are the theoretical foundation for the minimum-fuel control LP, LP duality, and mean–variance portfolio optimization.

---

## 9. License

If you intend to share this code openly, add a `LICENSE` file and mention it here, for example:

```text
This project is licensed under the MIT License – see the LICENSE file for details.
```

If you do not add a license, the code is by default “all rights reserved”.

---

## 10. Contact

For questions, feedback, or collaboration, please contact the authors listed in the project report, or open an issue in this repository.
