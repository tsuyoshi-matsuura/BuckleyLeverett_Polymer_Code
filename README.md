# Polymer analysis

## Purpose
The purpose of this notebook is to understand the 'traditional' solution for the polymer flood. The solution turned out to be more complex than initially anticipated, in particular, when the initial water saturation $S_{wi}$ is varied or the oil Corey exponent is 1.

Very helpful paper:  
Johansen, Thormod, and Ragnar Winther. “The Solution of the Riemann Problem for a Hyperbolic System of Conservation Laws Modeling Polymer Flooding.” SIAM Journal on Mathematical Analysis 19, no. 3 (May 1988): 541–66. https://doi.org/10.1137/0519039.

## Equations

The following equations need to be solved:
```math
\begin{cases}
 \phi \dfrac{\partial}{\partial t}\left( S_w \right) + u \dfrac{\partial}{\partial x}\left( f_w \right) = 0 \\
 \\
 \phi \dfrac{\partial}{\partial t}\left( S_w C_w + \dfrac{(1-\phi)}{\phi})\dfrac{\rho_s}{\rho_w} C_s \right) + u \dfrac{\partial}{\partial x}(f_w C_w) = 0

\end{cases}
```
with the boundary and initial conditions:
```math
\begin{cases}
S_w(0,t) &= S_{winj} = 1 - S_{orw} \\
C_w(0,t) &= C_{inj} \\
S_w(x,0) &= S_{wi} \\
C(_wx,0) &= 0 
\end{cases}
```
with ```math t \geq 0``` and $ 0 \leq x \leq L$.

In the above equations $S_w$, $C_w$ and $C_s$ are the water saturation, polymer mass fractions in the water resp. solid. Moreover, $u$ is the total fluid rate, $\phi$ the porosity and $\rho_w$ and $\rho_s$ are the water resp. rock densities. $f_w$ is the fractional flow of water, given by:
$$
f_w = \dfrac{u_w}{u} = \left. \dfrac{k_{rw}}{\mu_w} \middle/ \left( \dfrac{k_{rw}}{\mu_w} + 
\dfrac{k_{ro}}{\mu_o}\right) \right.
$$
where $u_w$ is the water rate, $k_{rw}$ and $k_{ro}$ are the aqueous resp. oil relative permeabilities and $\mu_w$ and $\mu_{o}$ the aqueous resp. oil viscosities. It is assumed that the relative permeabilities are only dependent on $S_w$ and the aqeous viscosity only a function of $C_w$. Finally, $S_{winj}$ is the water saturation corresponding to pure aqueous injection, i.e. $f_w = 1$ at $x=0$, $S_{wi}$ is the initial water saturation in the 'reservoir' and $C_{inj}$ the mass fraction of polymer in the injection solution.

The following dimensionless variables are introduced:
$$
\begin{align}
x_D &= \frac{x}{L}\\
t_D &= \frac{u}{\phi L}t = \frac{Q_{inj}}{V_p} \\
C &= \frac{C_w}{C_{inj}}
\end{align}
$$
Moreover, it is convenient to introduce:
$$
\begin{align}
S &= \frac{(S_w-S_{cw})}{(1-S_{cw}-S_{orw})} \\
a &= \frac{(1-\phi)}{\phi})\frac{\rho_s}{\rho_w} \frac{C_s}{C_{inj}}
\end{align}
$$
The variable $S$ is used for internal calculations of the relperms and fractional flow curves.  
With the above the equations become:
$$
\begin{cases}
 \dfrac{\partial}{\partial t_D} \left( S_w \right)+ \dfrac{\partial}{\partial x_D} \left( f_w \right)= 0 \\
 \\
 \dfrac{\partial}{\partial t_D}\left( S_w C + a \right) + \dfrac{\partial}{\partial x_D}(f_w C) = 0

\end{cases}
$$

By differentiating out the above equations and using the first equation to simplify the second, we get
$$
\begin{cases}
       \dfrac{\partial S_w}{\partial t_D}  &+& 
       \dfrac{\partial f_w}{\partial S_w} \dfrac{\partial S_w}{\partial x_D} &+& 
       \dfrac{\partial f_w}{\partial C} \dfrac{\partial C }{\partial x_D} &=0 \\
       \\
       \dfrac{\partial C}{\partial t_D}  & & &+&
       \dfrac{f_w}{S_w + \frac{d a}{d C}} \dfrac{\partial C}{\partial x_D} &=0
\end{cases}
$$

## Equations in matrix form

In matrix form we have:
$$
\begin{gather}
  \begin{pmatrix} S_w \\ C \end{pmatrix}_{tD} + A \left( S_w,C \right) \begin{pmatrix} S_w \\ C \end{pmatrix}_{xD} = 0

\end{gather}
$$
where the matrix $A$ is given by:
$$
A\left(S_w,C\right) = df_w = \begin{pmatrix} \dfrac{\partial f_w}{\partial S_w} & \dfrac{\partial f_w}{\partial C} \\
                                                \\
                                                0 & \dfrac{f_w}{S_w + \frac{d a}{d C}}
                               \end{pmatrix}
$$

## Eigenvalues and eigenvectors
The eigenvalues of the matrix $A$ are given by:
$$
\begin{align}
   \lambda_S & =  \dfrac{\partial f_w}{\partial S_w} \\
   \lambda_C & =  \dfrac{f_w}{S_w + \frac{d a}{d C}}
\end{align}
$$

The corresponding eigenvectors are:
$$
\begin{align}
   r_S &= \begin{pmatrix} 1 \\ 0 \end{pmatrix} \\
   r_C &= \begin{pmatrix} \dfrac{\partial f_w}{\partial C}  \\ 
                            \dfrac{f_w}{S_w + \frac{d a}{d C}} - \dfrac{\partial f_w}{\partial S_w}
           \end{pmatrix}
\end{align}
$$

The eigenvalues $\lambda_S$ and $\lambda_C$ are the speeds of the S- resp. C- wave rarefactions. Note that there is no ranking in speed, i.e. $\lambda_S \leq \lambda_C$ does not hold, as the problem is non-convex.

In this notebook we will investigate S- and C-wave integral curves and Hugoniot loci to better understand the construction of the polymer solution.
