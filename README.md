# Solutions for the Buckley Leverett and polymer model equations

In this repository Python code is provided for the solutions to the Buckley Leverett and polymer model equations.

## Buckley Leverett equation

<<<<<<< HEAD
The Buckley Leverett equation including gravity effects is given by:
```math
\dfrac{A \phi}{q}\dfrac{\partial S_w}{\partial t} + \dfrac{\partial f_w}{\partial S_w}\dfrac{\partial S_w}{\partial x} =0
$$
with the boundary an initial conditions:
$$
\begin{cases}
S_w(x,0) = S_{wi}\\
S_w(0,t) =S_{winj}\\
\end{cases}
```
with $t \geq 0$ and $0 \leq x \leq L$.

The fractional flow of water, $f_w$, is given by:
```math
f_w(S_w) = \dfrac{q_w}{q} = 
           \left. \dfrac{k_{rw}}{\mu_w} \middle/ \left( \dfrac{k_{rw}}{\mu_w} + 
                  \dfrac{k_{ro}}{\mu_o}\right) \right. 
           \left\{ 1- \dfrac{k k_{ro} A}{q \mu_o} \Delta \rho \sin\alpha \right\}
```
with $\Delta\rho = \rho_w - \rho_o$.

In the above equations $S_w$ is the water saturation, $q$ is the total volumetric fluid rate, $\phi$ the porosity and $A$ the area open to flow. 
Moreover, $k_{rw}$ and $k_{ro}$ are the water resp. oil relative permeabilities, $\mu_w$ and $\mu_{o}$ the water resp. oil viscosities and $\rho_w$ and $\rho_o$
are the water resp. oil densities.. It is assumed that the relative permeabilities are only dependent on $S_w$. 
The absolute permeability is denoted by $k$ and the dip angle by $\alpha$ (the angle is measured wrt the horizontal plane and the positive 
direction is anti-clockwise). Finally, $S_{winj}$ is the water saturation corresponding to pure water injection, i.e. $f_w = 1$ at $x=0$, 
$S_{wi}$ is the initial water saturation in the 'reservoir'.

For solving the Buckley Leverett equation he following dimensionless variables are introduced:
```math
\begin{align}
x_D &= \frac{x}{L}\\
t_D &= \frac{u}{\phi L}t = \frac{Q_{inj}}{V_p} \\
\end{align}
```
With the above the equation becomes:
```math
\dfrac{\partial S_w}{\partial t_D} + \dfrac{\partial f_w}{\partial S_w}\dfrac{\partial S_w}{\partial x_D} =0
```


## The polymer model equation

The polymer model with adsorption is described by the following equations:
=======
## Equations

The following equations need to be solved:
>>>>>>> 418b60cb6863498455055583b558c5551e32ded9
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
with $t \geq 0$ and $ 0 \leq x \leq L$.

In the above equations $S_w$, $C_w$ and $C_s$ are the water saturation, polymer mass fractions in the water resp. solid. Moreover, $u$ is the total fluid velocity,
$\phi$ the porosity and $\rho_w$ and $\rho_s$ are the water resp. rock densities. $f_w$ is the fractional flow of water, given by:
```math
f_w = \dfrac{u_w}{u} = \left. \dfrac{k_{rw}}{\mu_w} \middle/ \left( \dfrac{k_{rw}}{\mu_w} + 
\dfrac{k_{ro}}{\mu_o}\right) \right.
```
where $u_w$ is the water velocity, $k_{rw}$ and $k_{ro}$ are the aqueous resp. oil relative permeabilities and $\mu_w$ and $\mu_{o}$ the aqueous resp. oil viscosities.
It is assumed that the relative permeabilities are only dependent on $S_w$ and the aqeous viscosity only a function of $C_w$. 
Finally, $S_{winj}$ is the water saturation corresponding to pure aqueous injection, i.e. $f_w = 1$ at $x=0$, $S_{wi}$ is the initial water saturation in the 'reservoir' 
and $C_{inj}$ the mass fraction of polymer in the injection solution.

The following dimensionless variables are introduced:
```math
\begin{align}
x_D &= \frac{x}{L}\\
t_D &= \frac{u}{\phi L}t = \frac{Q_{inj}}{V_p} \\
C &= \frac{C_w}{C_{inj}}
\end{align}
```
Moreover, it is convenient to introduce:
```math
a &= \frac{(1-\phi)}{\phi})\frac{\rho_s}{\rho_w} \frac{C_s}{C_{inj}}
```
With the above the equations become:
```math
\begin{cases}
 \dfrac{\partial}{\partial t_D} \left( S_w \right)+ \dfrac{\partial}{\partial x_D} \left( f_w \right)= 0 \\
 \\
 \dfrac{\partial}{\partial t_D}\left( S_w C + a \right) + \dfrac{\partial}{\partial x_D}(f_w C) = 0

\end{cases}
```

By differentiating out the above equations and using the first equation to simplify the second, we get
```math
\begin{cases}
       \dfrac{\partial S_w}{\partial t_D}  &+& 
       \dfrac{\partial f_w}{\partial S_w} \dfrac{\partial S_w}{\partial x_D} &+& 
       \dfrac{\partial f_w}{\partial C} \dfrac{\partial C }{\partial x_D} &=0 \\
       \\
       \dfrac{\partial C}{\partial t_D}  & & &+&
       \dfrac{f_w}{S_w + \frac{d a}{d C}} \dfrac{\partial C}{\partial x_D} &=0
\end{cases}
```
