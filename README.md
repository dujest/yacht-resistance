# End-to-end ML project

<img src="https://drive.google.com/uc?id=1yEblTg9yQ2_qGoiiCHOHp1JxuttgnrOo" width="40%" >

## Yacht residuary resistance prediction ⛵

The resistance curve for the YD-40 yacht, towed upright in smooth water, is shown in the next figure.

<img src="https://drive.google.com/uc?id=1GIPdBzmPdTQadjAt6PwSGpo8lcLDTYw_" width="50%" >

The total or upright resistance consists of the viscous resistance, dominating component at low speeds, and the wave resistance, which occurs because the hull generates waves, transferring the energy away.

However, in a real sailing situation, the total resistance gets more complicated and the residuary resistance is the biggest component of the total force.

<img src="https://drive.google.com/uc?id=1PynJPYe4dSizE9y-huzpB96x0EzBA95e" width="50%" >

Therefore, prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the ship’s performance and for estimating the required propulsive power.

[The dataset](http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics) comprises the non-dimensional hull ratios, yacht speed (the Froude number) and measured values of the residuary resistance per unit weight of displacement:

$L_{CB}$ - *Centre of bouyancy* 

$C_{p}$ - *Prismatic coefficient*

$\frac{L_{WL}}{\nabla_{c}^{1/3}}$ - *Length/displacement ratio*

$\frac{B_{WL}}{T_{c}}$ - *Beam/draft ratio*

$\frac{L_{WL}}{B_{WL}}$ - *Length/beam ratio*

$F_{n}$ - *Froude number*

$\frac{R_{R}}{g \cdot m_{c}} \cdot 10^3$ - *residuary resistance per unit weight of displacement*

## References
L. Larsson and R. E. Eliasson, Principles of Yacht Design, Adlard Coles Nautical, 2000.