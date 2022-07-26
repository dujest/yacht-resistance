# End-to-end ML project

## Yacht residuary resistance prediction

The resistance curve for the YD-40 yacht, towed upright in smooth water, is shown in the next figure.

![yacht_resistance](https://user-images.githubusercontent.com/94038478/181022449-5e586634-cbeb-4fa3-baf3-a906fffe63a6.PNG)

The total or upright resistance consists of the viscous resistance, dominating component at low speeds, and the wave resistance, which occurs because the hull generates waves, transferring the energy away.

However, in a real sailing situation, the total resistance gets more complicated and the residuary resistance is the biggest component of the total force.

![yacht_resistance_breakdown](https://user-images.githubusercontent.com/94038478/181024088-35cc81c7-0001-4753-9dcb-6eaf6e949ee7.PNG)

Therefore, prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the ship’s performance and for estimating the required propulsive power.

[The dataset](http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics) comprises the non-dimensional hull ratios, yacht speed (Froude number) and measured values of the residuary resistance per unit weight of displacement.
