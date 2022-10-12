# End-to-end ML project

According to the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), there are eight main steps to an end-to-end Machine Learning project:
1. Frame the problem and look at the big picture.
2. Get the data.
3. Explore the data to gain insights.
4. Prepare the data to better expose underlying data patterns to Machine Learning alghoritms.
5. Explore many different models and shortlist the best one.
6. Fine tune your models and combine them into a great solution.
7. Present your solution.
8. Lunch, monitor, and maintain your system.

# 1. Frame the problem and look at the big picture

<p align="center">
<img src="https://drive.google.com/uc?id=1yEblTg9yQ2_qGoiiCHOHp1JxuttgnrOo" width="40%" >
</p>

## Problem Statement

**The object of this project is to predict the residuary resistance of sailing yachts from dimensions and velocity.**

[The dataset](http://archive.ics.uci.edu/ml/datasets/Yacht+Hydrodynamics) comprises the non-dimensional hull ratios, yacht velocity (the Froude number) and experimentally measured values of the residuary resistance:

$L_{CB}$ - *Centre of bouyancy*, 

$C_{p}$ - *Prismatic coefficient*,

$\frac{L_{WL}}{\nabla_{c}^{1/3}}$ - *Length/displacement ratio*,

$\frac{B_{WL}}{T_{c}}$ - *Beam/draft ratio*,

$\frac{L_{WL}}{B_{WL}}$ - *Length/beam ratio*,

$F_{n}$ - *Froude number*,

$\frac{R_{R}}{g \cdot m_{c}} \cdot 10^3$ - *residuary resistance per unit weight of displacement*.

Therefore, this project is framed as:
- **supervised learning task** - the dataset is labeled or each instance comes with the expected output,
- **regression task** - the model has to predict a value,
- **multiple regression problem** - the model will use multiple features to make a prediction,
- **univariate regression problem** - the model will predict a single value for each instance,
- **batch learning** - there is no continuous flow of data coming into the model and and the data is small enough to fit in memory.

## The residuary resistance of sailing yachts ⛵

The resistance curve for the YD-40 yacht, towed upright in smooth water, is shown in the next figure.

<p align="center">
<img src="https://drive.google.com/uc?id=1GIPdBzmPdTQadjAt6PwSGpo8lcLDTYw_" width="50%" >
</p>

The total or upright resistance consists of the viscous resistance, dominating component at low speeds, and the wave resistance, which occurs because the hull generates waves, transferring the energy away.

However, in a real sailing situation, the total resistance gets more complicated and the residuary resistance (wave resitance + viscous pressure resistance) is the biggest component of the total force.

<p align="center">
<img src="https://drive.google.com/uc?id=1PynJPYe4dSizE9y-huzpB96x0EzBA95e" width="50%" >
</p>

Hence, prediction of residuary resistance of sailing yachts at the initial design stage is of a great value for evaluating the ship’s performance and for estimating the required propulsive power.

# References
[1] A. Géron, Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, O'Reilly Media, 2019.

[2] Larsson and R. E. Eliasson, Principles of Yacht Design, Adlard Coles Nautical, 2000.