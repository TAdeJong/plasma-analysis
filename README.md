# Plasma-analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2069945.svg)](https://doi.org/10.5281/zenodo.2069945)


Repository containing code used to calculate field lines by massively parallel numerical integration using Runge-Kutta methods implemented in CUDA. And extract topological properties from the field line traces.

Also contains some Python code to plot resulting data.

The fields analysed contain field lines lying on toroidal surfaces. The program finds these tori (for an orientation of the main axis reasonably aligned with the z-axis) and calculates the winding number of the field lines relative to this torus.

This code was used to produce a part of the data and plots for Smiet et al., _Resistive evolution of toroidal field distributions and their relation to magnetic clouds_ (2018, https://arxiv.org/abs/1812.00005) (accepted in [Journal of Plasma Physics](https://www.cambridge.org/core/journals/journal-of-plasma-physics)).
