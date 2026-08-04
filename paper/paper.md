---
title: 'GEM: Ghost Embedding Method'
tags:
  - Python
  - condensed matter physics
  - strongly correlated electrons
  - quantum embedding
  - ghost Gutzwiller approximation
  - dynamical mean-field theory
authors:
  - name: Samuele Giuli
    orcid: 0009-0004-7341-3655
    #equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
    email: sgiuli@flatironinstitute.org
  - name: Author Without ORCID
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Center for Computational Quantum Physics (CCQ), Flatiron Institute, New York, New York 10010, USA
   index: 1
   #ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 4 August 2026
bibliography: paper.bib

---

# Summary

GEM (Ghost Embedding Method) is an open-source Python package for computing equilibrium properties of models of strongly correlated electrons. Such models describe materials in which interactions between electrons cannot be treated as a small correction and can produce phenomena including interaction-driven metal-insulator transitions, strong renormalization of quasiparticles, magnetism, and orbital-selective behavior. GEM implements the ghost Gutzwiller approximation (ghost-GA) at zero and finite temperature, providing a computational framework in which a correlated lattice problem is mapped to coupled auxiliary quasiparticle and quantum-embedding problems [@Lanata2017; @Giuli2026].

The central approximation is controlled by the number of auxiliary, or ``ghost``, electronic levels. With one auxiliary level per physical level, the method reduces to the conventional Gutzwiller approximation; increasing the auxiliary space enriches the representable correlation structure, while the infinite-auxiliary limit recovers dynamical mean-field theory (DMFT) [@Giuli2026]. GEM exposes this formulation through composable Python objects for lattices, correlated fragments, self-consistency updates, impurity solvers, and observable evaluation. The package supports single- and multi-fragment lattice models, zero- and finite-temperature calculations, reusable examples and automated tests.

# Statement of need

Nonperturbative calculations for correlated-electron models require a compromise between physical fidelity, numerical cost, and the range of accessible observables. DMFT is a standard framework for local quantum correlations, but its self-consistency loop requires repeated solutions of an interacting quantum impurity problem and the evaluation of frequency-dependent quantities [@Georges1996]. Depending on the impurity solver, calculations can become costly for many orbitals, low temperatures, broad parameter scans, or symmetry-broken phases. Conventional Gutzwiller methods are substantially lighter, but their restricted auxiliary space limits the spectral structures that they can represent and historically obscured their precise relation to dynamical embedding theories [@Gutzwiller1965; @Lanata2017].

GEM addresses this gap by making the systematically improvable ghost-GA formulation available as research software. The implementation follows a free-energy functional that unifies zero-temperature ghost-GA, its finite-temperature extension, and DMFT: stationary conditions are expressed using static or thermal expectation values of effective Hamiltonians, and DMFT is obtained as the number of auxiliary bath modes tends to infinity (B\rightarrow\infty) [@Giuli2026].
Importantly, this correspondence is not merely asymptotic in practical calculations. For the finite-temperature Hubbard-model benchmarks reported in [@Giuli2026], calculations with only (B=3) already reproduce DMFT thermodynamic results with high accuracy across the investigated interaction and temperature regimes. GEM therefore provides a controlled hierarchy in which increasing (B) systematically enlarges the auxiliary space, while small values of (B) can already offer an advantageous balance between computational cost and DMFT-level accuracy. This allows researchers to investigate ground-state and thermodynamic properties without requiring a conventional frequency-dependent DMFT workflow at every stage of a calculation.

The target users are researchers developing or applying quantum embedding methods in condensed-matter physics and materials theory. Representative uses include Hubbard and multiorbital model studies, phase-diagram calculations, tests of convergence with auxiliary-space size, investigations of Mott and orbital-selective transitions, and benchmarking or developing impurity solvers. By providing the method as a reusable package rather than a collection of project-specific scripts, GEM supports reproducible comparisons across models, solvers, temperatures, and approximation levels.

# State of the field

The computational ecosystem for correlated-electron and quantum-embedding calculations spans several complementary classes of impurity solvers. Exact-diagonalization approaches represent the bath with a finite set of levels and provide direct access to real-frequency and ground-state quantities, but their cost grows exponentially with the number of impurity and bath degrees of freedom [@Caffarel1994, @Amaricci2022, @Crippa2025]. Tensor-network methods, including matrix-product-state impurity solvers, can accommodate substantially larger discretized baths by exploiting low-entanglement structure and have been applied to both imaginary- and real-time DMFT calculations [@Wolf2015; @Ganahl2015]. At finite temperature, continuous-time quantum Monte Carlo methods are widely used because they avoid an explicit bath discretization and can treat general multiorbital impurity models, although statistical noise, sign problems, and analytic continuation may limit accessible regimes or observables [@Gull2011]. Implementations such as w2dynamics provide production-oriented continuous-time quantum Monte Carlo workflows for one- and two-particle quantities [@Wallerberger2019], while broader community projects such as ALPS include implementations and reusable components for exact diagonalization, tensor-network methods, and quantum Monte Carlo [@Bauer2011].

These solver technologies are complemented by software frameworks that provide common representations of Green’s functions, operators, data formats, and many-body workflows. TRIQS is one such framework [@Parcollet2015], but GEM is not tied conceptually to a single software ecosystem. Its distinguishing contribution is the implementation of a variational embedding hierarchy controlled by the number (B) of auxiliary bath modes. In contrast to packages centered on the numerical solution of a conventional quantum impurity action, GEM formulates the correlated problem through coupled quasiparticle and embedding Hamiltonians and supports systematic convergence toward the DMFT limit. It therefore occupies a distinct position between inexpensive static Gutzwiller approximations and fully dynamical impurity-solver workflows.

GEM serves a different role. It is not a replacement for general many-body frameworks or highly optimized impurity solvers; it is an implementation of a specific variational-to-dynamical embedding hierarchy that is not otherwise available as a maintained, documented package.
Contributing the implementation directly to a conventional DMFT solver would not expose the method's distinctive optimization variables, finite auxiliary-space hierarchy, or coupled quasiparticle/embedding stationarity equations cleanly. Conversely, implementing all infrastructure independently would duplicate capabilities already available in the scientific Python and TRIQS ecosystems.
GEM therefore adopts a focused package design: it implements the ghost-embedding algorithm and its domain objects while remaining interoperable with external numerical and impurity-solver components.

This positioning enables two complementary workflows. GEM can be used as a lower-cost variational embedding method for static and thermal observables at moderate auxiliary-space size, and it can be used as a controlled bridge toward DMFT by increasing that size. The latter provides a direct route for methodological studies that compare variational and dynamical descriptions within one formal and computational framework.

# Software Design

GEM separates the physical layers of the method into explicit software components. A lattice object stores the one-body dispersion and integration weights and solves the auxiliary quasiparticle problem. Correlated fragments store local interactions, embedding parameters, density matrices, and local observables. Solver objects handle the interacting embedding Hamiltonians, allowing the self-consistency machinery to remain independent of a particular impurity-solution strategy. A typical iteration solves the quasiparticle problem, updates hybridization parameters, solves each embedding problem, updates the self-energy parametrization, mixes parameters, and checks convergence.

This decomposition reflects several design trade-offs. First, GEM uses high-level Python interfaces to make model construction, experimentation, and inspection straightforward, while relying on NumPy and SciPy for array operations, sparse linear algebra, and optimization [@Harris2020; @Virtanen2020]. Numba is used where just-in-time compilation benefits basis construction [@Lam2015], and HDF5 support enables portable storage of numerical results [@Collette2013]. Second, the code represents the variational matrices directly rather than hiding them behind a monolithic solver. This increases transparency and makes new update schemes, symmetry constraints, mixing strategies, and observables easier to prototype. Third, the impurity-solver abstraction permits simple exact diagonalization for compact examples while leaving room for interfaces to more specialized solvers.

The repository includes executable examples for Bethe, square, and triangular lattices, including single-fragment, multi-fragment, plaquette, magnetic, and phase-diagram calculations. Automated tests and continuous-integration configuration exercise core functionality, while the user guide documents the theoretical mapping and the self-consistency sequence. GEM is distributed under the GNU General Public License version 3 or later and follows the build and packaging conventions of the TRIQS application ecosystem.


# AI usage disclosure

Generative AI was used to help reorganize and edit an initial draft of this JOSS paper.
Generative AI was also used in different stages to refactor some parts of the code and to uniform the testing framework.

# Acknowledgements

The authors thank the contributors to TRIQS and the open-source scientific Python ecosystem.
The Flatiron Institute is a division of the Simons Foundation.

# References
