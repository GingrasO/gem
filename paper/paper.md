---
title: 'GEMSTONE(S): Ghost Embedding Method for Static and Thermal Observables in Nonperturbative Electronic Systems'
tags:
  - Python
  - physics
  - quantum embedding
  - ghost Gutzwiller Approximation (ghost-GA)
  - Dynamical Mean-Field Theory (DMFT)
authors:
  - name: Samuele Giuli
    orcid: 0009-0004-7341-3655
    #equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
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
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

GEMSTONE(S) (Ghost Embedding Method for STatic Observables in Nonperturbative Electron systems) is an open-source Python package for solving strongly correlated electron problems within the ghost-Gutzwiller approximation at zero and finite temperature. The code implements a functional formulation of the ghost-Gutzwiller method that establishes a direct connection with dynamical mean-field theory (DMFT), enabling the computation of static observables in correlated lattice models with improved accuracy and flexibility over conventional variational approaches.

The software provides tools for constructing and solving multiorbital correlated-electron models, evaluating thermodynamic and ground-state properties, and studying interaction-driven phenomena such as the Mott transition, quasiparticle renormalization, and orbital-selective correlations. By combining a variational embedding framework with efficient numerical algorithms in Python, GEMSTONE(S) offers a flexible and extensible platform for both methodological developments and applications in condensed matter physics.

# Statement of need

Theoretical and computational studies of strongly correlated electron systems require methods capable of capturing nonperturbative many-body effects while remaining computationally tractable for realistic multiorbital models. Established approaches such as DMFT provide accurate descriptions of local electronic correlations, but often rely on computationally demanding impurity solvers whose complexity increases rapidly with orbital number and temperature. Variational methods based on the Gutzwiller approximation provide an attractive alternative due to their reduced computational cost, but traditional formulations can be difficult to generalize systematically and may lack a transparent connection to Green’s-function-based many-body frameworks.

GEMSTONE(S) addresses this need by implementing the ghost-Gutzwiller approximation within a functional formalism that naturally connects variational embedding methods to DMFT concepts. The package enables researchers to study zero- and finite-temperature properties of correlated lattice models using a unified framework that retains much of the physical insight and efficiency of Gutzwiller methods while extending their applicability and interpretability.

The software is primarily intended for researchers in condensed matter physics, computational many-body theory, and materials modeling. Potential users include scientists investigating Hubbard-like models, multiorbital correlation effects, finite-temperature phase diagrams, and methodological developments in variational embedding theories.

# State of the field

Several software packages are available for studying strongly correlated electron systems within dynamical mean-field theory (DMFT) and related embedding approaches. General-purpose frameworks such as TRIQS provide flexible infrastructures for DMFT calculations together with interfaces to multiple impurity solvers. A broad ecosystem of impurity solvers is also available, including exact diagonalization packages such as EDIpack, XDiag, and Pomerol, as well as density matrix renormalization group (DMRG) based solvers implemented in libraries such as ITensor, BLOCK, and ALPS. Continuous-time quantum Monte Carlo solvers and tensor-network approaches further extend the range of available numerical techniques for correlated-electron problems.

Within this landscape, GEMSTONE(S) is not intended to replace highly optimized impurity solvers, but rather to provide a complementary variational embedding framework based on the ghost-Gutzwiller approximation. The functional formulation implemented in GEMSTONE(S) establishes a direct connection with DMFT and therefore naturally enables interoperability with existing impurity-solver ecosystems. In particular, the formalism can exploit external solvers such as EDIpack, XDiag, Pomerol, and DMRG-based implementations to benchmark results, construct reference solutions, or extend the treatment of local correlations beyond the variational approximation.

Compared to conventional DMFT workflows, GEMSTONE(S) focuses on the efficient evaluation of static observables and thermodynamic quantities at zero and finite temperature within a variational framework that is computationally lighter than full impurity-based self-consistent calculations. This makes the software particularly attractive for exploratory studies of multiorbital systems, large parameter scans, and methodological developments where the computational cost of fully dynamical impurity solvers may become prohibitive.

The Python implementation additionally promotes accessibility, rapid prototyping, and integration with the broader scientific Python ecosystem, facilitating interaction with external numerical libraries and many-body solvers already widely used by the correlated-electron community.

# Implementation and availability

GEMSTONE(S) is implemented in Python and leverages the scientific Python ecosystem for numerical linear algebra, optimization, and data analysis. The code is organized in a modular fashion, separating model construction, variational optimization, finite-temperature solvers, and observable evaluation. This design facilitates extensibility and allows researchers to adapt the framework to new Hamiltonians, embedding schemes, and methodological developments.

The package supports calculations for generic correlated-electron lattice models, including multiorbital Hubbard Hamiltonians, and provides routines for evaluating ground-state and finite-temperature observables within the ghost-Gutzwiller approximation. Numerical workflows are accessible through Python interfaces, enabling interactive exploration and integration with external analysis tools.

The source code is openly available under an open-source license and hosted on a public version-control platform, ensuring reproducibility and community access. Documentation, examples, and installation instructions are provided alongside the repository to support both new users and developers.

# Acknowledgements

Acknowledge contributors, funding, and support.

# References
