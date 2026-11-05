# Continuous Normalizing Flows with Flow Matching for Free Energy Estimation in Condensed Matter
This repository contains the PyTorch implementation of Continuous Normalizing Flows (CNFs) trained via Riemannian Flow Matching for free energy estimation in condensed matter systems. 
This codebase supports two publications:

**1. Assessing generative modeling approaches for free energy estimates in condensed matter**  
[![J. Chem. Phys.](https://img.shields.io/badge/J._Chem._Phys.-10.1063%2F5.0320214-1DA462.svg)](https://doi.org/10.1063/5.0320214)
[![arXiv](https://img.shields.io/badge/arXiv-2512.23930-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.23930)

The baseline CNF implementation used in our multi-model benchmark study. 

**2. Boltzmann Generators for Condensed Matter via Riemannian Flow Matching**  
[![ICLR AI4Mat Workshop](https://img.shields.io/badge/ICLR_AI4Mat_Workshop-accepted-0053D6.svg)](https://openreview.net/forum?id=FTIxxfPLSG)
[![arXiv](https://img.shields.io/badge/arXiv-2602.18482-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18482)

Extends the baseline model with an updated vector field architecture and a bias correction for the log-density change of the samples.


## Abstracts

<details>
<summary><b>1. Assessing generative modeling approaches for free energy estimates in condensed matter (J. Chem. Phys.)</b></summary>
The accurate estimation of free energy differences between two states is a long-standing challenge in molecular simulations. Traditional approaches generally rely on sampling multiple intermediate states to ensure sufficient overlap in phase space and are, consequently, computationally expensive. Boltzmann generators and related generative-model-based methods have recently addressed this challenge by learning a direct probability density transform between two states. However, it remains unclear which approach provides the best trade-off between efficiency, accuracy, and scalability. In this work, we review and benchmark selected generative approaches for condensed-matter systems, including discrete and continuous normalizing flows for targeted free energy perturbation and FEAT (Free Energy Estimators with Adaptive Transport) combined with the escorted Jarzynski equality, using coarse-grained monatomic ice and Lennard-Jones solids as benchmark systems. All models yield highly accurate free energy estimates and, depending on the system, may require fewer energy evaluations than traditional methods. Continuous flows and FEAT are most efficient in energy evaluations, whereas discrete flows have substantially lower inference costs. By releasing all data together with our results, we enable future benchmarking of free energy estimation methods in condensed-phase systems.
</details>

<details>
<summary><b>2. Boltzmann Generators for Condensed Matter via Riemannian Flow Matching (ICLR AI4Mat Workshop, 2026)</b></summary>
Sampling equilibrium distributions is fundamental to statistical mechanics. While flow matching has emerged as scalable state-of-the-art paradigm for generative modeling, its potential for equilibrium sampling in condensed-phase systems remains largely unexplored. We address this by incorporating the periodicity inherent to these systems into continuous normalizing flows using Riemannian flow matching. The high computational cost of exact density estimation intrinsic to continuous normalizing flows is mitigated by using Hutchinson’s trace estimator, utilizing a crucial bias-correction step based on cumulant expansion to render the stochastic estimates suitable for rigorous thermodynamic reweighting. Our approach is validated on monatomic ice, demonstrating the ability to train on systems of unprecedented size and obtain highly accurate free energy estimates without the need for traditional multistage estimators.
</details>

## Citations

<details>
<summary><b>1. Assessing generative modeling approaches for free energy estimates in condensed matter (J. Chem. Phys.)</b></summary>

```
@article{10.1063/5.0320214,
    author = {Schebek, Maximilian and He, Jiajun and Hoffmann, Emil and Du, Yuanqi and Noé, Frank and Rogal, Jutta},
    title = {Assessing generative modeling approaches for free energy estimates in condensed matter},
    journal = {The Journal of Chemical Physics},
    volume = {164},
    number = {18},
    pages = {184111},
    year = {2026},
    month = {05},
    issn = {0021-9606},
    doi = {10.1063/5.0320214},
    url = {https://doi.org/10.1063/5.0320214},
    eprint = {https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/5.0320214/21000702/184111_1_5.0320214.pdf},
}
```

</details>


<details>
<summary><b>2. Boltzmann Generators for Condensed Matter via Riemannian Flow Matching (ICLR AI4Mat Workshop, 2026)</b></summary>

```
@misc{hoffmann2026boltzmanngeneratorscondensedmatter,
      title={Boltzmann Generators for Condensed Matter via Riemannian Flow Matching}, 
      author={Emil Hoffmann and Maximilian Schebek and Leon Klein and Frank Noé and Jutta Rogal},
      year={2026},
      eprint={2602.18482},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2602.18482},
      note={Published as a workshop paper at AI4Mat, ICLR 2026},
}
```

</details>



## License
This project is licensed under the MIT License, except for src/optimizer.py, which is licensed under the Apache License 2.0 © Meta Platforms, In
