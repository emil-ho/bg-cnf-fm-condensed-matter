# Continuous Normalizing Flows with Flow Matching for Free Energy Estimation in Condensed Matter
[![arXiv](https://img.shields.io/badge/arXiv-2512.23930-b31b1b.svg)](https://arxiv.org/abs/2512.23930)
[![J. Chem. Phys.](https://img.shields.io/badge/J._Chem._Phys.-published-1DA462.svg)](https://doi.org/10.1063/5.0320214)

This repository contains the code for the continuous normalizing flows in the paper "Assessing generative modeling approaches for free energy estimates in condensed matter".

## Abstract
The accurate estimation of free energy differences between two states is a long-standing challenge in molecular simulations. Traditional approaches generally rely on sampling multiple intermediate states to ensure sufficient overlap in phase space and are, consequently, computationally expensive. Boltzmann generators and related generative-model-based methods have recently addressed this challenge by learning a direct probability density transform between two states. However, it remains unclear which approach provides the best trade-off between efficiency, accuracy, and scalability. In this work, we review and benchmark selected generative approaches for condensed-matter systems, including discrete and continuous normalizing flows for targeted free energy perturbation and FEAT (Free Energy Estimators with Adaptive Transport) combined with the escorted Jarzynski equality, using coarse-grained monatomic ice and Lennard-Jones solids as benchmark systems. All models yield highly accurate free energy estimates and, depending on the system, may require fewer energy evaluations than traditional methods. Continuous flows and FEAT are most efficient in energy evaluations, whereas discrete flows have substantially lower inference costs. By releasing all data together with our results, we enable future benchmarking of free energy estimation methods in condensed-phase systems.

## Citation

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

## License
This project is licensed under the MIT License, except for src/optimizer.py, which is licensed under the Apache License 2.0 © Meta Platforms, In
