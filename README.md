# Official Repository for "Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation" [ICLR 2024]

__Authors__: Luca Eyring*, Dominik Klein*, Théo Uscidda*, Giovanni Palla, Niki Kilbertus, Zeynep Akata, Fabian Theis

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.15100)

We have split up the codebase for each method into one repository as follows:
- This repositoriy contains all conducted `Flow Matching` experiments implemented leveraging `Jax`, `Equinox`, and `ott-jax`.
- The code for the single-cell trajcetory inference experiments using `OT-ICNN` can be found at https://github.com/theislab/moscot_not.
- For the single-cell pertubations experiments using `Monge Gap`, we plan to release the code soon.

---

## Overview

### Abstract

In optimal transport (OT), a Monge map is known as a mapping that transports a source distribution to a target distribution in the most cost-efficient way. Recently, multiple neural estimators for Monge maps have been developed and applied in diverse unpaired domain translation tasks, e.g. in single-cell biology and computer vision. However, the classic OT framework enforces mass conservation, which makes it prone to outliers and limits its applicability in real-world scenarios. The latter can be particularly harmful in OT domain translation tasks, where the relative position of a sample within a distribution is explicitly taken into account. While unbalanced OT tackles this challenge in the discrete setting, its integration into neural Monge map estimators has received limited attention. We propose a theoretically grounded method to incorporate unbalancedness into any Monge map estimator. We improve existing estimators to model cell trajectories over time and to predict cellular responses to perturbations. Moreover, our approach seamlessly integrates with the OT flow matching (OT-FM) framework. While we show that OT-FM performs competitively in image translation, we further improve performance by incorporating unbalancedness (UOT-FM), which better preserves relevant features. We hence establish UOT-FM as a principled method for unpaired image translation.

![](assets/emnist_concept.png "EMNIST Unbalancedness concept")

---

## Setup

The code will be released in the coming days.

## Experiments


![](assets/celeba256_samples.png)

## Citation

```bibtex
@inproceedings{eyring2024unbalancedness,
    title={Unbalancedness in Neural Monge Maps Improves Unpaired Domain Translation},
    author={Luca Eyring and Dominik Klein and Th{\'e}o Uscidda and Giovanni Palla and Niki Kilbertus and Zeynep Akata and Fabian J Theis},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=2UnCj3jeao}
}
```