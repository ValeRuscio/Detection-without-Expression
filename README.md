# The Phenomenology of Hallucinations

This repository contains the official code and data for the paper:

**The Phenomenology of Hallucinations**

We investigate hallucination across autoregressive language models and diffusion-based image generators, showing that models reliably detect uncertainty internally but fail to integrate it into output generation due to geometric compartmentalization.

---

## 📁 Repository Structure


### `notebooks/`

Contains the full Jupyter notebooks used to run all experiments and analyses reported in the paper.

These notebooks are self-contained and document the complete experimental pipeline, including representation extraction, geometric metrics, functional probes, and causal interventions.

---

### `dataset/`

Contains the evaluation datasets created for this study:

- Two datasets for language models (factual and confabulation regimes)
- One dataset for diffusion models (paradoxical prompts)

---

### `data/`

Contains precomputed analysis outputs for all evaluated models.

These files allow reproduction of figures and tables without rerunning model inference.

---

## 🔁 Reproducing Results

To reproduce the analysis:

1. Use the notebooks in `notebooks/`
2. Load datasets from `dataset/`
3. Generated outputs will match the artifacts provided in `data/`

---

## 📄 Citation

If you use this repository, please cite the accompanying paper.

---

## 📬 Contact

For questions or issues, please open a GitHub issue or contact the authors.
