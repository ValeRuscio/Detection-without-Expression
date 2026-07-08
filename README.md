# On the Tip of the LLM

This repository contains the official code and data for the paper:

**On the Tip of the LLM: A Selection Margin Account of  Hallucination**

We investigate factual hallucination across autoregressive language models. Behavioral recognition probes suggest that some answers are more available than direct generation reveals, but they do not define a stable notion of what the model knows at the item level. The read/write distinction gives a more mechanical target: whether the gold answer token can be decoded from intermediate residual states under type controls, and whether it receives enough final support to rank first.

Across models, many direct generation failures are readable but not selected, even under hard decoy controls and a tuned lens replication. Calibrated interventions show that answer direction support can causally change first token selection, and differences between models show when the support of the selected alternative is also limiting. The margin decomposition explains why answer support at successful levels is not always enough: the final readout contains a context averaged, frequency linked baseline that usually favors the selected alternative, and the item specific contextual support may also point partly toward that alternative.

The component results make the account more mechanistic. A stable set of late attention and MLP components produces much of the readout margin, and patching between paraphrases of the same fact causally transfers support from successful prompts to failed prompts. The account is deliberately scoped: a single first token patch does not solve full factual generation. It isolates one concrete way in which internally decodable factual evidence can fail to determine generation.

---

## 📁 Repository Structure


### `notebooks/`

Contains the full Jupyter notebooks used to run all experiments and analyses reported in the paper.

These notebooks are self-contained and document the complete experimental pipeline, including representation extraction, geometric metrics, functional probes, and causal interventions.

---

## 📄 Citation

If you use this repository, please cite the accompanying paper.

---

## 📬 Contact

For questions or issues, please open a GitHub issue or contact the authors.
