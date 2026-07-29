# Learning to Detect Cryptographic Defects in Ethereum Smart Contracts from Bytecode

Reproduction artifacts for the paper *"Learning to Detect Cryptographic Defects in
Ethereum Smart Contracts from Bytecode."* Source-level analyzers for cryptographic defects
(signature replay, signature malleability, insufficient signature verification,
Merkle-proof misuse, hash collisions, weak on-chain randomness) need verified
Solidity source, but fewer than 1% of deployed contracts are source-verified. This
project distills a source-level analyzer (the *teacher*) into **bytecode-only**
detectors (the *students*), so detection works on the unverified majority of on-chain
code. Everything here runs on a single consumer laptop at zero cloud cost.

## Repository layout

```
scripts/     the full pipeline (fetch -> disassemble -> cluster -> split -> train -> evaluate -> figures)
data/        dataset.csv -- the only provided input: per contract, its 12 defect labels,
             its clone-cluster id, and its train/validation/test split
```

Running the pipeline creates `data/` and `results/` artifacts locally (bytecode, opcodes,
clustering, metrics, figures, trained models); none of these are shipped.


## Requirements

Python 3.10+. Install with:

```
pip install -r requirements.txt
```

CPU is enough for the boosting baseline, ablations, interpretation and figures. The
neural arms (`train_cnn.py`, `train_transformer.py`) use a CUDA GPU (they were trained
on a 4 GB card with mixed precision).



## License and citation

Code is released under the MIT License (see `LICENSE`). See `CITATION.cff` for how to
cite this work. On-chain data derive from public Ethereum state; defect labels derive
from the publicly released source-level analyzer cited in the paper.
