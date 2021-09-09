# Chosen Ciphertext k-Trace Attacks on Masked CCA2 Secure Kyber

This is the source code repo of the paper 
*Chosen Ciphertext k-Trace Attacks on Masked CCA2 Secure Kyber*
to be published in TCHES Volume 2021/4.

ePrint: https://eprint.iacr.org/2021/956

## License

Dual license MIT + Apache v2

## Setup

- Python virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- RUST (see https://rustup.rs/):
optional: you can adjust `CARGO_HOME` and `RUSTUP_HOME` before install
```
curl https://sh.rustup.rs -sSf | sh -s
```

## Test Run 
- run simple example experiment (only 2 iterations, instead of ~200):
```
# Don't forget to source the python virtual environment
bash run_experiments.sh
```

## Results

- are located in /runs as tarballs
