# Trojan Defender

E6998 Robustness and Security in ML Systems (Columbia University)

Eduardo Blancas Reyes (eb3079), Daniel Speyer

## Abstract

Organizations in need of neural nets often outsource the implementation and training of the nets. This opens opportunities for a malicious contractor to insert hidden behavior in the net: a neural trojan. We examine six possible attacks and three possible defenses. So far, no attack evades all defenses and no defense catches all attacks. While our survey of attacks is nowhere near exhaustive, we believe we have seen enough to begin generalizing from our experience.

## Folder structure

* report/
    - Source for final report
* experiments/
    - Notebooks that we used during the project
* gcloud/
    - Some shell scripts for managing the machine we used in Google Compute Engine
* pkg/
    - Python package

## Getting started

Install Python package to run the notebooks:

```shell
pip install pkg/
```

## Testing

Run tests and verify that the notebooks run:

```shell
make test
```
