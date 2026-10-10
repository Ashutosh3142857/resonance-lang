# Resonance

**A tiny programming language that solves optimization problems the way nature does: by letting coupled oscillators settle into sync.**

You describe a problem as variables and the links between them. Resonance compiles each variable into an oscillator and each link into a coupling, lets the network settle, and reads the answer from each oscillator's phase (in step = `+`, out of step = `-`).

This is an *oscillator Ising machine* (OIM). Today Resonance simulates the oscillators in software; the same programs are meant to run on physical oscillator chips, where settling takes microseconds and very little energy.

## Quick start

Requires Python 3.9+ and numpy.

```bash
pip install numpy
python resonance.py examples/square.res
```

Output:

```
assignment: a=- b=+ c=- d=+
energy: -4
cut size: 4
trials reaching best: 100%   sim time: 0.08s
```

## The language

A program is a plain-text `.res` file:

```
# Split a square's corners into two teams so every edge crosses teams
var a b c d          # declare variables (each becomes an oscillator)
maxcut a b 1         # Max-Cut edge between a and b, weight 1
maxcut b c 1
maxcut c d 1
maxcut d a 1
run trials=16        # optional settings
```

| Instruction | Meaning |
| --- | --- |
| `var x y z` | Declare binary variables (+1 / -1) |
| `couple x y w` | Add energy term `w * x * y` |
| `bias x w` | Add energy term `w * x` |
| `maxcut x y w` | Max-Cut edge (same as `couple`, also reports the cut size) |
| `run key=value ...` | Settings: `steps`, `dt`, `trials`, `seed`, `ks_max` (locking strength), `noise0` (starting noise) |

Resonance **minimizes** `E = sum J_ij s_i s_j + sum h_i s_i`. Any problem you can write in that form (a QUBO / Ising problem) can be expressed: scheduling, routing, partitioning, portfolio selection, graph colouring with extra encoding, and more.

## How it works

Each oscillator *i* has a phase φᵢ, and the network follows

```
dφᵢ/dt = Σⱼ Jᵢⱼ sin(φᵢ − φⱼ) − Ks(t) sin(2φᵢ) + noise(t)
```

- The **coupling term** pulls linked oscillators in or out of step, lowering the problem's energy.
- The **sin(2φ) term** (sub-harmonic injection locking) ramps up over time and forces every phase to 0 or π, turning the answer into clean +1 / -1 values.
- The **noise** starts high and fades, helping the network escape poor solutions.
- **Biases** are handled with an extra reference oscillator.

Many independent trials run in parallel; the best one is reported.

## Is it real? Run the tests

```bash
python verify.py
```

`verify.py` checks the results instead of trusting them:

1. A hand-checkable problem gives the known optimum.
2. 10 random 16-variable problems are checked against **all 32,768 possible answers**; Resonance finds the true optimum on all of them.
3. Every reported score is recomputed independently.
4. **Control test:** with the couplings switched off, results fall to random-guess level. With them on, they are far better. The oscillators are doing the work.
5. Resonance beats the best of 100,000 random guesses.
6. Invalid programs are rejected with clear errors.

`python benchmark.py` compares against exhaustive search and simulated annealing (within about 1% of annealing on 300-node problems).

## Honest status

- On a normal CPU this is a **simulation**, so it is slower than classical solvers (a few seconds vs. a fraction of a second). Speed and energy advantages require physical oscillator hardware.
- Results are very good but not guaranteed optimal on large problems.
- The underlying method is published research; see the references below. Resonance's contribution is a small, readable language and toolchain for it.

## Roadmap / ideas for contributors

- Hardware-aware compiling: model real chip flaws (frequency mismatch, few coupling levels, limited wiring) and adapt problems to them
- Automatic choice of locking strength and noise schedule
- Higher-level commands (`schedule`, `route`, `assign`) that generate couplings automatically
- GPU backend, and backends for physical oscillator chips and FPGAs
- Multi-state (Potts) oscillators for colouring and scheduling

Pull requests and issues are welcome.

## References

- T. Wang, J. Roychowdhury, *Oscillator-based Ising Machine*, 2017 — https://arxiv.org/abs/1709.08102
- W. Moy et al., *A 1,968-node coupled ring oscillator circuit for combinatorial optimization*, Nature Electronics, 2022 — https://www.nature.com/articles/s41928-022-00749-3
- Y. E. Gonul et al., *GPU-Accelerated Simulated Oscillator Ising/Potts Machine*, 2025 — https://arxiv.org/abs/2505.22631

Anyone building physical oscillator hardware should check existing hardware patents in their jurisdiction.

## License

MIT — see [LICENSE](LICENSE).
