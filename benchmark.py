#!/usr/bin/env python3
"""Benchmark Resonance's oscillator machine against exact search and simulated annealing.
Every problem is emitted as Resonance source text and goes through parse -> compile -> simulate."""
import time
import numpy as np
from resonance import parse, run_program, compile_program


def maxcut_source(n, density, seed):
    rng = np.random.default_rng(seed)
    lines = ["var " + " ".join(f"x{i}" for i in range(n))]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                lines.append(f"maxcut x{i} x{j} 1")
    return "\n".join(lines)


def spinglass_source(n, seed):
    rng = np.random.default_rng(seed)
    lines = ["var " + " ".join(f"x{i}" for i in range(n))]
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"couple x{i} x{j} {rng.choice([-1, 1])}")
        lines.append(f"bias x{i} {rng.normal():.3f}")
    return "\n".join(lines)


def brute_force(net):
    n, best = net.n, np.inf
    for start in range(0, 2 ** n, 1 << 16):
        idx = np.arange(start, min(start + (1 << 16), 2 ** n))
        S = ((idx[:, None] >> np.arange(n)) & 1) * 2 - 1
        best = min(best, net.energy(S).min())
    return best


def annealing(net, sweeps, seed):
    rng = np.random.default_rng(seed)
    J, h, n = net.J_raw, net.h_raw, net.n
    s = rng.choice([-1.0, 1.0], n)
    field = J @ s + h
    E = float(net.energy(s)[0])
    best = E
    scale = np.abs(J).sum(1).max() + np.abs(h).max()
    for T in np.geomspace(scale, 0.01, sweeps):
        u = rng.random(n)
        for k, i in enumerate(rng.permutation(n)):
            dE = -2 * s[i] * field[i]
            if dE <= 0 or u[k] < np.exp(-dE / T):
                s[i] = -s[i]
                field += 2 * s[i] * J[:, i]
                E += dE
                best = min(best, E)
    return best


def cut_from_energy(prog, E):
    total = sum(w for _, _, w in prog.edges)
    return (total - E) / 2


print("1) Small Max-Cut (n=18, exact optimum known)")
for seed in range(5):
    prog = parse(maxcut_source(18, 0.5, seed))
    net = compile_program(prog)
    opt = brute_force(net)
    r = run_program(prog, seed=seed)
    print(f"   instance {seed}: optimum cut {cut_from_energy(prog, opt):.0f} | "
          f"oscillators {r['cut']:.0f} | trials hitting optimum "
          f"{np.mean(np.isclose(r['energies'], opt)):.0%}")

print("\n2) Spin glass with biases (n=18) — tests the reference oscillator")
for seed in range(3):
    prog = parse(spinglass_source(18, seed))
    net = compile_program(prog)
    opt = brute_force(net)
    r = run_program(prog, seed=seed)
    print(f"   instance {seed}: optimum E {opt:.3f} | oscillators {r['energy']:.3f} | "
          f"hit rate {np.mean(np.isclose(r['energies'], opt)):.0%}")

print("\n3) Large Max-Cut (n=300, density 0.05) vs simulated annealing")
for seed in range(3):
    prog = parse(maxcut_source(300, 0.05, 100 + seed))
    net = compile_program(prog)
    r = run_program(prog, seed=seed)
    t0 = time.perf_counter()
    sa = annealing(net, sweeps=1000, seed=seed)
    sa_t = time.perf_counter() - t0
    print(f"   instance {seed}: oscillators cut {r['cut']:.0f} ({r['seconds']:.1f}s, 32 trials) | "
          f"annealing cut {cut_from_energy(prog, sa):.0f} ({sa_t:.1f}s, 1 run)")
