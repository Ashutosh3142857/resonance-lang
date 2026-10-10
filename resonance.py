#!/usr/bin/env python3
"""
Resonance v0.1 — a tiny language that compiles optimization problems onto a
network of coupled oscillators (an Oscillator Ising Machine), then lets the
physics settle to the answer.

Every variable becomes an oscillator. Every constraint becomes a coupling
between oscillators. The answer is read from each oscillator's PHASE:
phase ~ 0  -> +1,  phase ~ pi -> -1.

Program format (.res)
---------------------
    var a b c d            # declare binary variables (spins, +1/-1)
    couple a b 1.0         # energy term  +1.0 * s_a * s_b
    bias a -0.5            # energy term  -0.5 * s_a
    maxcut a b 1.0         # Max-Cut edge (sugar for couple, also reports cut size)
    run steps=4000 dt=0.2 trials=32 seed=0 ks_max=0.08 noise0=0.15
                           # ks_max = phase-locking strength, noise0 = initial noise

The machine MINIMIZES  E(s) = sum_{i<j} J_ij s_i s_j + sum_i h_i s_i

Physics (gradient flow of the oscillator Lyapunov function):
    dphi_i/dt = sum_j J_ij sin(phi_i - phi_j) - Ks(t) sin(2 phi_i) + noise(t)
The sin(2 phi) term is second-harmonic injection locking: it forces each
oscillator into one of two phases (0 or pi), i.e. binarizes the answer.
Biases are realized with an extra reference oscillator.
"""
import sys
import time
import numpy as np


class ResonanceSyntaxError(Exception):
    pass


class Program:
    def __init__(self):
        self.names, self.index = [], {}
        self.J, self.h = {}, {}
        self.edges = []          # Max-Cut edges, for reporting cut size
        self.run = dict(steps=4000, dt=0.2, trials=32, seed=0, ks_max=0.08, noise0=0.15)


def parse(src):
    p = Program()
    for ln, raw in enumerate(src.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        op, *args = line.split()

        def idx(name):
            if name not in p.index:
                raise ResonanceSyntaxError(f"line {ln}: undeclared variable '{name}'")
            return p.index[name]

        if op == "var":
            for a in args:
                if a in p.index:
                    raise ResonanceSyntaxError(f"line {ln}: '{a}' declared twice")
                p.index[a] = len(p.names)
                p.names.append(a)
        elif op in ("couple", "maxcut"):
            if len(args) != 3:
                raise ResonanceSyntaxError(f"line {ln}: {op} needs <var> <var> <weight>")
            i, j, w = idx(args[0]), idx(args[1]), float(args[2])
            if i == j:
                raise ResonanceSyntaxError(f"line {ln}: cannot couple a variable to itself")
            key = (min(i, j), max(i, j))
            p.J[key] = p.J.get(key, 0.0) + w
            if op == "maxcut":
                p.edges.append((i, j, w))
        elif op == "bias":
            if len(args) != 2:
                raise ResonanceSyntaxError(f"line {ln}: bias needs <var> <weight>")
            i = idx(args[0])
            p.h[i] = p.h.get(i, 0.0) + float(args[1])
        elif op == "run":
            for a in args:
                k, _, v = a.partition("=")
                if k not in p.run:
                    raise ResonanceSyntaxError(f"line {ln}: unknown run option '{k}'")
                p.run[k] = float(v) if k in ("dt", "ks_max", "noise0") else int(float(v))
        else:
            raise ResonanceSyntaxError(f"line {ln}: unknown instruction '{op}'")
    if not p.names:
        raise ResonanceSyntaxError("program declares no variables")
    return p


class Network:
    """Compiled form: the physical parameters of an oscillator network."""

    def __init__(self, program):
        n = len(program.names)
        self.n = n
        self.J_raw = np.zeros((n, n))
        for (i, j), w in program.J.items():
            self.J_raw[i, j] = self.J_raw[j, i] = w
        self.h_raw = np.zeros(n)
        for i, w in program.h.items():
            self.h_raw[i] = w
        self.has_ref = bool(np.any(self.h_raw))
        N = n + (1 if self.has_ref else 0)
        K = np.zeros((N, N))
        K[:n, :n] = self.J_raw
        if self.has_ref:              # bias h_i == coupling to an oscillator pinned at +1
            K[:n, n] = K[n, :n] = self.h_raw
        scale = np.abs(K).sum(1).max() or 1.0
        self.K = K / scale            # normalized coupling matrix (stable integration)

    def energy(self, spins):
        spins = np.atleast_2d(spins)
        return 0.5 * np.sum((spins @ self.J_raw) * spins, 1) + spins @ self.h_raw


def compile_program(program):
    return Network(program)


def simulate(net, steps=4000, dt=0.2, trials=32, seed=0, ks_max=0.08, noise0=0.15):
    """Integrate the oscillator dynamics for `trials` independent machines in parallel."""
    rng = np.random.default_rng(seed)
    K = net.K
    phi = rng.uniform(0, 2 * np.pi, (trials, K.shape[0]))
    for t in range(steps):
        frac = t / steps
        c, s = np.cos(phi), np.sin(phi)
        coupling = s * (c @ K) - c * (s @ K)        # sum_j K_ij sin(phi_i - phi_j)
        dphi = coupling - ks_max * frac * np.sin(2 * phi)
        phi += dt * dphi + np.sqrt(dt) * noise0 * (1 - frac) * rng.standard_normal(phi.shape)
    spins = np.where(np.cos(phi) >= 0, 1, -1)
    if net.has_ref:                   # read every phase relative to the reference oscillator
        spins = spins[:, :-1] * spins[:, -1:]
    return spins


def run_program(program, **overrides):
    net = compile_program(program)
    opts = {**program.run, **overrides}
    t0 = time.perf_counter()
    spins = simulate(net, **opts)
    elapsed = time.perf_counter() - t0
    E = net.energy(spins)
    b = int(np.argmin(E))
    result = dict(
        spins=spins[b], energy=float(E[b]), energies=E, seconds=elapsed,
        hit_rate=float(np.mean(np.isclose(E, E[b]))),
    )
    if program.edges:
        s = spins[b]
        result["cut"] = float(sum(w for i, j, w in program.edges if s[i] != s[j]))
    return result


def main(argv):
    if len(argv) != 2:
        print("usage: python resonance.py program.res")
        return 1
    with open(argv[1]) as f:
        program = parse(f.read())
    r = run_program(program)
    print("assignment:", " ".join(f"{n}={'+' if v > 0 else '-'}"
                                   for n, v in zip(program.names, r["spins"])))
    print(f"energy: {r['energy']:g}")
    if "cut" in r:
        print(f"cut size: {r['cut']:g}")
    print(f"trials reaching best: {r['hit_rate']:.0%}   sim time: {r['seconds']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
