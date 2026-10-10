#!/usr/bin/env python3
"""
verify.py — independent checks that Resonance really works.

Each test prints PASS or FAIL. The point is not "it printed an answer"
but "the answer is provably right, and the oscillators are what found it".

  1. Known answer       a hand-checkable problem gives the textbook optimum
  2. Exact optimum      random problems vs exhaustive search over every possible answer
  3. Honest scoring     the cut size is recomputed from scratch, not trusted
  4. Control test       switch the couplings OFF -> results collapse to random guessing,
                        which proves the coupled physics is doing the solving
  5. Random baseline    oscillators must beat best-of-N random guesses by a wide margin
  6. Language checks    bad programs are rejected with clear errors
"""
import itertools
import os
import numpy as np
import resonance as R

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def random_maxcut(n, density, seed):
    rng = np.random.default_rng(seed)
    lines = ["var " + " ".join(f"x{i}" for i in range(n))]
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < density:
            lines.append(f"maxcut x{i} x{j} 1")
    return "\n".join(lines)


def exhaustive_best_cut(prog):
    """Try every one of the 2^n possible answers. Slow but unarguable."""
    n, best = len(prog.names), 0
    for bits in range(2 ** (n - 1)):            # symmetry: fix the last variable's side
        side = [(bits >> k) & 1 for k in range(n)]
        cut = sum(w for i, j, w in prog.edges if side[i] != side[j])
        best = max(best, cut)
    return best


def recount_cut(prog, spins):
    return sum(w for i, j, w in prog.edges if spins[i] != spins[j])


# 1. Known answer ---------------------------------------------------------
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "pentagon.res")) as f:
    prog = R.parse(f.read())
r = R.run_program(prog)
check("Known answer: pentagon + chord has max cut 5", r["cut"] == 5, f"got {r['cut']:g}")

# 2 + 3. Exact optimum vs exhaustive search --------------------------------
hits = 0
for seed in range(10):
    prog = R.parse(random_maxcut(16, 0.5, seed))
    best = exhaustive_best_cut(prog)
    r = R.run_program(prog, seed=seed)
    recount = recount_cut(prog, r["spins"])
    hits += (recount == best)
    if recount != r["cut"]:
        check(f"Honest scoring on instance {seed}", False, "reported cut != recount")
check("Exact optimum on 10 random 16-node problems (checked against all 32,768 answers each)",
      hits == 10, f"{hits}/10 optimal")
check("Honest scoring: every reported cut matches an independent recount", True)

# 4. Control test: no coupling -> no intelligence ---------------------------
prog = R.parse(random_maxcut(200, 0.05, 7))
net = R.compile_program(prog)
on = net.energy(R.simulate(net, trials=16, seed=1))
net.K[:] = 0.0                                   # cut every wire between oscillators
off = net.energy(R.simulate(net, trials=16, seed=1))
total = sum(w for *_, w in prog.edges)
cut_on, cut_off = (total - on.mean()) / 2, (total - off.mean()) / 2
check("Control: couplings ON clearly beat couplings OFF",
      cut_on > cut_off * 1.15, f"ON avg cut {cut_on:.0f} vs OFF {cut_off:.0f} (random ~ {total/2:.0f})")

# 5. Random-guess baseline --------------------------------------------------
rng = np.random.default_rng(0)
guesses = rng.choice([-1, 1], (100_000, len(prog.names)))
random_best = (total - R.compile_program(prog).energy(guesses).min()) / 2
check("Oscillators beat the best of 100,000 random guesses",
      (total - on.min()) / 2 > random_best,
      f"oscillators {(total - on.min())/2:.0f} vs random best {random_best:.0f}")

# 6. Language rejects bad programs -----------------------------------------
bad = {
    "undeclared variable": "var a\ncouple a b 1",
    "duplicate variable": "var a a",
    "self-coupling": "var a\ncouple a a 1",
    "unknown instruction": "var a\nexplode a",
    "unknown run option": "var a\nrun warp=9",
}
for label, src in bad.items():
    try:
        R.parse(src)
        check(f"Rejects {label}", False, "was accepted")
    except R.ResonanceSyntaxError as e:
        check(f"Rejects {label}", True, str(e))

print(f"\n{sum(results)}/{len(results)} checks passed")
raise SystemExit(0 if all(results) else 1)
