from fractions import Fraction
from collections import defaultdict

N = 200  # search n up to this value

coincidences = defaultdict(list)

for nf in range(1, N):
    for ni in range(nf + 1, N + 1):
        val = Fraction(1, nf * nf) - Fraction(1, ni * ni)
        coincidences[val].append((nf, ni))

found = {val: pairs for val, pairs in coincidences.items() if len(pairs) > 1}

if not found:
    print(f"No coincident lines found for n up to {N}.")
else:
    print(f"Coincident spectral lines (n up to {N}):\n")
    for val in sorted(found, reverse=True):
        pairs = found[val]
        print(f"  1/nf² - 1/ni² = {val} = {float(val):.8f}")
        for nf, ni in pairs:
            series = {1: "Lyman", 2: "Balmer", 3: "Paschen",
                      4: "Brackett", 5: "Pfund", 6: "Humphreys"}
            name = series.get(nf, f"nf={nf}")
            print(f"    {name:10s}  nf={nf:3d}, ni={ni:3d}")
        print()
