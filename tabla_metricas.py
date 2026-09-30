"""Tablas del informe derivadas SOLO de resultados.csv (Partes 2.b y 3).

    python tabla_metricas.py
"""
import csv
from collections import defaultdict
from pathlib import Path


def media(xs):
    return sum(xs) / len(xs) if xs else float("nan")


grupos = defaultdict(list)
for r in csv.DictReader(Path("resultados.csv").open(encoding="utf-8")):
    grupos[(r["recuperacion"], int(r["k"]))].append(r)

print(f"{'recuperación':<52} {'k':>2} {'HitRate@k':>9} {'MRR':>6} {'abst.correcta':>13} {'abst.indebida':>13}  n(resp/neg)")
for (rec, k), filas in sorted(grupos.items()):
    resp = [f for f in filas if f["respondible"] == "True"]
    neg = [f for f in filas if f["respondible"] == "False"]
    hr = media([f["hit"] == "True" for f in resp])
    mrr = media([float(f["reciprocal_rank"]) for f in resp])
    ac = media([f["abstuvo"] == "True" for f in neg if f["abstuvo"]])
    ai = media([f["abstuvo"] == "True" for f in resp if f["abstuvo"]])
    print(f"{rec:<52} {k:>2} {hr:>9.3f} {mrr:>6.3f} {ac:>13.3f} {ai:>13.3f}  {len(resp)}/{len(neg)}")
