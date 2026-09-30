"""
Lab 02 — Métricas de recuperación y de abstención sobre el golden set.

    python evaluation.py --golden golden_set.json --k 5
    python evaluation.py --golden golden_set.json --k 3 --sin-generar     # solo recuperación
    QDRANT_URL=":memory:" python evaluation.py --corpus ejemplos --golden golden_ejemplo.json

Qué mide, y por qué así:

1. **Las negativas no entran en el Hit Rate ni en el MRR.** Una pregunta sin documento
   esperado —las negativas obligatorias, cuya respuesta correcta es abstenerse— no tiene
   nada que recuperar: si contara como fallo, con tres negativas de diez un sistema
   **perfecto** reportaría 0,70 y no podría subir.
2. **El golden set se anota por documento y por fragmento literal**, no por `chunk_id`: un
   `chunk_id` cambia al re-fragmentar, y la Opción A del taller invalidaría las anotaciones
   sin fallar. Un acierto es «algún fragmento recuperado pertenece a un documento
   fuente y, si se dio un fragmento esperado, lo contiene».
3. **La tasa de abstención se mide de verdad**, generando, y en sus dos direcciones: la
   correcta (negativas en las que el sistema se abstuvo) y la indebida (respondibles en las
   que también se abstuvo). Son dos números, no uno (sesión 09). La detección compara
   normalizado, con la misma constante que el prompt.
4. **Una fila sin rellenar se cuenta y se avisa.** Las filas de la plantilla que siguen con
   `REEMPLAZAR` no se evalúan —no hay nada que evaluar—, pero saltarlas en silencio haría
   que un golden set a medio anotar reportara métricas sobre menos preguntas de las que el
   taller pide, sin que nada lo dijera. El resumen trae `n_sin_rellenar` e
   `ids_sin_rellenar`, y la función lo imprime.
5. **Escribe `resultados.csv`** con una fila por consulta: es el entregable crudo del que se
   derivan todas las tablas del informe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from rag_pipeline import RagPipeline, se_abstuvo, _normalizar_texto


def load_golden_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def es_respondible(item: dict) -> bool:
    """Tiene al menos un documento fuente. Las negativas no; la adversarial, según el
    estudiante decida (ver la plantilla)."""
    return bool(item.get("documentos_fuente"))


def acierta(hit: dict, item: dict) -> bool:
    if hit["document"] not in item.get("documentos_fuente", []):
        return False
    esperado = (item.get("fragmento_esperado") or "").strip()
    return not esperado or _normalizar_texto(esperado) in _normalizar_texto(hit["text"])


def posicion_del_primer_acierto(hits: list[dict], item: dict) -> int | None:
    for pos, hit in enumerate(hits, start=1):
        if acierta(hit, item):
            return pos
    return None


def _media(valores: list[float]) -> float:
    return sum(valores) / len(valores) if valores else math.nan


def evaluate_retrieval(golden_set: list[dict], pipeline: RagPipeline, k: int = 5,
                       generar: bool = True) -> dict:
    rows = []
    sin_rellenar = []
    for item in golden_set:
        pregunta = item["pregunta"]
        if not pregunta or pregunta.startswith("REEMPLAZAR"):
            # Fila de la plantilla sin rellenar. NO se evalúa, pero se cuenta y se avisa:
            # saltarla en silencio hacía que un golden set a medio anotar reportara
            # métricas sobre menos preguntas de las que el taller pide, sin que nada lo
            # dijera. Es la misma falla silenciosa que la Parte 0 hace reproducir.
            sin_rellenar.append(item.get("id"))
            continue
        respondible = es_respondible(item)
        hits = pipeline.retrieve(pregunta, top_k=k)
        pos = posicion_del_primer_acierto(hits, item) if respondible else None
        fila = {
            "id": item["id"],
            "tipo": item.get("tipo", "sin_tipo"),
            "respondible": respondible,
            "k": k,
            "hit": (pos is not None) if respondible else None,
            "reciprocal_rank": (1 / pos if pos else 0.0) if respondible else None,
            "posicion": pos,
            "score_top1": hits[0]["score"] if hits else None,
            "retrieved_ids": [h["chunk_id"] for h in hits],
            "abstuvo": None,
            "generador": None,
        }
        if generar:
            salida = pipeline.answer(pregunta, top_k=k)
            fila["abstuvo"] = salida["abstained"]
            fila["generador"] = salida["generator"]
            fila["respuesta"] = salida["answer"][:500]
        rows.append(fila)

    respondibles = [r for r in rows if r["respondible"]]
    negativas = [r for r in rows if not r["respondible"]]
    resumen = {
        "k": k,
        "n_preguntas": len(rows),
        "n_sin_rellenar": len(sin_rellenar),
        "ids_sin_rellenar": sin_rellenar,
        "n_respondibles": len(respondibles),
        "n_negativas": len(negativas),
        "hit_rate": _media([float(r["hit"]) for r in respondibles]),
        "mrr": _media([r["reciprocal_rank"] for r in respondibles]),
        "abstencion_correcta": _media([float(r["abstuvo"]) for r in negativas if r["abstuvo"] is not None]),
        "abstencion_indebida": _media([float(r["abstuvo"]) for r in respondibles if r["abstuvo"] is not None]),
        "por_tipo": {},
        "rows": rows,
    }
    for tipo in sorted({r["tipo"] for r in rows}):
        del_tipo = [r for r in rows if r["tipo"] == tipo]
        resp = [r for r in del_tipo if r["respondible"]]
        resumen["por_tipo"][tipo] = {
            "n": len(del_tipo),
            "hit_rate": _media([float(r["hit"]) for r in resp]),
            "mrr": _media([r["reciprocal_rank"] for r in resp]),
            "abstuvo": _media([float(r["abstuvo"]) for r in del_tipo if r["abstuvo"] is not None]),
        }
    if sin_rellenar:
        print(f"AVISO: {len(sin_rellenar)} de {len(golden_set)} preguntas del golden set "
              f"siguen sin rellenar y NO se evaluaron (ids {sin_rellenar}). "
              f"Las métricas de abajo se calcularon sobre {len(rows)}.")
    return resumen


def escribir_csv(rows: list[dict], path: Path, modelo_embeddings: str, anexar: bool = False,
                 recuperacion: str = "denso") -> None:
    """Una fila por consulta: el crudo del que se derivan las tablas del informe.
    Con `anexar`, añade filas a un CSV existente (k=3 y k=5 en un solo archivo)."""
    columnas = ["id", "tipo", "respondible", "modelo_embeddings", "recuperacion", "k", "posicion",
                "hit", "reciprocal_rank", "score_top1", "abstuvo", "generador", "retrieved_ids",
                "respuesta"]
    nuevo = not (anexar and path.exists())
    with path.open("w" if nuevo else "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        if nuevo:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in columnas}
                       | {"modelo_embeddings": modelo_embeddings, "recuperacion": recuperacion,
                          "retrieved_ids": " ".join(r["retrieved_ids"])})


def _fmt(x) -> str:
    return "sin medir" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--csv", type=Path, default=Path("resultados.csv"))
    ap.add_argument("--sin-generar", action="store_true",
                    help="solo recuperación: no llama a ningún LLM y no mide abstención")
    ap.add_argument("--sin-indexar", action="store_true",
                    help="usa la colección ya indexada en Qdrant (no sirve con :memory:)")
    ap.add_argument("--anexar", action="store_true",
                    help="añade las filas al CSV existente en vez de reescribirlo")
    args = ap.parse_args()

    from rag_pipeline import EMBEDDING_MODEL
    pipeline = RagPipeline()
    if not args.sin_indexar:
        chunks = pipeline.ingest(args.corpus)
        if not chunks:
            raise SystemExit(f"ningún fragmento indexable en {args.corpus}/")
        pipeline.index(chunks)
    golden_set = load_golden_set(args.golden)
    report = evaluate_retrieval(golden_set, pipeline, k=args.k, generar=not args.sin_generar)
    escribir_csv(report["rows"], args.csv, EMBEDDING_MODEL, anexar=args.anexar)

    print(f"golden set: {report['n_preguntas']} preguntas ({report['n_respondibles']} respondibles, "
          f"{report['n_negativas']} negativas) · k = {report['k']} · modelo {EMBEDDING_MODEL}")
    print(f"Hit Rate@{args.k} (respondibles): {_fmt(report['hit_rate'])}")
    print(f"MRR (respondibles):          {_fmt(report['mrr'])}")
    print(f"abstención correcta:         {_fmt(report['abstencion_correcta'])}   (negativas)")
    print(f"abstención indebida:         {_fmt(report['abstencion_indebida'])}   (respondibles)")
    for tipo, m in report["por_tipo"].items():
        print(f"  {tipo:<12} n={m['n']}  hit={_fmt(m['hit_rate'])}  mrr={_fmt(m['mrr'])}  abstuvo={_fmt(m['abstuvo'])}")
    print(f"filas crudas en {args.csv}")
