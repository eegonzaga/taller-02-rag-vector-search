"""
Taller 02 — Parte 3, Opción C: reranking con un cross-encoder.

    python parte3_rerank.py --k 3 --anexar     # añade filas recuperacion=rerank a resultados.csv
    python parte3_rerank.py --k 5 --anexar

Modo de falla declarado ANTES de implementar (medido en la Parte 2): el baseline tiene
recall@20 = 1,0 pero Hit Rate@5 = 0,75 y MRR = 0,594; en los fallos (ids 1 y 10) el documento
correcto domina el top-5 y el fragmento con la respuesta está en las posiciones 10 y 15. Es un
fallo de ORDEN dentro del documento correcto, no de cobertura: lo que un cross-encoder, que lee
pregunta y fragmento juntos, puede corregir y un bi-encoder no.

Recupera top-20 con bge-m3 (la misma colección del baseline), reordena con la fila
`rerank_local_ingles` (corpus en inglés) y entrega top-k. Evalúa con el MISMO golden set y el
MISMO evaluador (`evaluation.evaluate_retrieval`), que solo ve `retrieve`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation import _fmt, escribir_csv, evaluate_retrieval, load_golden_set
from rag_pipeline import EMBEDDING_MODEL, RagPipeline, fila_de_la_tabla

FILA_RERANKER = fila_de_la_tabla("rerankers", "rerank_local_ingles")
RERANKER = FILA_RERANKER.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")
CANDIDATOS = 20


class RagPipelineRerank(RagPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(RERANKER)
        self.tope_reranker = int(self.reranker.max_seq_length or 512)
        self.pares_truncados = 0
        self.pares_totales = 0

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        candidatos = super().retrieve(question, top_k=CANDIDATOS)
        pares = [(question, c["text"]) for c in candidatos]
        # El cross-encoder también trunca en silencio (Parte 0.b): se cuenta cuántos pares
        # (pregunta + fragmento) no caben en su tope, en SUS tokens, no en los de bge-m3.
        tok = self.reranker.tokenizer
        for q, t in pares:
            self.pares_totales += 1
            if len(tok(q, t, truncation=False)["input_ids"]) > self.tope_reranker:
                self.pares_truncados += 1
        puntajes = self.reranker.predict(pares)
        for c, s, pos_denso in zip(candidatos, puntajes, range(1, CANDIDATOS + 1)):
            c["score_denso"], c["pos_denso"], c["score"] = c["score"], pos_denso, float(s)
        return sorted(candidatos, key=lambda c: c["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("golden_set.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--csv", type=Path, default=Path("resultados.csv"))
    ap.add_argument("--anexar", action="store_true")
    ap.add_argument("--sin-generar", action="store_true")
    args = ap.parse_args()

    pipeline = RagPipelineRerank()   # usa la colección ya indexada por el baseline
    report = evaluate_retrieval(load_golden_set(args.golden), pipeline, k=args.k,
                                generar=not args.sin_generar)
    escribir_csv(report["rows"], args.csv, EMBEDDING_MODEL, anexar=args.anexar,
                 recuperacion=f"rerank:{RERANKER}@{CANDIDATOS}")

    print(f"reranker: fila rerank_local_ingles · {RERANKER} · verified_at "
          f"{FILA_RERANKER.get('verified_at')} · tope {pipeline.tope_reranker} tokens · "
          f"candidatos {CANDIDATOS} → top-{args.k}")
    print(f"pares truncados por el cross-encoder: {pipeline.pares_truncados}/{pipeline.pares_totales}")
    print(f"Hit Rate@{args.k} (respondibles): {_fmt(report['hit_rate'])}")
    print(f"MRR (respondibles):          {_fmt(report['mrr'])}")
    print(f"abstención correcta:         {_fmt(report['abstencion_correcta'])}   (negativas)")
    print(f"abstención indebida:         {_fmt(report['abstencion_indebida'])}   (respondibles)")
    for r in report["rows"]:
        print(f"  id={r['id']:<2} {r['tipo']:<12} posicion={r['posicion']}  abstuvo={r['abstuvo']}")
