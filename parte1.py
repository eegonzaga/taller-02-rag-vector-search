"""
Taller 02 — Parte 1: baseline sobre el corpus (Opción B, papers de las semanas 1 y 2).

    python parte1.py                 # ingesta + índice en Qdrant (contenedor) + 3 preguntas
    python parte1.py --sin-indexar   # reutiliza la colección ya indexada

Escribe `data/chunks.jsonl` (los fragmentos tal como se indexaron, para inspeccionar qué se
degradó en la ingesta) y `salidas/parte1_top5.txt`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ingestion import write_chunks_jsonl
from rag_pipeline import (CHUNK_TOKENS, EMBEDDING_BACKEND, EMBEDDING_MODEL, FILA_EMBEDDING,
                          FILA_EMBEDDINGS, OVERLAP_TOKENS, QDRANT_URL, RagPipeline)

PREGUNTAS_DE_PRUEBA = [
    "How many parameters does the largest GPT-3 model have?",
    "What two kinds of non-parametric memory does RAG combine with a parametric seq2seq model, and what retriever does it use?",
    "Why does DPO not need to train an explicit reward model?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("corpus"))
    ap.add_argument("--sin-indexar", action="store_true")
    ap.add_argument("--sin-generar", action="store_true")
    args = ap.parse_args()

    pipeline = RagPipeline()
    print(f"embeddings: fila {FILA_EMBEDDINGS[EMBEDDING_BACKEND]} · {EMBEDDING_MODEL} · "
          f"verified_at {FILA_EMBEDDING.get('verified_at')} · tope {pipeline.max_seq_tokens} tokens")
    print(f"Qdrant {QDRANT_URL} · colección {pipeline.collection} · "
          f"chunk_tokens={CHUNK_TOKENS} overlap_tokens={OVERLAP_TOKENS}")
    if not args.sin_indexar:
        chunks = pipeline.ingest(args.corpus)
        write_chunks_jsonl(chunks, Path("data/chunks.jsonl"))
        t0 = time.time()
        pipeline.index(chunks)
        print(f"indexados {len(chunks)} fragmentos de {len({c.document for c in chunks})} "
              f"documento(s) en {time.time() - t0:.1f} s")

    for n, pregunta in enumerate(PREGUNTAS_DE_PRUEBA, start=1):
        print(f"\n### Pregunta {n}: {pregunta}")
        for pos, hit in enumerate(pipeline.retrieve(pregunta, top_k=5), start=1):
            texto = " ".join(hit["text"].split())
            print(f"  {pos}. {hit['score']:.4f}  {hit['document']} / {hit['chunk_id']}  {texto[:110]}…")
        if not args.sin_generar:
            salida = pipeline.answer(pregunta, top_k=5)
            print(f"  → [{salida['generator']}] abstuvo={salida['abstained']}")
            print("    " + salida["answer"].strip().replace("\n", "\n    "))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
