"""
Taller 02 — Parte 0: las tres fallas que no fallan, reproducidas sobre `ejemplos/`.

    EMBEDDING_BACKEND=local QDRANT_URL=":memory:" python parte0.py

- 0.a  Qué produciría la ingesta con `instructivo_escaneado.pdf` SIN la comprobación de
       `load_corpus`: se fragmenta el texto crudo tal como lo devuelve `read_document`.
- 0.b  Ingesta de `ejemplos/` pidiendo chunk_tokens=900 (aviso), y coseno entre el vector de
       un texto de ~900 palabras y el del mismo texto recortado a max_seq_length tokens.
- 0.c  Se corre con `rag_pipeline.py` (ver salidas/parte0*.txt).
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("QDRANT_URL", ":memory:")

import numpy as np
from pypdf import PdfReader

from ingestion import (caracteres_utiles, fixed_size_chunks_tokens, load_corpus,
                       normalize_text, read_document)
from rag_pipeline import crear_codificador

EJEMPLOS = Path("ejemplos")
TEXTO_900 = Path(__file__).resolve().parent / "corpus" / "lewis-2020-rag.pdf"


def parte_0a(tokenizer) -> None:
    print("=" * 30, "0.a — sin la comprobación de load_corpus", "=" * 30)
    doc = read_document(EJEMPLOS / "instructivo_escaneado.pdf")
    texto = normalize_text(doc.text)
    print(f"read_document: {doc.pages} página(s), useful_chars={doc.useful_chars}, "
          f"parece_escaneado={doc.parece_escaneado}")
    print(f"texto extraído (repr): {texto!r}")
    piezas = fixed_size_chunks_tokens(texto, tokenizer, 126, 25)
    print(f"fragmentos que se habrían indexado: {len(piezas)} → {piezas!r}")
    print(f"caracteres útiles de ese fragmento: {[caracteres_utiles(p) for p in piezas]} "
          f"(ninguna excepción lanzada)")


def parte_0b(cod) -> None:
    print("\n" + "=" * 30, "0.b — el fragmento que se corta a 128", "=" * 30)
    chunks = load_corpus(EJEMPLOS, tokenizer=cod.tokenizer, max_seq_tokens=cod.max_seq_tokens,
                         chunk_tokens=900, verbose=True)
    print(f"fragmentos: {len(chunks)} (chunk_tokens pedido = 900)")

    # ~900 palabras de texto real: Lewis et al. (2020), páginas 1-3.
    reader = PdfReader(str(TEXTO_900))
    palabras = " ".join((p.extract_text() or "") for p in reader.pages[:3]).split()
    texto = " ".join(palabras[:900])
    tok = cod.tokenizer
    ids = tok(texto, add_special_tokens=False, truncation=False)["input_ids"]
    tope = cod.max_seq_tokens
    recortado = tok.decode(ids[: tope - 2])   # el tope incluye [CLS] y [SEP]
    v_entero, v_recortado = cod.encode([texto, recortado])
    coseno = float(np.dot(v_entero, v_recortado))
    print(f"texto: {len(palabras[:900])} palabras de {TEXTO_900.name}")
    print(f"tokens del texto (tokenizador del modelo): {len(ids)}")
    print(f"max_seq_length del modelo: {tope}  → llegan al vector {tope - 2} tokens de contenido "
          f"({(tope - 2) / len(ids):.1%} del texto)")
    print(f"coseno(vector del texto entero, vector del texto recortado a {tope - 2} tokens) = {coseno:.6f}")


if __name__ == "__main__":
    codificador = crear_codificador()
    parte_0a(codificador.tokenizer)
    parte_0b(codificador)
