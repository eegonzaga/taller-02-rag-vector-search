"""
Taller 02 — Parte 2.c: evidencia de los tres peores casos del baseline (k=5, denso).

    python parte2c_evidencia.py > salidas/parte2c_evidencia.txt

Para cada caso: la salida de la ingesta del documento fuente, los top-5 con puntaje (y dónde
cae el fragmento correcto en el top-20), la respuesta generada. Al final, las tres fallas
silenciosas de la Parte 0 buscadas en el corpus real.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation import acierta, es_respondible
from ingestion import MIN_CARACTERES_UTILES, read_document
from rag_pipeline import RagPipeline

PEORES = [10, 1, 6]
K = 5


def main() -> None:
    golden = {i["id"]: i for i in json.loads(Path("golden_set.json").read_text(encoding="utf-8"))}
    chunks = [json.loads(l) for l in Path("data/chunks.jsonl").open(encoding="utf-8")]
    p = RagPipeline()

    for qid in PEORES:
        it = golden[qid]
        print("=" * 100)
        print(f"id={qid} · {it['tipo']} · {it['pregunta']}")
        print(f"esperado: {it['respuesta_esperada']}")
        for doc in it["documentos_fuente"]:
            d = read_document(Path("corpus") / doc)
            n = sum(1 for c in chunks if c["document"] == doc)
            print(f"ingesta: {doc}: {d.pages} página(s), {d.useful_chars} caracteres útiles → {n} fragmento(s)")
            if it["fragmento_esperado"]:
                con = [c["id"] for c in chunks if c["document"] == doc
                       and acierta({"document": doc, "text": c["text"]}, it)]
                print(f"  fragmentos que contienen «{it['fragmento_esperado']}»: {con}")
        top20 = p.retrieve(it["pregunta"], top_k=20)
        print(f"top-{K} (denso, bge-m3):")
        for pos, h in enumerate(top20[:K], start=1):
            marca = "ACIERTO" if acierta(h, it) else ""
            print(f"  {pos}. {h['score']:.4f} {h['chunk_id']:<48} {marca} {' '.join(h['text'].split())[:80]}…")
        aciertos20 = [pos for pos, h in enumerate(top20, start=1) if acierta(h, it)]
        print(f"posiciones con acierto en el top-20: {aciertos20}")
        salida = p.answer(it["pregunta"], top_k=K)
        print(f"respuesta [{salida['generator']}] abstuvo={salida['abstained']}:")
        print("  " + salida["answer"].strip().replace("\n", "\n  "))

    print("\n" + "=" * 100)
    print("Las tres fallas silenciosas de la Parte 0, buscadas en el corpus real")
    print("-- 0.a escaneados: caracteres útiles por página de cada documento (umbral del lab: "
          f"{MIN_CARACTERES_UTILES} por documento)")
    for path in sorted(Path("corpus").glob("*.pdf")):
        d = read_document(path)
        print(f"  {path.name:<48} {d.pages:3d} pág · {d.useful_chars // d.pages:5d} car/pág")
    print("-- 0.b truncado: tope de bge-m3 = 8192; fragmento más largo en tokens de bge-m3:")
    tok = p.encoder.tokenizer
    largos = max(len(tok(c["text"], add_special_tokens=False)["input_ids"]) for c in chunks)
    print(f"  {largos} tokens (chunk_tokens=512: re-tokenizar el texto decodificado puede variar)")
    print("-- 0.c el índice no se queja: score_top1 de negativas vs respondibles (resultados.csv, denso)")
    filas = [r for r in csv.DictReader(Path("resultados.csv").open(encoding="utf-8"))
             if r["recuperacion"] == "denso" and r["k"] == str(K)]
    neg = sorted(float(r["score_top1"]) for r in filas if r["respondible"] == "False")
    resp = sorted(float(r["score_top1"]) for r in filas if r["respondible"] == "True")
    print(f"  negativas:    {neg}")
    print(f"  respondibles: {resp}")
    print(f"  la negativa más alta ({max(neg):.4f}) supera a {sum(s < max(neg) for s in resp)} "
          f"de {len(resp)} respondibles: ningún umbral de puntaje las separa.")


if __name__ == "__main__":
    main()
