"""
Lab 02 — Ingesta y fragmentación baseline para RAG.

Este archivo evita que el lab empiece desde boilerplate. Los estudiantes deben ajustar
parseo, fragmentación y metadatos al corpus elegido.

Dos comprobaciones que existen porque sus fallas **no fallan**:

1. **Un PDF sin capa de texto** (escaneado, fotografiado) devuelve `""` en cada página; sin
   comprobación, el documento entero se convertiría en UN fragmento hecho solo de
   marcadores `[page=n]`, sin ninguna excepción, y el documento sencillamente no estaría.
   `read_document` cuenta los caracteres útiles y `load_corpus` **avisa y no indexa** lo
   que no tiene texto (Parte 0.a del taller).
2. **Los fragmentos se miden en tokens del modelo**, no en palabras, porque el modelo de
   embeddings tiene un tope en **tokens**. `rag_pipeline.py` le pasa a `load_corpus` el
   tokenizador y el tope del codificador en uso —8192 para `bge-m3`, 128 para el MiniLM de
   la Parte 0.b— y la ingesta avisa si se pide un fragmento que no cabe.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import re
import sys

from pypdf import PdfReader

MARCADOR_PAGINA = re.compile(r"\[page=\d+\]")
MIN_CARACTERES_UTILES = 200   # por documento; por debajo, casi seguro es un escaneo
MIN_CARACTERES_FRAGMENTO = 20  # un fragmento con menos que esto son marcadores o ruido
CHUNK_TOKENS_POR_DEFECTO = 512  # notas del curso, §4.2; acotado por el tope del modelo


@dataclass
class Chunk:
    id: str
    text: str
    document: str
    chunk_index: int
    metadata: dict


@dataclass
class Documento:
    name: str
    path: str
    text: str
    pages: int
    useful_chars: int

    @property
    def parece_escaneado(self) -> bool:
        return self.useful_chars < MIN_CARACTERES_UTILES


def caracteres_utiles(text: str) -> int:
    """Letras y dígitos, sin contar los marcadores de página que nosotros mismos añadimos."""
    return sum(ch.isalnum() for ch in MARCADOR_PAGINA.sub("", text))


def read_pdf(path: Path) -> tuple[str, int]:
    reader = PdfReader(str(path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[page={page_number}]\n{text}")
    return "\n".join(pages), len(reader.pages)


def read_document(path: Path) -> Documento:
    if path.suffix.lower() == ".pdf":
        text, pages = read_pdf(path)
    elif path.suffix.lower() in {".txt", ".md"}:
        text, pages = path.read_text(encoding="utf-8"), 1
    else:
        raise ValueError(f"Formato no soportado: {path}")
    return Documento(name=path.name, path=str(path), text=text, pages=pages,
                     useful_chars=caracteres_utiles(text))


def normalize_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fixed_size_chunks_tokens(text: str, tokenizer, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Fragmentos de `chunk_tokens` tokens del modelo, con solapamiento, decodificados a texto.

    La unidad es el token del **mismo** tokenizador que va a vectorizar: así `chunk_tokens`
    y `max_seq_length` se comparan en la misma escala y el truncado deja de ser invisible.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens debe ser menor que chunk_tokens")
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunks.append(tokenizer.decode(ids[start:end]).strip())
        if end == len(ids):
            break
        start = end - overlap_tokens
    return chunks


def fixed_size_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """La versión en PALABRAS. Se conserva para que se pueda medir la diferencia con la
    versión en tokens; no es el baseline."""
    if overlap >= chunk_size:
        raise ValueError("overlap debe ser menor que chunk_size")
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def load_corpus(corpus_dir: Path, tokenizer=None, max_seq_tokens: int | None = None,
                chunk_tokens: int | None = None, overlap_tokens: int | None = None,
                verbose: bool = True) -> list[Chunk]:
    """Lee los documentos del corpus y los fragmenta en tokens del modelo.

    - `tokenizer` y `max_seq_tokens` los pasa `rag_pipeline.py` desde el modelo cargado.
      Sin tokenizador cae a la versión en palabras y lo dice, porque entonces nadie sabe
      cuánto se trunca.
    - `chunk_tokens` por defecto es `min(512, max_seq_tokens - 2)`: 512 tokens, el tamaño de
      las notas del curso, salvo que el modelo no los admita —el MiniLM de la Parte 0.b ve
      126, su tope menos los dos tokens especiales—. El solapamiento por defecto es un quinto
      del fragmento. Pedir más que el tope produce un AVISO, no una excepción, porque puede
      ser una decisión deliberada que hay que justificar en el informe.
    - Un documento con menos de MIN_CARACTERES_UTILES caracteres útiles **no se indexa** y
      se avisa: es el PDF escaneado de la Parte 0.a del Taller 2.
    """
    if chunk_tokens is None:
        chunk_tokens = max(32, min(CHUNK_TOKENS_POR_DEFECTO, (max_seq_tokens or 128) - 2))
    if overlap_tokens is None:
        overlap_tokens = chunk_tokens // 5
    if max_seq_tokens and chunk_tokens > max_seq_tokens - 2 and verbose:
        print(f"AVISO ingesta: chunk_tokens={chunk_tokens} supera el tope del modelo "
              f"({max_seq_tokens} tokens): lo que pase de {max_seq_tokens - 2} tokens no "
              f"llega al índice —el MiniLM lo trunca en silencio; la H200 rechaza la "
              f"petición—.", file=sys.stderr)

    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        doc = read_document(path)
        if doc.parece_escaneado:
            if verbose:
                print(f"AVISO ingesta: {doc.name}: {doc.useful_chars} caracteres útiles en "
                      f"{doc.pages} página(s). ¿Escaneado o fotografiado? NO se indexa: sin "
                      f"texto no hay nada que recuperar (OCR aparte).", file=sys.stderr)
            continue
        text = normalize_text(doc.text)
        if tokenizer is not None:
            piezas = fixed_size_chunks_tokens(text, tokenizer, chunk_tokens, overlap_tokens)
            estrategia = {"strategy": "fixed_size", "unit": "tokens",
                          "chunk_tokens": chunk_tokens, "overlap_tokens": overlap_tokens,
                          "max_seq_tokens": max_seq_tokens}
        else:
            if verbose:
                print("AVISO ingesta: sin tokenizador, fragmentando en PALABRAS: la unidad no es "
                      "la del modelo y el truncado no se puede medir.", file=sys.stderr)
            piezas = fixed_size_chunks(text)
            estrategia = {"strategy": "fixed_size", "unit": "words",
                          "chunk_size": 900, "overlap": 150}
        idx = 0
        for chunk_text in piezas:
            if caracteres_utiles(chunk_text) < MIN_CARACTERES_FRAGMENTO:
                continue   # solo marcadores de página o espacio: no se indexa
            chunks.append(Chunk(id=f"{path.stem}-{idx:04d}", text=chunk_text,
                                document=path.name, chunk_index=idx,
                                metadata={"source_path": str(path), **estrategia}))
            idx += 1
        if verbose:
            print(f"ingesta: {doc.name}: {doc.pages} página(s), {doc.useful_chars} caracteres "
                  f"útiles → {idx} fragmento(s)")
    return chunks


def write_chunks_jsonl(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")
    tokenizer, tope = None, None
    try:
        from rag_pipeline import crear_codificador
        codificador = crear_codificador()
        tokenizer, tope = codificador.tokenizer, codificador.max_seq_tokens
    except (Exception, SystemExit) as exc:  # noqa: BLE001
        print(f"(sin modelo de embeddings: {exc}; se fragmenta en palabras)", file=sys.stderr)
    chunks = load_corpus(corpus, tokenizer=tokenizer, max_seq_tokens=tope)
    write_chunks_jsonl(chunks, Path("data/chunks.jsonl"))
    print(f"Chunks generados: {len(chunks)}")
