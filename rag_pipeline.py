"""
Lab 02 — Pipeline RAG baseline.

Ingesta → embeddings → Qdrant → recuperación → prompt → generación. La generación queda
aislada para que la recuperación pueda evaluarse sin gastar tokens.

Cinco decisiones que conviene conocer antes de tocar nada:

1. **Los embeddings son de `bge-m3`, servido en la H200 de la USFQ** (fila
   `embed_local_multilingue` de la tabla semestral: multilingüe, 1024 dimensiones, tope de
   8192 tokens). Hace falta estar en la red de la universidad —GlobalProtect conectada—. Sin
   VPN, la alternativa es la API de OpenAI (`EMBEDDING_BACKEND=openai`, fila
   `embed_api_economico`; `EMBEDDING_MODEL` elige otro de sus modelos). La ruta `local`
   —el MiniLM de los notebooks, tope de 128— existe **solo para la Parte 0.b del taller**.
2. **El tope de tokens se lee, no se supone**, y la ingesta fragmenta en tokens del **mismo**
   tokenizador que vectoriza, así que `chunk_tokens` y el tope se comparan en la misma escala.
   Contra la H200 se pide además `truncate: false`: si un texto no cabe, el servidor da error
   en vez de recortarlo en silencio.
3. **Una sola frase de abstención**, `ABSTENCION`, compartida con el golden set y con el
   notebook del miércoles, y detectada **normalizada** (`se_abstuvo`).
4. **La generación elige su ruta por la clave disponible**: con `OPENAI_API_KEY`, la fila
   `propietario_economico`; con `ANTHROPIC_API_KEY`, `juez_economico`; sin ninguna, Ollama
   con `open_weight_pequeno`; y si tampoco hay Ollama, «modo inspección», que devuelve el
   prompt para que la recuperación se pueda depurar igual.
5. **La API de Qdrant es la del notebook del martes** —`create_collection` +
   `query_points`—, y `QDRANT_URL=":memory:"` corre sin Docker.

Ningún nombre de modelo se escribe aquí a mano: salen de `fuentes/modelos/modelos-2026-1.json`
por su `id`, y las variables de entorno solo los sobreescriben.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import unicodedata

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
import numpy as np

from ingestion import Chunk, load_corpus

load_dotenv()

# La tabla viaja con el repositorio (modelos/, copia de fuentes/modelos/ con las claves
# embeddings y rerankers del Taller 02); si no está, se busca en el repositorio del curso.
REPO = Path(__file__).resolve().parents[3]
TABLA_MODELOS = next((p for p in (Path(__file__).resolve().parent / "modelos" / "modelos-2026-1.json",
                                  REPO / "fuentes" / "modelos" / "modelos-2026-1.json") if p.exists()),
                     REPO / "fuentes" / "modelos" / "modelos-2026-1.json")


def fila_de_la_tabla(clave: str, id_: str) -> dict:
    """Una fila de la tabla semestral por su `id`. Si el lab se copió fuera del repositorio
    y la tabla no está, devuelve {} y las variables de entorno mandan."""
    try:
        datos = json.loads(TABLA_MODELOS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return next((f for f in datos.get(clave, []) if f.get("id") == id_), {})


# ── Abstención: una frase, comparada normalizada ─────────────────────────────────────
ABSTENCION = "El corpus no contiene información suficiente."


def _normalizar_texto(t: str) -> str:
    """minúsculas, sin tildes, sin puntuación: 'informacion suficiente' e
    'información suficiente.' son la misma frase."""
    t = unicodedata.normalize("NFKD", t.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())


def se_abstuvo(respuesta: str) -> bool:
    return _normalizar_texto(ABSTENCION) in _normalizar_texto(respuesta or "")


# ── Configuración ────────────────────────────────────────────────────────────────────
COLLECTION = os.getenv("QDRANT_COLLECTION", "mmia6013_rag")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")   # ":memory:" corre sin Docker
GENERATION_MODEL = os.getenv("GENERATION_MODEL")   # si no, se decide por la clave disponible
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")   # generación local
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "512"))   # tope de tokens generados por Ollama
# Fragmentación del baseline (Taller 02, Parte 1.2): explícita y en tokens del modelo. Sin
# variable, la ingesta usa sus valores por defecto (512 y un quinto de solapamiento).
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS")) if os.getenv("CHUNK_TOKENS") else None
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS")) if os.getenv("OVERLAP_TOKENS") else None

# Embeddings: h200 (por defecto) · openai (sin VPN) · local (solo la Parte 0.b del taller).
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "h200").strip().lower()
H200_EMBED_URL = os.getenv("H200_EMBED_URL", "http://172.28.230.10:11434")   # Ollama de la H200
FILA_EMBEDDINGS = {"h200": "embed_local_multilingue",     # bge-m3
                   "openai": "embed_api_economico",
                   "local": "embed_notebook_s2"}          # MiniLM, tope 128: Parte 0.b
if EMBEDDING_BACKEND not in FILA_EMBEDDINGS:
    raise SystemExit(f"EMBEDDING_BACKEND={EMBEDDING_BACKEND!r}: usa h200, openai o local")
FILA_EMBEDDING = fila_de_la_tabla("embeddings", FILA_EMBEDDINGS[EMBEDDING_BACKEND])
EMBEDDING_MODEL = (os.getenv("EMBEDDING_MODEL") or FILA_EMBEDDING.get("model")
                   or {"h200": "BAAI/bge-m3", "openai": "text-embedding-3-small",
                       "local": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}
                   [EMBEDDING_BACKEND])


def _normalizar(vectores) -> np.ndarray:
    v = np.asarray(vectores, dtype=np.float32)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


class _TokenizadorTiktoken:
    """Adaptador de tiktoken con la misma interfaz que usa la ingesta de un tokenizador de
    Hugging Face: `tok(texto, add_special_tokens=False, truncation=False)["input_ids"]` y
    `tok.decode(ids)`."""

    def __init__(self, modelo: str):
        import tiktoken
        try:
            self._enc = tiktoken.encoding_for_model(modelo)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def __call__(self, texto, add_special_tokens=False, truncation=False):
        return {"input_ids": self._enc.encode(texto)}

    def decode(self, ids):
        return self._enc.decode(list(ids))


class CodificadorH200:
    """`bge-m3` en el Ollama de la H200 (puerto 11434). Solo biblioteca estándar para la red.

    El tokenizador es el de `BAAI/bge-m3` en Hugging Face (se descarga solo el tokenizador,
    no el modelo), para que la ingesta cuente en los mismos tokens que el servidor.
    """

    LOTE = 32

    def __init__(self, modelo: str, tope: int):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelo)
        self.max_seq_tokens = int(tope)
        # El id que sirve Ollama no se escribe: se busca en su catálogo el que empieza igual
        # que el nombre corto del modelo de la tabla («bge-m3» → «bge-m3:latest»).
        corto = modelo.split("/")[-1].lower()
        servidos = [m["name"] for m in self._pedir("/api/tags", timeout=10).get("models", [])]
        candidatos = sorted(n for n in servidos if n.lower().startswith(corto))
        if not candidatos:
            raise SystemExit(f"la H200 no sirve {corto} (sirve: {', '.join(servidos) or 'nada'})")
        self.id_servido = candidatos[0]

    def _pedir(self, ruta: str, cuerpo: dict | None = None, timeout: int = 180) -> dict:
        import urllib.error
        import urllib.request
        datos = None if cuerpo is None else json.dumps(cuerpo).encode()
        pet = urllib.request.Request(f"{H200_EMBED_URL}{ruta}", data=datos,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(pet, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as err:
            raise SystemExit(f"la H200 respondió {err.code} en {ruta}: "
                             f"{err.read().decode(errors='replace')[:300]}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            raise SystemExit(
                f"la H200 no responde en {H200_EMBED_URL} ({err}). ¿Está conectada la VPN "
                "GlobalProtect? Sin VPN, usa la API de OpenAI: EMBEDDING_BACKEND=openai "
                "con OPENAI_API_KEY en tu .env.") from None

    def encode(self, textos: list[str]) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), self.LOTE):
            r = self._pedir("/api/embed", {
                "model": self.id_servido, "input": textos[i:i + self.LOTE],
                # Sin truncado silencioso: un texto que no cabe es un error, no un recorte.
                "truncate": False, "options": {"num_ctx": self.max_seq_tokens}})
            salida.extend(r["embeddings"])
        return _normalizar(salida)


class CodificadorOpenAI:
    """Embeddings por la API de OpenAI: la alternativa sin VPN. Gasta clave, poco."""

    LOTE = 64

    def __init__(self, modelo: str, tope: int):
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("EMBEDDING_BACKEND=openai necesita OPENAI_API_KEY en tu .env")
        from openai import OpenAI
        self._cliente = OpenAI()
        self.modelo = modelo
        self.tokenizer = _TokenizadorTiktoken(modelo)
        self.max_seq_tokens = int(tope)

    def encode(self, textos: list[str]) -> np.ndarray:
        salida = []
        for i in range(0, len(textos), self.LOTE):
            r = self._cliente.embeddings.create(model=self.modelo, input=textos[i:i + self.LOTE])
            salida.extend(d.embedding for d in r.data)
        return _normalizar(salida)


class CodificadorLocal:
    """El MiniLM de los notebooks, en tu máquina. Solo para la Parte 0.b del taller: trunca a
    128 tokens sin avisar, que es justamente lo que esa parte hace ver."""

    def __init__(self, modelo: str, tope: int | None = None):
        from sentence_transformers import SentenceTransformer
        self._modelo = SentenceTransformer(modelo)
        self.tokenizer = self._modelo.tokenizer
        self.max_seq_tokens = int(self._modelo.max_seq_length)   # leído del modelo

    def encode(self, textos: list[str]) -> np.ndarray:
        return self._modelo.encode(textos, normalize_embeddings=True)


def crear_codificador():
    tope = int(FILA_EMBEDDING.get("max_seq_tokens") or 8192)
    clase = {"h200": CodificadorH200, "openai": CodificadorOpenAI,
             "local": CodificadorLocal}[EMBEDDING_BACKEND]
    return clase(EMBEDDING_MODEL, tope)


class RagPipeline:
    def __init__(self, collection: str = COLLECTION):
        self.collection = collection
        self.encoder = crear_codificador()
        self.max_seq_tokens = self.encoder.max_seq_tokens
        self.client = QdrantClient(":memory:") if QDRANT_URL == ":memory:" else QdrantClient(url=QDRANT_URL)
        self.ultima_generacion = "sin generar"

    # ── ingesta ──
    def ingest(self, corpus_dir: Path, chunk_tokens: int | None = CHUNK_TOKENS,
               overlap_tokens: int | None = OVERLAP_TOKENS) -> list[Chunk]:
        """Fragmenta en tokens del modelo cargado, con su tope leído y no supuesto."""
        return load_corpus(Path(corpus_dir), tokenizer=self.encoder.tokenizer,
                           max_seq_tokens=self.max_seq_tokens,
                           chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)

    # ── índice ──
    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.encoder.encode([chunk.text for chunk in chunks])
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        points = [
            models.PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "document": chunk.document,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    # ── recuperación ──
    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        query_vector = self.encoder.encode([question])[0].tolist()
        respuesta = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": hit.score,
                "chunk_id": hit.payload["chunk_id"],
                "document": hit.payload["document"],
                "text": hit.payload["text"],
            }
            for hit in respuesta.points
        ]

    # ── prompt ──
    @staticmethod
    def build_prompt(question: str, contexts: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[{idx}] {ctx['document']} / {ctx['chunk_id']}\n{ctx['text']}"
            for idx, ctx in enumerate(contexts, start=1)
        )
        return f"""Responde usando solo el contexto recuperado.
Si el contexto no contiene la respuesta, di exactamente: "{ABSTENCION}"

Contexto:
{context_text}

Pregunta: {question}
Respuesta:"""

    # ── generación ──
    def generate(self, prompt: str) -> str:
        """Elige la ruta por la clave disponible; sin ninguna, modo inspección."""
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "propietario_economico").get("model")
            self.ultima_generacion = f"openai:{modelo}"
            r = OpenAI().chat.completions.create(
                model=modelo, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return r.choices[0].message.content or ""
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "juez_economico").get("model")
            self.ultima_generacion = f"anthropic:{modelo}"
            r = anthropic.Anthropic().messages.create(
                model=modelo, max_tokens=400, temperature=0,
                messages=[{"role": "user", "content": prompt}])
            return "".join(b.text for b in r.content if b.type == "text")
        try:
            import urllib.request
            modelo = GENERATION_MODEL or fila_de_la_tabla("models", "open_weight_pequeno").get("model")
            # Taller 02: Ollama 0.34 activa el razonamiento de qwen3 por defecto, aunque la
            # fila open_weight_pequeno declara think=false. Con él activo, algunos prompts
            # razonaban hasta pasar los 120 s de timeout —la excepción caía en «modo
            # inspección» y la fila quedaba sin medir— y, con tope de tokens, el razonamiento
            # se comía el tope y `response` llegaba VACÍA: se_abstuvo("") es False, así que
            # contaba como respuesta. think=false es el valor de la fila; num_predict, un tope.
            cuerpo = json.dumps({"model": modelo, "prompt": prompt, "stream": False,
                                 "think": False,
                                 "options": {"temperature": 0, "num_predict": NUM_PREDICT}}).encode()
            req = urllib.request.Request(f"{OLLAMA_URL}/api/generate", data=cuerpo,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.ultima_generacion = f"ollama:{modelo}"
                return json.loads(r.read())["response"]
        except Exception:  # noqa: BLE001 — sin Ollama tampoco: modo inspección
            self.ultima_generacion = "modo inspección (sin clave ni Ollama)"
            return "[modo inspección: no hay clave de API ni Ollama; este es el prompt]\n" + prompt

    def answer(self, question: str, top_k: int = 5) -> dict:
        contexts = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, contexts)
        texto = self.generate(prompt)
        # En modo inspección la «respuesta» es el prompt, que contiene la frase de abstención
        # en su instrucción: contarla sería una abstención falsa. Queda sin medir (None).
        inspeccion = self.ultima_generacion.startswith("modo inspección")
        return {"question": question, "contexts": contexts, "prompt": prompt,
                "answer": texto, "abstained": None if inspeccion else se_abstuvo(texto),
                "generator": self.ultima_generacion}


if __name__ == "__main__":
    import sys
    corpus = Path(os.getenv("CORPUS_DIR", "corpus"))
    pipeline = RagPipeline()
    print(f"embeddings: {EMBEDDING_BACKEND} · {EMBEDDING_MODEL} · tope {pipeline.max_seq_tokens} "
          f"tokens · Qdrant {QDRANT_URL}")
    chunks = pipeline.ingest(corpus)
    if not chunks:
        sys.exit(f"ningún fragmento indexable en {corpus}/ (¿PDFs escaneados? ¿carpeta vacía?)")
    pipeline.index(chunks)
    print(f"indexados {len(chunks)} fragmentos de {len({c.document for c in chunks})} documento(s)")
    question = " ".join(sys.argv[1:]) or "REEMPLAZAR por una pregunta de prueba"
    for hit in pipeline.retrieve(question):
        print(f"  {hit['score']:.3f}  {hit['document']} / {hit['chunk_id']}  {hit['text'][:80]}…")
    salida = pipeline.answer(question)
    print(f"\n[{salida['generator']}] abstuvo={salida['abstained']}\n{salida['answer'][:1200]}")
