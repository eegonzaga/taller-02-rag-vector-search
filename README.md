# Taller 02 — RAG sobre los papers del curso

MMIA 6013 IA Generativa y Agentes · USFQ · Estefanía Gonzaga

Un RAG baseline reproducible sobre un corpus real, medido con un golden set, con abstención y
una extensión (reranking) elegida por el modo de falla medido. Construido sobre el andamiaje
del Lab-02.

## Resultados

Corpus: **Opción B**, 16 papers de las semanas 1 y 2 (440 páginas, inglés) → 1010 fragmentos.

| Recuperación | k | Hit Rate@k | MRR | Abst. correcta | Abst. indebida |
|---|---|---|---|---|---|
| Denso (baseline) | 3 | 0,625 | 0,562 | 1,000 | 0,875 |
| Denso (baseline) | 5 | 0,750 | 0,594 | 1,000 | 0,625 |
| + reranking | 3 | 0,875 | 0,729 | 1,000 | 0,625 |
| + reranking | 5 | 0,875 | 0,729 | 1,000 | 0,375 |

Hit Rate y MRR se calculan sobre las 8 preguntas respondibles; las 2 negativas solo cuentan
para la abstención. La tabla sale de `resultados.csv` con `python tabla_metricas.py`.

## Configuración

| Pieza | Fila de la tabla de modelos | Modelo | verified_at |
|---|---|---|---|
| Embeddings | `embed_local_multilingue` | BAAI/bge-m3 (1024 dim, tope 8192), en la H200 | 2026-08-27 |
| Parte 0 | `embed_notebook_s2` | paraphrase-multilingual-MiniLM-L12-v2 (tope 128) | 2026-08-27 |
| Generación | `open_weight_pequeno` | qwen3:1.7b por Ollama (temp 0, `think=false`) | 2026-09-18 |
| Reranking | `rerank_local_ingles` | cross-encoder/ms-marco-MiniLM-L6-v2 | 2026-09-19 |

- **Fragmentación:** `chunk_tokens=512`, `overlap_tokens=128`, en tokens de bge-m3.
- **Índice:** Qdrant (contenedor), coseno, colección `mmia6013_rag`.
- **Costo:** 0 USD. Todo corre en local o en la H200; no se usa ninguna API de pago.

## Estructura

```
├── corpus/                  16 PDFs del corpus (copia de fuentes/papers/s1 y s2)
├── ejemplos/                Corpus mínimo + PDF escaneado, para la Parte 0
├── modelos/                 Tabla semestral de modelos (con embeddings y rerankers)
├── ingestion.py             Carga, parseo y fragmentación en tokens del modelo
├── rag_pipeline.py          Ingesta → índice → recuperación → prompt → generación
├── evaluation.py            Hit Rate, MRR, dos tasas de abstención → resultados.csv
├── golden_set.json          10 preguntas: 4 simples, 3 multi-fragmento, 2 negativas, 1 adversarial
├── parte0.py                Parte 0.a y 0.b
├── parte1.py                Parte 1: ingesta, índice y top-5 de 3 preguntas de prueba
├── parte2c_evidencia.py     Parte 2.c: evidencia de los tres peores casos
├── parte3_rerank.py         Parte 3.C: reranking top-20 → top-k
├── tabla_metricas.py        Tablas del informe a partir de resultados.csv
├── resultados.csv           Crudo: una fila por consulta, k y tipo de recuperación
└── salidas/                 Salidas crudas de cada parte (y de las corridas descartadas)
```

## Requisitos

- Python 3.12 y [uv](https://docs.astral.sh/uv/) (o `pip`).
- Docker, para Qdrant.
- [Ollama](https://ollama.com) con `qwen3:1.7b`.
- VPN GlobalProtect de la USFQ conectada, para bge-m3 en la H200. La Parte 0 no la necesita.

## Cómo reproducir

### 1. Entorno

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt       # versiones exactas

cp .env.example .env                     # deja las claves VACÍAS (ver nota abajo)
docker run -d -p 6333:6333 qdrant/qdrant
ollama serve &                           # si Ollama no está corriendo
ollama pull qwen3:1.7b
```

> **Deja `OPENAI_API_KEY` y `ANTHROPIC_API_KEY` vacías.** `generate` elige el generador según
> la clave disponible; con una clave puesta respondería gpt-4o-mini o Claude en vez de
> qwen3:1.7b y las tasas de abstención no coincidirían con las del informe.

### 2. Parte 0 — sin VPN y sin Docker

```bash
# 0.a y 0.c: ingesta de ejemplos/ y las dos consultas
export EMBEDDING_BACKEND=local QDRANT_URL=":memory:" CORPUS_DIR=ejemplos CHUNK_TOKENS= OVERLAP_TOKENS=
python rag_pipeline.py "¿cuál es la política de mascotas?"
python rag_pipeline.py "¿qué exige el instructivo de prácticas de campo sobre el seguro de accidentes?"

# 0.a (qué pasaría sin la comprobación) y 0.b (chunk_tokens=900 y coseno truncado)
python parte0.py
unset EMBEDDING_BACKEND QDRANT_URL CORPUS_DIR CHUNK_TOKENS OVERLAP_TOKENS
```

`CHUNK_TOKENS=` vacío hace que la Parte 0 use el valor por defecto del lab (el tope del MiniLM)
en lugar de los 512 del `.env`.

### 3. Parte 1 — baseline (VPN conectada)

```bash
python parte1.py        # ingesta, índice en Qdrant y top-5 de 3 preguntas (~1 min)
```

Escribe `data/chunks.jsonl` con los fragmentos tal como se indexaron.

### 4. Parte 2 — evaluación

```bash
python evaluation.py --k 3 --sin-indexar
python evaluation.py --k 5 --sin-indexar --anexar
python parte2c_evidencia.py > salidas/parte2c_evidencia.txt
```

`--sin-indexar` reutiliza la colección de la Parte 1; `--anexar` agrega las filas al mismo
`resultados.csv`.

### 5. Parte 3 — reranking

```bash
python parte3_rerank.py --k 3 --anexar
python parte3_rerank.py --k 5 --anexar
python tabla_metricas.py
```

Con esto `resultados.csv` queda con 40 filas: 10 preguntas × 2 valores de k × 2 recuperaciones.

## Cambios al andamiaje original

- **`modelos/modelos-2026-1.json`:** la tabla viaja con el repositorio y añade las claves
  `embeddings` y `rerankers` del anexo del enunciado. `rag_pipeline.py` la busca aquí primero
  y, si no está, en `fuentes/modelos/` del repositorio del curso.
- **Generación por Ollama:** `think=false` y `num_predict=512`. Ollama 0.34 activa el
  razonamiento de qwen3 por defecto; con él, una pregunta tardó más de 10 minutos (quedaba sin
  medir) y, con solo el tope, llegaban respuestas vacías que contaban como «no se abstuvo».
- **`evaluation.py`:** opción `--anexar` y columnas `recuperacion` y `respuesta` en el CSV.
- **Fragmentación:** `CHUNK_TOKENS` y `OVERLAP_TOKENS` se leen del `.env`.
- **`requirements.txt`:** versiones exactas; se añadió `transformers`, que el código usaba y no
  estaba listado.

## Notas

- Las tasas de abstención dependen del generador y de su configuración; Hit Rate y MRR solo del
  índice. Las corridas con otras configuraciones de Ollama están en `salidas/` para comparar.
- Ninguna clave va al repositorio: `.env` está en `.gitignore`.
