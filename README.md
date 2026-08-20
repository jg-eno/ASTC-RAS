<div align="center">

# Activity-Aware VideoRAG Pipeline

**Retrieval-Augmented Generation for Context-Aware Video Summarization**

</div>

This repository implements a powerful **Activity-Aware Graph Summarization** pipeline. It extends basic Video Retrieval-Augmented Generation (RAG) by representing video-derived insights via an interconnected Knowledge Graph. Rather than executing flat, context-free queries across an entire vector database, the system dynamically clusters entities based on generated activities to build highly accurate, context-bound localized summaries.

## Core Conceptual Flow

The pipeline operates in two major phases:

### Phase 1: Video Extraction & Graph Construction (`construct_graph.py`)
1. **Video Ingestion & Chunking**: Audio is transcribed using the `faster-distil-whisper-large-v3` model, and visual keyframes are captioned using the `MiniCPM-V-2_6-int4` Vision Language Model. The outputs are binned into text chunks mapped tightly against the video timeline.
2. **Extraction Engine (LLM)**: An LLM dynamically scans these chunks to identify distinct **Entities**, their inter-node **Relationships**, and their temporal **Activities**.
3. **Graph Building**: The extracted entities and relationships are stitched into a localized Graph Database. Critically, each entity node's metadata is permanently tagged with the precise dynamic activities found in its respective chunks (e.g., `["running", "speaking"]`).

### Phase 2: Activity-Aware Traversals & Summarization (`ask_activity.py`)
1. **Clustering & Pruning**: The system automatically pulls from the graph and groups entity nodes into localized subsets matching specific activities. Textually, it filters chunk references to structurally process data natively bound to a `target_video_name`.
2. **Subgraph Traversal**: For each local activity subset, the system selects the most structurally integral nodes using Degree Centrality. It then executes a Breadth-First-Search (BFS) expanding 2 hops outward to retrieve surrounding contextual peripheral nodes.
3. **Hierarchical Summarization**:
   - **Local Level**: It maps the filtered nodes exactly back to the source data transcripts and creates a localized summary describing that single activity event.
   - **Global Level**: It integrates and aggregates every structured localized mini-summary into one final, polished, macro summary describing the combined sequence of events without redundancy.

## Repository Layout

```text
ASTC-RAS/
├── README.md                         # This document
├── setup.sh                          # pip dependency install script (torch, faster-whisper, etc.)
├── .gitignore                        # Ignores weights, .env, and Python bytecode
├── LICENSE                           # Project license
│
├── construct_graph.py                # Phase 1: ingest videos -> transcript/captions -> build Knowledge Graph
├── ask_activity.py                   # Phase 2: Activity-Aware subgraph traversal + hierarchical summary
├── ask_graph.py                      # Auxiliary / testing direct graph query engine
│
├── videos/                           # Drop target .mp4 clips here (indexed by construct_graph.py)
│   ├── Amritsar Woman Stops Robbers.mp4
│   └── Human Activity Recognition I3D Demo 720P.mp4
│
└── VideoRAG_algorithm/               # Core algorithm library (forked VideoRAG, trimmed to essentials)
    └── videorag/
        ├── __init__.py               # Exposes VideoRAG, QueryParam
        ├── videorag.py               # Main pipeline: insert_video, query, generate_activity_summary
        ├── activity_summarization.py # Activity clustering + BFS subgraph traversal + summarization
        ├── _op.py                    # Graph/entity extraction + retrieval operator
        ├── _splitter.py              # Text chunk splitting
        ├── _llm.py                   # LLM + embedding configs (Gemini, OpenAI, Azure, Ollama)
        ├── base.py                   # Storage base interfaces
        ├── prompt.py                 # LLM prompts
        ├── _utils.py                 # Shared helpers / hashing
        ├── _videoutil/               # Video ingestion utilities
        │   ├── split.py              # Video -> clip splitting
        │   ├── feature.py            # Visual feature extraction
        │   ├── caption.py            # VLM captioning (MiniCPM-V)
        │   └── asr.py                # Audio transcription (faster-whisper)
        └── _storage/                 # Pluggable storage backends
            ├── kv_json.py            # JSON key-value store
            ├── gdb_networkx.py       # In-memory NetworkX graph
            ├── gdb_neo4j.py          # Optional Neo4j graph
            ├── vdb_hnswlib.py        # HNSW vector store
            └── vdb_nanovectordb.py   # NanoVectorDB vector store (+ video segment store)
```

> Runtime note: the caption (MiniCPM-V-2_6-int4) and ASR (faster-distil-whisper-large-v3) model weights are downloaded locally and kept out of git via `.gitignore` (see the "Large local model weights" section). The `videorag-workdir*` directories generated at runtime are also gitignored.

## Quick Start Guide

1. **Environment Initialization:** Ensure all necessary weights are downloaded to their folders natively and initialize your local conda virtual environment (`conda activate videorag`).
2. **Setup API Credentials:** Place a `.env` file at the root containing a valid `GEMINI_API_KEY` to grant the algorithm proper extraction capabilities.
3. **Index your Videos:** Place your desired `.mp4` clip files directly into the `videos/` directory. Target them uniformly from inside the array inside `construct_graph.py` and run `python construct_graph.py` to ingest the stream and build the network.
4. **Acquire Summaries:** Run `python ask_activity.py` to trigger the localized multi-video filtering querying (`generate_activity_summary(target_video_name="...")`). 

This architecture allows numerous clips to be indexed into a single working directory while safely shielding extraction traversals to exact individual clip queries automatically!
