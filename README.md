# MCP Latin Tools Server

A Model Context Protocol (MCP) server for Latin Natural Language Processing (NLP), reported speech detection, and LiLa Knowledge Base querying.

## Features

- Latin tokenization with enclitic `-que` splitting
- UDPipe-based lemmatization, POS tagging, and morphological analysis
- Non-finite verb identification
- Reported speech detection using a fine-tuned LaBERTa transformer latin model
- LiLa Knowledge Base SPARQL querying and exporting results
- MCP-compatible tool interface for Claude Desktop, VS Code, and MCP Inspector

---

# System Overview

The server provides a pipeline of interoperable MCP tools for Latin NLP and Digital Humanities workflows.

The tools are designed to be used sequentially, but may also be used independently.

---


## Available Tools

| Tool | NLP Task | Description |
|---|---|---|
| `tokenize_latin_text` | Tokenization | Latin sentence splitting and enclitic `-que` handling |
| `parser` | Morphology & Syntax | UDPipe-based tokenization, lemmatization, POS tagging, morphological analysis, and dependency parsing (CoNLL-U output) |
| `prepare_latin_input` | Preprocessing | Convert CoNLL-U parser output into model-ready input for downstream tasks |
| `detect_reported_speech_from_text` | Sequence Labeling | Transformer-based reported speech detection for Latin texts (fine-tuned Latin LaBerta) |
| `get_lila_lemma_info` | Knowledge Base Query | Retrieve lexical and morphological information from the LiLa Knowledge Base |
| `get_lila_lemma_tokens_dataframe` | Corpus Retrieval | Retrieve corpus attestations and occurrence counts for a lemma |
| `export_lila_lemma_tokens_csv` | Export | Export LiLa corpus attestations as CSV |


# 1. Latin NLP Parsing Pipeline

The `parser` tool performs:

- tokenization
- sentence segmentation
- lemmatization
- POS tagging
- morphological analysis
- non-finite verb identification

The tool calls the UDPipe API and uses the model:

latin-evalatin24-240520

The model was evaluated on EvaLatin campaign in 2024 and trained with on Latin Dependency Treebanks.

## References

- EvaLatin 2024 overview:
  https://aclanthology.org/2024.lt4hala-1.21/

- UDPipe model repository:
  https://github.com/ufal/evalatin2024-latinpipe

---

# 2. Reported Speech Detection

## A. Preparation Tool

The `parser` tool also prepares the linguistic input required by the reported speech detection model.

This preparation stage:

- aligns UDPipe tokenization with the original tokens
- prepares aligned linguistic features
- formats the input for transformer inference

---

## B. Reported Speech Detector

The `detect_reported_speech` tool performs token-level reported speech prediction.

It takes the output of the parsing/preparation stage as input and returns:

- token-level predictions
- confidence scores

The model is:

- the first experimental Latin reported speech detection model at token level
- a fine-tuned LaBERTa model for token classification

## References

- Hugging Face model repository:
  https://huggingface.co/agudei/latin-reported-speech-laberta

- Paper describing the experiment:
  https://aclanthology.org/2026.latechclfl-1.24/

---

# 3. LiLa Knowledge Base Querying

The `get_lila_lemma_info` tool provides simplified access to the LiLa Knowledge Base.

The tool:

- accepts a Latin lemma
- performs SPARQL queries automatically
- retrieves lexical and linguistic information
- simplifies access to Linked Open Data resources

The tool is designed to help users interact with LiLa without manually writing SPARQL queries.

## LiLa

- LiLa Knowledge Base:
  https://lila-erc.eu/sparql/



# 4. LiLa Corpus Attestation Retrieval

The `get_lila_lemma_tokens_dataframe` tool retrieves corpus attestations linked to a Latin lemma in the LiLa Knowledge Base.

The tool:

- retrieves token occurrences associated with a lemma
- retrieves token URIs
- retrieves work titles
- computes occurrence frequencies per work
- structures results as a dataframe-like output

This enables corpus-based lexical exploration and quantitative analysis of lemma attestations across Latin works.

---

# 5. LiLa CSV Export

The `export_lila_lemma_tokens_csv` tool exports LiLa corpus attestation results as a CSV file.

The exported CSV includes:

- token forms
- token URIs
- work titles

The tool is designed for:

- corpus analysis
- spreadsheet analysis
- downstream NLP workflows
- Digital Humanities research pipelines

---

# Installation

Install dependencies with:

```bash
uv sync
```

---

# Running the Server

## Recommended

```bash
uv run mcp-latin
```

## Alternative

```bash
uv run python -m mcp_latin -vv
```

The MCP server will run at:

```
http://localhost:8001/mcp
```

---

# MCP Inspector

You can test the server locally with:

```bash
npx @modelcontextprotocol/inspector
```

Then connect Inspector to:

```
http://localhost:8001/mcp
```

---

# Example Prompts

## Tokenization

```text
Use the MCP tool tokenize_latin_text on:
Senatus populusque Romanus.
```

---

## Parsing

```
Use the MCP tool parser on:
Non potui, inquit, sustinere illud durum spectaculum.
```

---

## Reported Speech Detection

```
Use the Latin MCP tools only.

1. Parse:
HISPO ROMANIUS alio colore dixit illam non amore adulescentis sed odio patris sui secutam

2. Detect reported speech.

```

---

## LiLa Query

```
Use the MCP tool get Lila information on the  "probabilis".
```


## LiLa Query

```
Use the MCP tool get occurrences of the on the lemma "probabilis" and export the results.
```


---

# Development Container

A reproducible VS Code devcontainer is included in:

.devcontainer/

See:

.devcontainer/README.md

for details.

---

# Notes

- UDPipe requests require internet access.
- Hugging Face model weights are downloaded automatically.
- The server is designed for MCP-compatible clients such as Claude Desktop and VS Code MCP integration.
