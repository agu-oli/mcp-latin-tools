from fastmcp import FastMCP, Context
from pydantic import BaseModel
from fastmcp.exceptions import ToolError
import requests
import uuid

import pandas as pd
from mcp_latin.tokenizer import tokenize, expand_tokens_with_que, split_sentences, normalize

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import hf_hub_download
import json


_INSTRUCTIONS = (

    "Use tokenize_latin_text for sentence splitting and enclitic -que tokenization. "

    """
    Run UDPipe morphological and syntactic analysis on Latin text.

    Returns CoNLL-U formatted output including:
    - tokenization
    - lemmatization
    - POS tagging
    - morphology
    - dependency parsing
    """

    "For reported speech detection, first use parser to obtain CoNLL-U output. "
    "Then use prepare_latin_input to obtain a prepared_id. "
    "Then use detect_reported_speech with that prepared_id. "

    "Use get_lila_lemma_info to query the LiLa Knowledge Base for lexical information about a lemma from a token form. "

    "Find the following information from the Lila Knowledge Base for a given lemma:" \
    "- Lemma variants\n" \
    "Gender\n" \
    "Base\n" \
    "Inflection type\n" \
    "Part of speech\n" \
    "Prefix\n" \
    "Suffix\n" \
    
    "Use get_lila_lemma_tokens_dataframe to retrieve corpus token occurrences for a lemma and count occurrences per work. "

    "Use export_lila_lemma_tokens_csv to export the LiLa token occurrences query results in a csv file. "
    
    "Do not run Python or bash commands. "
    "Always use MCP tools for Latin analysis."

)

mcp = FastMCP(name = "Latin tools MCP",
              instructions=_INSTRUCTIONS)

prepared_store = {}
prediction_store = {}


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic input and output models for MCP tools
# ──────────────────────────────────────────────────────────────────────────────


class TokenizedToken(BaseModel):
    form: str
    token_type: str  # "single" or "enclitic_split"
    original_surface: str | None = None  # only for enclitic_split, the original token surface


class TokenizedSentence(BaseModel):
    sentence_id: int
    tokens: list[TokenizedToken]

class TokenizationResult(BaseModel):
    sentences: list[TokenizedSentence]  


class PreparedInput(BaseModel):
    prepared_id: str

    tokens: list[str]
    lemmas: list[str]
    pos: list[str]
    morph: list[str]

    non_finite_verb: list[int]

    lemma_ids: list[int] | None = None
    pos_ids: list[int] | None = None

class PrepareOutput(BaseModel):
    prediction_id: str
    reported_speech: list[int]
    confidence: list[float]


class LilaLemmaTokenRow(BaseModel):
    token: str
    token_uri: str
    work: str


class LilaLemmaTokensDataFrame(BaseModel):
    lemma: str
    total_occurrences: int
    work_counts: dict[str, int]
    rows: list[LilaLemmaTokenRow]


class LilaCSVExport(BaseModel):
    lemma: str
    csv_path: str
    total_occurrences: int
    work_counts: dict[str, int]

# ══════════════════════════════════════════════════════════════════════════════
# TOOLS 
# 1. Tokenzation: tokenizer, lemmatizer, POS tagger, and morph analyzer, non-finite verb identifier
# the tool call UDPipe API to get all the information 
# it uses latin-evalatin24-240520, evaluated on EvaLatin campaign in 2024 and trained with on Latin Dependency Treebanks
# EvaLatin overview: https://aclanthology.org/2024.lt4hala-1.21/
# Model repository: https://github.com/ufal/evalatin2024-latinpipe

# 2.Reported speech detection:

# A. and preparation tool for reported speech detection: 
# It prepares the input for the reported speech detection model, 
# It aligns the tokenization output with the original tokens and prepares the input format for the model.

# B.Reported speech detector: a single reported speech predictor which takes the output of 1 as input
# It returns a list of token predicted to be in reported speech, along with confidence score
# The model is the first experiment of a latin reported speech detection model at the token level
# It is a LaBerta fine-tuned model for token classification.
# Model repository: https://huggingface.co/agudei/latin-reported-speech-laberta
# Paper describing the experiment: https://aclanthology.org/2026.latechclfl-1.24/

# 3. LiLa lemma info tool: Query a Knowledge Base easily
# A tool to query the LiLa Knowledge Base departing from a lemma.
# The tool is designe to help users query LiLa
# Sparql code is difficult. 
# The tool is designed to run a general SPARQL query to get various information about a lemma.

# 4. LiLa Corpus Attestation Retrieval

# `get_lila_lemma_tokens_dataframe` tool retrieves corpus attestations linked to a Latin lemma in the LiLa Knowledge Base.
# retrieves :
# - token occurrences associated with a lemma
# - token URIs
# - work titles
# computes occurrence frequencies per work
# structures results as a dataframe output

# 5. LiLa CSV Export

# `export_lila_lemma_tokens_csv` tool exports LiLa corpus attestation results as a CSV file.

# Exported CSV includes:

# - token forms
# - token URIs
# - work titles

# The tools are designed to be used in sequence, but they can also be used independently.
# Prompts suggestions in the readme file.

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION TOOL
# ══════════════════════════════════════════════════════════════════════════════


def is_non_finite_verb(upos: str, feats: str) -> int:
    if upos not in {"VERB", "AUX"}:
        return 0

    non_finite_forms = {
        "VerbForm=Inf",
        "VerbForm=Part",
    }

    return int(any(form in feats for form in non_finite_forms))

model = "latin-evalatin24-240520"
base_url = "https://lindat.mff.cuni.cz/services/udpipe/api"

@mcp.tool(annotations={"readOnlyHint": True})
async def tokenize_latin_text(
    text: str,
    ctx: Context
) -> TokenizationResult:
    """Tokenize Latin text with sentence splitting and enclitic -que splitting."""

    await ctx.info("Running Latin tokenizer with -que splitting.")

    text = normalize(text)
    sentence_strings = split_sentences(text)

    if not sentence_strings:
        await ctx.warning("No sentences found in input text.")
        return TokenizationResult(sentences=[])

    sentences: list[TokenizedSentence] = []

    for sent_id, sent in enumerate(sentence_strings, start=1):
        raw_tokens = tokenize(sent)
        expanded = expand_tokens_with_que(raw_tokens)

        tokens: list[TokenizedToken] = []

        for item in expanded:
            if item["type"] == "single":
                tokens.append(
                    TokenizedToken(
                        form=item["form"],
                        token_type="single",
                    )
                )
            else:
                for part in item["parts"]:
                    tokens.append(
                        TokenizedToken(
                            form=part,
                            token_type="enclitic_split",
                            original_surface=item["surface"],
                        )
                    )

        sentences.append(
            TokenizedSentence(
                sentence_id=sent_id,
                tokens=tokens,
            )
        )

    return TokenizationResult(sentences=sentences)      

# ══════════════════════════════════════════════════════════════════════════════
# PARSER AND PREPARE INPUT FOR REPORTED SPEECH DETECTION TOOLS
# ══════════════════════════════════════════════════════════════════════════════



def call_udpipe_api(text: str, model = model, base_url = base_url) -> str:
    resp = requests.post(
        f"{base_url}/process",
        data={
        "model": model,
        "tokenizer": "",
        "tagger": "",
         "parser": "", 
        "data": text,
    },
    timeout=60,
)

    resp.raise_for_status()
    result = resp.json()

    return(result["result"])


def parse_conllu(doc: str):

    rows = []

    for line in doc.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        cols = line.split("\t")

        # skip malformed lines
        if len(cols) != 10:
            continue

        # skip multiword tokens like 2-3
        if "-" in cols[0]:
            continue

        rows.append({
            "form": cols[1],
            "lemma": cols[2],
            "upos": cols[3],
            "feats": cols[5],
            "head": cols[6],
            "deprel": cols[7],
        })

    return rows

@mcp.tool(annotations={"readOnlyHint": True})
async def parser(
    text: str,
    ctx: Context
) -> str:

    """
    Morphological analysis and preparation of Latin text for reported speech detection:
    tokenization, lemmatization, POS tagging, morphology, syntax,
    and non-finite verb detection.
    It prepares Latin text for the subsquent reported speech detection:

    """

    await ctx.info("Parsing Latin input with UDPipe.")

    conllu_output = call_udpipe_api(text)

    return conllu_output


@mcp.tool(annotations={"readOnlyHint": True})
async def prepare_latin_input(
    conllu_output: str,
    ctx: Context
) -> PreparedInput:
    """Convert UDPipe CoNLL-U output into model-ready input."""

    rows = parse_conllu(conllu_output)

    if not rows:
        raise ToolError("No tokens returned by UDPipe.")

    prepared_input = PreparedInput(
        prepared_id=str(uuid.uuid4()),
        tokens=[r["form"] for r in rows],
        lemmas=[r["lemma"] for r in rows],
        pos=[r["upos"] for r in rows],
        morph=[r["feats"] for r in rows],
        non_finite_verb=[
            is_non_finite_verb(r["upos"], r["feats"])
            for r in rows
        ],
    )
 
    prepared_store[prepared_input.prepared_id] = prepared_input # Store this value under prepared_id.

    return prepared_input 

# ══════════════════════════════════════════════════════════════════════════════
# REPORTED SPEECH DETECTOR
# ══════════════════════════════════════════════════════════════════════════════


def align(lemmas, pos, nfv, word_ids, lemma2id, pos2id):
    lemma_ids=[]
    pos_ids=[]
    nfv_ids=[]

    for w in word_ids:
        if w is None:
            lemma_ids.append(lemma2id["[PAD]"])
            pos_ids.append(pos2id["[PAD]"])
            nfv_ids.append(0)
        else:
            lemma_ids.append(
                lemma2id.get(
                    lemmas[w],
                    lemma2id["[UNK]"]
                )
            )
            pos_ids.append(
                pos2id.get(
                    pos[w],
                    pos2id["[UNK]"]
                )
            )
            nfv_ids.append(nfv[w])

    return lemma_ids,pos_ids,nfv_ids



MODEL_REPO = "agudei/latin-reported-speech-laberta"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        add_prefix_space=True
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_REPO,
        trust_remote_code=True
    )

    model.eval()

    lemma_path = hf_hub_download(
        MODEL_REPO,
        "lemma2id.json"
    )

    pos_path = hf_hub_download(
        MODEL_REPO,
        "pos2id.json"
    )

    with open(lemma_path, encoding="utf-8") as f:
        lemma2id = json.load(f)

    with open(pos_path, encoding="utf-8") as f:
        pos2id = json.load(f)

    return tokenizer, model, lemma2id, pos2id


model_package = None

def get_model():
    global model_package

    if model_package is None:
        model_package = load_model()

    return model_package


@mcp.tool(annotations={"readOnlyHint": True})
async def detect_reported_speech(
    prepared_id: str,
    ctx: Context
) -> PrepareOutput:
    """Run reported speech detection on an already prepared Latin input."""

    await ctx.info("Running reported speech detection from prepared_id.")



    prepared_input = prepared_store.get(prepared_id)

    if prepared_input is None:
        raise ToolError(f"No prepared input found for id: {prepared_id}")
    
    tokenizer, model, lemma2id, pos2id = get_model()


    tokens = prepared_input.tokens
    lemmas = prepared_input.lemmas
    pos = prepared_input.pos
    nfv = prepared_input.non_finite_verb

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    

    word_ids = enc.word_ids(batch_index=0)

    lemma_ids, pos_ids, nfv_ids = align(
        lemmas,
        pos,
        nfv,
        word_ids,
        lemma2id,
        pos2id,
    )


    inputs = {
        **enc,
        "lemma_ids_aligned": torch.tensor([lemma_ids]),
        "pos_ids_aligned": torch.tensor([pos_ids]),
        "non_finite_verb_aligned": torch.tensor([nfv_ids]),
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    preds = probs.argmax(dim=-1)

    word_preds = []
    word_conf = []
    seen = set()

    for i, w in enumerate(word_ids):
        if w is None or w in seen:
            continue

        seen.add(w)
        word_preds.append(int(preds[i]))
        word_conf.append(float(probs[i, 1]))

    prediction = PrepareOutput(
        prediction_id=str(uuid.uuid4()),
        reported_speech=word_preds,
        confidence=word_conf,
    )

    prediction_store[prediction.prediction_id] = prediction


    return prediction



# ══════════════════════════════════════════════════════════════════════════════
# MCP TOOL FOR LiLa Knowledge Base Sparql queries
# ══════════════════════════════════════════════════════════════════════════════



lila_endpoint = "https://lila-erc.eu/sparql/lila_knowledge_base/sparql"


def normalize_lila_lemma(lemma: str) -> str:
    return (
        lemma
        .replace('"', '\\"')
        .lower()
        .replace("v", "u")
        .replace("j", "i")
    )


def lila_lemma_tokens_df(lila_output: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(lila_output)


def get_lemma_from_token(text: str) -> str:
    """Use UDPipe to retrieve the lemma of the token given by the user."""

    conllu_output = call_udpipe_api(text)
    rows = parse_conllu(conllu_output)

    if not rows:
        raise ToolError(f"No lemma found for input: {text}")

    return rows[0]["lemma"]

@mcp.tool(annotations={"readOnlyHint": True})
async def get_lila_lemma_info(
    lemma: str,
    ctx: Context
) -> str:

    """Query LiLa for information about a Latin token form."""

    await ctx.info(f"Retrieving lemma with UDPipe for input: {lemma}")

    lemma = get_lemma_from_token(lemma)

    await ctx.info(f"Querying LiLa for lemma: {lemma}")

    safe_lemma = normalize_lila_lemma(lemma)

    query = f"""
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>
PREFIX lime: <http://www.w3.org/ns/lemon/lime#>
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos#>

SELECT DISTINCT ?l ?p ?v ?definition
       (GROUP_CONCAT(DISTINCT ?varLabel; separator=", ") AS ?variants)
WHERE {{
    ?l rdfs:label "{safe_lemma}" ;
       ?p ?v ;
       dcterms:isPartOf <http://lila-erc.eu/data/id/lemma/LemmaBank> .

    OPTIONAL {{
    <http://lila-erc.eu/data/lexicalResources/LewisShort/Lexicon> lime:entry ?le. # Lewis and Short dictionary definition
    ?le ontolex:canonicalForm ?l;  # the lexical entry ?le has canonical form ?l
        ontolex:sense ?lewishortdefinition.}}

    OPTIONAL {{
        ?le ontolex:canonicalForm ?l ;
            ontolex:sense ?sense .
        ?sense skos:definition ?definition .  # other dictionaries definitions linked to the lemma, not Lewis and Short
    }}

    OPTIONAL {{
        ?l lila:lemmaVariant ?var .
        ?var rdfs:label ?varLabel .
    }}
}}

GROUP BY ?l ?p ?v ?definition

LIMIT 50
"""

    resp = requests.post(
        lila_endpoint,
        data={
            "query": query,
            "format": "json",
        },
        timeout=30,
    )

    resp.raise_for_status()
    data = resp.json()
    bindings = data.get("results", {}).get("bindings", [])

    if not bindings:
        return f"No LiLa information found for lemma '{lemma}'."
    
    output = [f"Résultats pour '{lemma}':"]

    properties = {
        "http://lila-erc.eu/ontologies/lila/lemmaVariant": "Lemma variant",
        "http://lila-erc.eu/ontologies/lila/hasGender": "Gender",
        "http://lila-erc.eu/ontologies/lila/hasBase": "Base",
        "http://lila-erc.eu/ontologies/lila/hasInflectionType": "Inflection type",
        "http://lila-erc.eu/ontologies/lila/hasPOS": "Part of speech",
        "http://lila-erc.eu/ontologies/lila/hasPrefix": "Prefix",
        "http://lila-erc.eu/ontologies/lila/hasSuffix": "Suffix",
    }

    for b in bindings:
        uri = b.get("l", {}).get("value", "N/A")
        definition = b.get("definition", {}).get("value", "No definition available")
        #variants = b.get("varLabels", {}).get("value", "Aucune")

        p_uri = b.get("p", {}).get("value")
        v_value = b.get("v", {}).get("value", "N/A")

        if p_uri not in properties:
            continue

        property_name = properties[p_uri]

        output.append(
        f"- URI: {uri}\n"
        f"  Property: {property_name}\n"
        f"  Value: {v_value}\n"
        f"  Definition: {definition}\n"
        #f"  Variants: {variants}"
    )
        
    return "\n\n".join(output)




@mcp.tool(annotations={"readOnlyHint": True})
async def get_lila_lemma_tokens_dataframe(
    lemma: str,
    ctx: Context,
    limit: int = 1000,
) -> LilaLemmaTokensDataFrame:
    """
    Retrieve all LiLa corpus token occurrences for a Latin lemma,
    create a pandas DataFrame,
    and compute occurrence counts by work.
    """

    await ctx.info(f"Retrieving lemma with UDPipe for input: {lemma}")

    lemma = get_lemma_from_token(lemma)


    await ctx.info(
        f"Querying LiLa token occurrences for lemma: {lemma}, 1000 is default limit"
    )

    safe_lemma = normalize_lila_lemma(lemma)

    query = f"""
PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX powla: <http://purl.org/powla/powla.owl#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX lila: <http://lila-erc.eu/ontologies/lila/>

SELECT ?tokenURI ?tokenLabel ?lemma ?label
WHERE {{
  ?lemma ontolex:writtenRep "{safe_lemma}"@la .

  ?tokenURI lila:hasLemma ?lemma ;
            rdf:type powla:Terminal ;
            powla:hasStringValue ?tokenLabel .

  ?tokenURI powla:hasLayer ?docLayer .
  ?docLayer powla:hasDocument ?doc .

  ?doc dc:title ?label .
}}

ORDER BY ?label
LIMIT {limit}
"""

    resp = requests.post(
        lila_endpoint,
        data={
            "query": query,
            "format": "json",
        },
        headers={
            "Accept": "application/sparql-results+json"
        },
        timeout=60,
    )

    resp.raise_for_status()

    data = resp.json()

    bindings = data.get("results", {}).get("bindings", [])

    if not bindings:
        raise ToolError(
            f"No LiLa corpus attestations found for lemma '{lemma}'."
        )

    output = []

    for b in bindings:

        output.append({
            "token": b["tokenLabel"]["value"],
            "token_uri": b["tokenURI"]["value"],
            "work": b["label"]["value"],
        })

    df = lila_lemma_tokens_df(output)
    
    work_counts = (
        df["work"]
        .value_counts()
        .to_dict()
    )

    rows = [
        LilaLemmaTokenRow(
            token=row["token"],
            token_uri=row["token_uri"],
            work=row["work"],
        )
        for row in output
    ]

    return LilaLemmaTokensDataFrame(
        lemma=lemma,
        total_occurrences=len(df),
        work_counts=work_counts,
        rows=rows,
    )

@mcp.tool(annotations={"readOnlyHint": True})
async def export_lila_lemma_tokens_csv(
    lemma: str,
    ctx: Context,
    limit: int = 1000,
) -> LilaCSVExport:
    
    """
    Export LiLa corpus token occurrences for a lemma as a CSV file.
    """


    await ctx.info(
        f"Exporting LiLa corpus attestations for lemma: {lemma}"
    )

    output = await get_lila_lemma_tokens_dataframe(
        lemma=lemma,
        ctx=ctx,
        limit=limit,
    )


    df = pd.DataFrame([
    {
        "token": row.token,
        "token_uri": row.token_uri,
        "work": row.work,
    }
    for row in output.rows
    ])

    safe_name = normalize_lila_lemma(lemma)

    csv_path = f"/tmp/{safe_name}_lila_tokens.csv"

    df.to_csv(csv_path, index=False)

    return LilaCSVExport(
        lemma=lemma,
        csv_path=csv_path,
        total_occurrences=output.total_occurrences,
        work_counts=output.work_counts,
    )





# ══════════════════════════════════════════════════════════════════════════════
# App factory — called from __main__.py
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    return mcp.http_app(path="/mcp")
