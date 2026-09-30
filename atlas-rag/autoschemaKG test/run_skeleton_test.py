"""
Skeleton test for AutoSchemaKG (atlas-rag), following the official quickstart:
https://hkust-knowcomp.github.io/AutoSchemaKG/intro/quickstart.html

LLM backend: local Ollama (llama3.1:latest) via its OpenAI-compatible endpoint,
instead of a real OpenAI key or a vLLM server -- no cloud cost, no GPU needed.

Input: data/skeleton_test.jsonl, a single made-up paragraph (NOT NormAd data),
per the docs' JSONL schema {"id", "text", "metadata"} -- confirmed against the
installed package source, since load_dataset() in triple_extraction.py only
accepts .json/.jsonl(.gz), not the .txt/.md the docs also mention.

===============================================================================
REPRODUCIBILITY -- venv setup (venv/ itself is gitignored, not committed)
===============================================================================
Installation guide: https://hkust-knowcomp.github.io/AutoSchemaKG/intro/installation.html#requirements
Tested on: macOS, Apple Silicon (arm64), Python 3.14, no NVIDIA GPU -> CPU-only
PyTorch + faiss-cpu (the guide's CUDA-specific commands do not apply here).

    cd "atlas-rag/autoschemaKG test"
    python3.14 -m venv venv
    source venv/bin/activate

    # Prerequisites (guide's CPU-only commands; use the CUDA commands instead
    # on a Linux/Windows machine with an NVIDIA GPU -- see the guide link above)
    pip install torch torchvision torchaudio
    pip install faiss-cpu

    # Main package + the embedding model dependency the quickstart's RAG step needs
    pip install atlas-rag sentence-transformers

Also required, outside the venv, to actually run this script:
  - Ollama (https://ollama.com) installed and running (`ollama serve`), with
    the `llama3.1:latest` model pulled (`ollama pull llama3.1`). Any other
    Ollama model, or any other OpenAI-compatible endpoint (see the guide's
    "LLM Providers" page), can be substituted by changing the `client`/
    `model_name` below.
"""
import os
from openai import OpenAI

from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator
from sentence_transformers import SentenceTransformer
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
from atlas_rag.vectorstore import create_embeddings_and_index
from atlas_rag.retriever import HippoRAG2Retriever

HERE = os.path.dirname(os.path.abspath(__file__))

print("=== Setting up LLM (Ollama, llama3.1:latest) ===")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
llm_generator = LLMGenerator(client=client, model_name="llama3.1:latest", max_workers=2)

config = ProcessingConfig(
    model_path="llama3.1:latest",
    data_directory=os.path.join(HERE, "data"),
    filename_pattern="skeleton_test",
    output_directory=os.path.join(HERE, "output"),
    max_workers=2,
    batch_size_triple=4,
    debug_mode=False,
)

kg_extractor = KnowledgeGraphExtractor(model=llm_generator, config=config)

print("\n=== STEP 1/5: run_extraction (LLM triple extraction) ===")
kg_extractor.run_extraction()

print("\n=== STEP 2/5: convert_json_to_csv ===")
kg_extractor.convert_json_to_csv()

print("\n=== STEP 3/5: generate_concept_csv_temp (LLM concept generation) ===")
kg_extractor.generate_concept_csv_temp()

print("\n=== STEP 4/5: create_concept_csv ===")
kg_extractor.create_concept_csv()

print("\n=== STEP 5/5: convert_to_graphml ===")
kg_extractor.convert_to_graphml()

print("\n=== Building sentence embeddings + FAISS index ===")
encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_model = SentenceTransformer(encoder_model_name, trust_remote_code=True)
sentence_encoder = SentenceEmbedding(sentence_model)

data = create_embeddings_and_index(
    sentence_encoder=sentence_encoder,
    model_name=encoder_model_name,
    working_directory=config.output_directory,
    keyword=config.filename_pattern,
    include_concept=False,
    include_events=False,
    normalize_embeddings=True,
)

print("\n=== Retrieval + answer generation (HippoRAG2) ===")
hipporag2_retriever = HippoRAG2Retriever(
    llm_generator=llm_generator,
    sentence_encoder=sentence_encoder,
    data=data,
)

query = "Who took over as head baker at the Riverside Bakery, and which restaurants does it supply?"
content, sorted_context_ids = hipporag2_retriever.retrieve(query, topN=3)
sorted_context = "\n".join(content)
print("\n--- Retrieved context ---")
print(sorted_context)

response = llm_generator.generate_with_context(
    query, sorted_context, max_new_tokens=512, temperature=0.5
)
print("\n=== FINAL ANSWER ===")
print(response)
print("\n=== SKELETON TEST COMPLETE ===")
