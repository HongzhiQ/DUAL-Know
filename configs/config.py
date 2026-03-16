
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANESGLM_MODEL_PATH = "/path/to/your/AnesGLM"
BGE_MODEL_PATH = "/path/to/your/bge-base-zh-v1.5"

DATA_DIR = os.path.join(PROJECT_ROOT, "graphrag_export", "processed_data_standard")
ENTITY_TABLE_PATH = os.path.join(DATA_DIR, "entity_table.jsonl")
KG_TRIPLES_PATH = os.path.join(DATA_DIR, "kg_triples.jsonl")
KG_GRAPH_PATH = os.path.join(DATA_DIR, "kg_graph.pkl")
TEST_QA_PATH = os.path.join(DATA_DIR, "testQAFinal.jsonl")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
INDEX_DIR = os.path.join(OUTPUT_DIR, "index")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")


REWRITE_TEMPERATURE = 0.6
REWRITE_TOP_P = 0.9


TRIPLE_TEMPERATURE = 0.2
TRIPLE_TOP_P = 0.7

ANSWER_TEMPERATURE = 0.8
ANSWER_TOP_P = 0.7


LLM_TEMPERATURE = ANSWER_TEMPERATURE
LLM_TOP_P = ANSWER_TOP_P

LLM_MAX_INPUT_LENGTH = 1024
LLM_MAX_GEN_LENGTH = 512


NUM_QUERY_REWRITES = 3

EXPLICIT_TOPK = 10
DESC_TOPK = 10
SUBGRAPH_HOP = 2
EMBEDDING_DIM = 768

DGHMA_NUM_LAYERS = 3
DGHMA_NUM_HEADS = 4
DGHMA_HIDDEN_DIM = 256
DGHMA_DROPOUT = 0.1

NUM_CANDIDATE_PATHS = 10
PATH_TOPK = 4
UNCERTAINTY_LAMBDA = 0.5
MAX_PATH_LENGTH = 3

FUSION_WEIGHT_CONF = 0.5
FUSION_WEIGHT_OVERLAP = 0.3
FUSION_WEIGHT_SIM = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32



NODE_EMB_CACHE_DIR = os.path.join(OUTPUT_DIR, "node_emb_cache")


# Optional: "transformers"  | "vllm_offline"  | "vllm_server"
LLM_BACKEND = "vllm_offline"


VLLM_SERVER_URL = "http://localhost:8000"
VLLM_TENSOR_PARALLEL = 1
VLLM_GPU_MEM_UTIL = 0.90
VLLM_MAX_MODEL_LEN = 4096
VLLM_QUANTIZATION = None