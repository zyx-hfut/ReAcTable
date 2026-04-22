"""Configuration for local Qwen2.5-3B-Instruct inference via vLLM."""

# vLLM server settings
VLLM_BASE_URL = "http://localhost:8000/v1"
VLLM_API_KEY = "EMPTY"

# Model settings (must match vLLM --served-model-name)
MODEL_NAME = "qwen2.5-3b"

# Generation parameters
MAX_TOKENS = 128  # keep original value; shorter output forces simpler SQL
TEMPERATURE = 0.6  # for majority vote diversity
MAX_RETRY = 3  # API call retry count

# Experiment parameters
REPEAT_TIMES = 5  # majority vote repetitions
MAX_DEMO = 5  # few-shot example count
N_THREADS = 1  # vLLM handles batching internally
LINE_LIMIT = 10  # table row limit per prompt

# Paths
BASE_PATH = "../dataset/WikiTableQuestions/"
DATA_FILE = BASE_PATH + "data/pristine-unseen-tables.tsv"
TEMPLATE = "original-sql-py-no-intermediate"
PROGRAM = "sql-py"
RESULTS_DIR = BASE_PATH + "results/"
