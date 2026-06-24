"""Application configuration file."""

FALLBACK_MODELS:list[str] =[
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "gpt-3.5-turbo"
]

CHUNK_SIZE:int = 1000
CHUNK_OVERLAP:int = 200

TOP_K:int = 4