import os
from chromadb.config import Settings

#Initialize Client

CHROMA_SETTINGS = Settings(is_persistent=True, 
                              persist_directory='db',
                              anonymized_telemetry=False)