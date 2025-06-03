# config/settings.py

# 청크 사이즈 및 오버랩 정의 (LangChain 텍스트 분할기용)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 사용할 임베딩 모델 이름
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 또는 "all-MiniLM-L6-v2" 등