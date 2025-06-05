import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_batch

# =============================================================================
# 1. 설정: 파일 경로 및 DB 접속 정보
# =============================================================================
examples_PATH   = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/distortion_examples.csv'
description_PATH = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/distortion_description.csv'
reframing_PATH   = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/reframing_dataset.csv'

MODEL_NAME = 'all-MiniLM-L6-v2'

DB_PARAMS = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'cognitive_distortion',
    'user': 'postgres',
    'password': ''
}

# =============================================================================
# 2. 임베딩 모델 로드
# =============================================================================
model = SentenceTransformer(MODEL_NAME)

# =============================================================================
# 3. CSV 데이터 로딩
# =============================================================================
# 3-1) 기존 “왜곡 예시” 데이터
df_examples   = pd.read_csv(examples_PATH)
df_examples['Distortion_ID'] = df_examples['Distortion_ID'].fillna(0).astype(int)

# 3-2) 왜곡 설명 데이터
df_definition = pd.read_csv(description_PATH)


# 3-3) 리프레이밍 데이터
df_reframe = pd.read_csv(reframing_PATH)

# =============================================================================
# 4. 임베딩 생성
# =============================================================================
# 4-1) 예시 데이터(df_examples)의 “Thought”를 임베딩
df_examples['embedding'] = df_examples['Thought'].astype(str).apply(
    lambda x: model.encode(x).tolist()
)

# =============================================================================
# 5. DB 연결 및 테이블 생성/삽입
# =============================================================================
conn = None
try:
    conn = psycopg2.connect(**DB_PARAMS)
    cur  = conn.cursor()

    # ========= 기존 테이블 삭제 (초기화) =========
    cur.execute("DROP TABLE IF EXISTS logs;")
    cur.execute("DROP TABLE IF EXISTS users;")
    cur.execute("DROP TABLE IF EXISTS example_dataset;")
    cur.execute("DROP TABLE IF EXISTS example_embeddings;")
    cur.execute("DROP TABLE IF EXISTS reframing_dataset;")
    cur.execute("DROP TABLE IF EXISTS reframe_embeddings;")
    cur.execute("DROP TABLE IF EXISTS distortions;")

    # ========= 1) 왜곡 설명 테이블(distortions) 생성 =========
    cur.execute("""
        CREATE TABLE distortions (
            distortion_id INTEGER PRIMARY KEY,
            trap_name     TEXT NOT NULL,
            definition    TEXT,
            example       TEXT,
            tips          TEXT
        );
    """)

    insert_distortions_sql = """
        INSERT INTO distortions (
            distortion_id, trap_name, definition, example, tips
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (distortion_id) DO NOTHING;
    """

    records_distortions = [
        (
            int(row['Distortion_ID']),
            row['Distortion'],
            row['Definition'],
            row['Example'],
            row['Tips to Overcome']
        )
        for _, row in df_definition.iterrows()
        if not pd.isna(row['Distortion_ID'])
    ]
    execute_batch(cur, insert_distortions_sql, records_distortions)

    # ========= 2) “예시 임베딩” 테이블(example_embeddings) 생성 =========
    #   - df_examples의 각 Thought 임베딩을 저장
    cur.execute("""
        CREATE TABLE example_embeddings (
            embedding_id SERIAL PRIMARY KEY,
            embedding    vector(384) NOT NULL
        );
    """)

    insert_example_embeddings_sql = """
        INSERT INTO example_embeddings (embedding) VALUES (%s) RETURNING embedding_id;
    """

    example_embedding_ids = []
    for vec in df_examples['embedding']:
        cur.execute(insert_example_embeddings_sql, (vec,))
        eid = cur.fetchone()[0]
        example_embedding_ids.append(eid)

    # df_examples에 embedding_id 컬럼 추가
    df_examples['embedding_id'] = example_embedding_ids

    # ========= 3) “예시 데이터” 테이블(example_dataset) 생성 =========
    #   - ID (CSV), Thought, Distortion, Distortion_ID, embedding_id (FK)
    cur.execute("""
        CREATE TABLE example_dataset (
            id             INTEGER PRIMARY KEY,
            thought        TEXT NOT NULL,
            distortion     TEXT,
            distortion_id  INTEGER,
            embedding_id   INTEGER,
            CONSTRAINT fk_distortion_example FOREIGN KEY (distortion_id)
                REFERENCES distortions (distortion_id)
                ON DELETE SET NULL,
            CONSTRAINT fk_embedding_example FOREIGN KEY (embedding_id)
                REFERENCES example_embeddings (embedding_id)
                ON DELETE SET NULL
        );
    """)

    insert_example_dataset_sql = """
        INSERT INTO example_dataset (
            id, thought, distortion, distortion_id, embedding_id
        ) VALUES (%s, %s, %s, %s, %s);
    """

    records_examples = [
        (
            int(row['ID']),
            row['Thought'],
            row['Distortion'],
            int(row['Distortion_ID']),
            int(row['embedding_id'])
        )
        for _, row in df_examples.iterrows()
    ]
    execute_batch(cur, insert_example_dataset_sql, records_examples)

    # ========= 5) “리프레이밍 데이터” 테이블(reframing_dataset) 생성 =========
    cur.execute("""
        CREATE TABLE reframing_dataset (
            situation            TEXT,
            thought              TEXT,
            reframe              TEXT,
            distortion_id        INTEGER,
            CONSTRAINT fk_distortion_reframe FOREIGN KEY (distortion_id)
                REFERENCES distortions (distortion_id)
                ON DELETE SET NULL
        );
    """)

    insert_reframing_dataset_sql = """
        INSERT INTO reframing_dataset (
            situation, thought, reframe, distortion_id
        ) VALUES (%s, %s, %s, %s);
    """

    records_reframes = [
        (
            row.get('situation'),
            row['thought'],
            row.get('reframe'),
            int(row['distortion_id'])
        )
        for _, row in df_reframe.iterrows()
    ]
    execute_batch(cur, insert_reframing_dataset_sql, records_reframes)


    # ========= 6) users 테이블 생성 (기존과 동일) =========
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    TEXT PRIMARY KEY
        );
    """)
    print("✅ users 테이블 준비 완료")

    # ======= 6) logs 테이블 생성 =======
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            log_id     SERIAL PRIMARY KEY,
            user_id    TEXT REFERENCES users(user_id) ON DELETE SET NULL,
            query      TEXT    NOT NULL,
            answer     TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)
    print("✅ logs 테이블 준비 완료")

    # ========= 8) 커밋 =========
    conn.commit()
    print(f"✅ distortions: {len(records_distortions)}개, "
          f"example_dataset: {len(records_examples)}개, "
          f"reframing_dataset: {len(records_reframes)}개 삽입 완료.")

except Exception as e:
    if conn:
        conn.rollback()
    print("🚨 처리 중 에러 발생:", e)

finally:
    if conn:
        cur.close()
        conn.close()