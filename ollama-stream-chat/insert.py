import pandas as pd
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import execute_batch

# =============================================================================
# 1. 설정: 파일 경로 및 DB 접속 정보
# =============================================================================
reframing_PATH = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/reframing_dataset_rev.csv'
distortion_PATH = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/distortions_cleaned_full.csv'
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
# 3. 데이터 로딩
# =============================================================================
df_reframing = pd.read_csv(reframing_PATH)
df_definition = pd.read_csv(distortion_PATH)

# ▶ NaN distortion_id → 14 ("Unknown")으로 변환
df_reframing['distortion_id'] = df_reframing['distortion_id'].fillna(14)

# ▶ 임베딩 처리
def make_concat(row):
    return str(row['situation']).strip() + ' ' + str(row['thought']).strip()

df_reframing['text_to_embed'] = df_reframing.apply(make_concat, axis=1)
df_reframing['comb_embedding'] = df_reframing['text_to_embed'].apply(lambda txt: model.encode(txt).tolist())
df_definition['embedding'] = df_definition['Example'].astype(str).apply(lambda x: model.encode(x).tolist())

# =============================================================================
# 4. DB 연결 및 테이블 생성 / 삽입
# =============================================================================
conn = None
try:
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()

    # 테이블 DROP (기존 데이터 완전 삭제 후 재삽입)
    cur.execute("DROP TABLE IF EXISTS reframing_dataset;")
    cur.execute("DROP TABLE IF EXISTS distortions;")

    # 4-1. distortions 테이블 생성
    cur.execute("""
        CREATE TABLE distortions (
            distortion_id INTEGER PRIMARY KEY,
            trap_name TEXT NOT NULL,
            definition TEXT,
            example TEXT,
            tips TEXT
        );
    """)

    # 4-2. reframing_dataset 테이블 생성
    cur.execute("""
        CREATE TABLE reframing_dataset (
            id SERIAL PRIMARY KEY,
            situation TEXT,
            thought TEXT,
            reframe TEXT,
            distortion_id INTEGER,
            comb_embedding vector(384),
            CONSTRAINT fk_reframing_distortion
                FOREIGN KEY (distortion_id)
                REFERENCES distortions (distortion_id)
                ON UPDATE CASCADE
                ON DELETE SET NULL
        );
    """)

    # 4-3. distortions 삽입 (정상 데이터 + Unknown 추가)
    insert_distortions_sql = """
        INSERT INTO distortions (
            distortion_id, trap_name, definition, example, tips
        ) VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (distortion_id) DO NOTHING;
    """
    records_distortions = [
        (
            row['distortion_id'],
            row['Thinking Traps'],
            row['Definition'],
            row['Example'],
            row['Tips to Overcome']
        )
        for _, row in df_definition.iterrows()
        if not pd.isna(row['distortion_id'])
    ]

    # ▶ Unknown 레이블 수동 추가
    records_distortions.append((
        14,
        'None',
        'This case does not involve any cognitive distortion.',
        None,
        None
    ))

    execute_batch(cur, insert_distortions_sql, records_distortions)

    # 4-4. reframing_dataset 삽입
    insert_reframing_sql = """
        INSERT INTO reframing_dataset (
            situation, thought, reframe, distortion_id, comb_embedding
        ) VALUES (%s, %s, %s, %s, %s);
    """
    records_reframing = [
        (
            row['situation'],
            row['thought'],
            row['reframe'],
            int(row['distortion_id']),
            row['comb_embedding']
        )
        for _, row in df_reframing.iterrows()
    ]
    execute_batch(cur, insert_reframing_sql, records_reframing)

    conn.commit()
    print(f"✅ distortions: {len(records_distortions)}건, reframing_dataset: {len(records_reframing)}건 삽입 완료.")

    # 5-4) users 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    password TEXT NOT NULL,
    time_stamp TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    print("✅ users 테이블 준비 완료")

    # 5-5) logs 테이블
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
    input_id UUID PRIMARY KEY,
    query TEXT NOT NULL,
    answer TEXT, 
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    print("✅ logs 테이블 준비 완료")

except Exception as e:
    if conn:
        conn.rollback()
    print("🚨 처리 중 에러 발생:", e)


finally:
    if conn:
        cur.close()
        conn.close()