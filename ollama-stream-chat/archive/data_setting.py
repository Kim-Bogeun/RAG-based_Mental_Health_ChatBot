import pandas as pd

# 1. CSV 불러오기
csv_path = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/reframing_dataset.csv'
df_reframing = pd.read_csv(csv_path)

# 2. 인지 왜곡(Thinking Traps) 목록 정의 및 ID 매핑
canonical = [
    "all-or-nothing thinking",
    "overgeneralizing",
    "labeling",
    "fortune telling",
    "mind reading",
    "emotional reasoning",
    "should statements",
    "personalizing",
    "disqualifying the positive",
    "catastrophizing",
    "comparing and despairing",
    "blaming",
    "negative feeling or emotion"
]

name_to_id = {name: idx + 1 for idx, name in enumerate(canonical)}

# 3. 첫 번째 레이블만 취하고 ID 매핑 함수 정의
def get_first_distortion_id(val):
    if pd.isna(val):
        return None
    normalized = val.strip().lower()
    if normalized == 'none':
        return None
    first_label = normalized.split(',')[0].strip()
    return name_to_id.get(first_label, None)

# 4. 새로운 컬럼 'distortion_id' 생성
df_reframing['distortion_id'] = df_reframing['thinking_traps_addressed'].apply(get_first_distortion_id)

# 5. 매핑되지 않는(유효하지 않은) 첫 번째 레이블 확인 (None으로 처리된 경우)
unmapped = df_reframing[
    (~df_reframing['thinking_traps_addressed'].isna()) &
    (df_reframing['thinking_traps_addressed'].str.strip().str.lower() != 'none') &
    (df_reframing['distortion_id'].isna())
]

df_reframing.head(10), unmapped['thinking_traps_addressed'].unique()

df_reframing.to_csv('/Users/kim-bogeun/projects/ollama-stream-chat/archive/reframing_dataset_rev.csv', index=False)