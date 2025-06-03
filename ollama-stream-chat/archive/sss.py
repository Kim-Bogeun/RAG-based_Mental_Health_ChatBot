import pandas as pd

# =============================================================================
# 1.── 설정: 입력/출력 파일 경로
# =============================================================================
# 실제 엑셀 파일 경로로 수정하세요 (예: '/Users/kim-bogeun/.../cognitive-distortion-data2.xlsx')
INPUT_EXCEL_PATH = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/cognitive-distortion-data2.xlsx'

# 결과를 저장할 경로 (원하는 위치로 수정하세요)
OUTPUT_CSV_PATH   = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/cognitive_distortion_labeled.csv'
OUTPUT_EXCEL_PATH = '/Users/kim-bogeun/projects/ollama-stream-chat/archive/cognitive_distortion_labeled.xlsx'

# =============================================================================
# 2.── 인지 왜곡 목록 및 ID 매핑 준비 (정확한 대소문자 그대로)
# =============================================================================
canonical = [
    "All-or-Nothing Thinking",
    "Overgeneralizing",
    "Labeling",
    "Fortune Telling",
    "Mind Reading",
    "Emotional Reasoning",
    "Should Statements",
    "Personalizing",
    "Disqualifying the Positive",
    "Catastrophizing",
    "Comparing and Despairing",
    "Blaming",
    "Negative Feeling or Emotion"
]
# 위 13개 항목을 순서대로 1,2,3,…,13번으로 매핑
name_to_id = {name: idx + 1 for idx, name in enumerate(canonical)}

# =============================================================================
# 3.── 엑셀 파일 불러오기
#     - 반드시 실제 있는 경로를 지정해야 합니다.
#     - 엑셀에는 최소 'Thinking Traps', 'Definition', 'Example', 'Tips to Overcome' 컬럼이 있어야 합니다.
# =============================================================================
df = pd.read_excel(INPUT_EXCEL_PATH)

# =============================================================================
# 4.── 'Thinking Traps' 칼럼 값이 canonical 목록에 정확히 일치하면 ID 매핑
# =============================================================================
def get_distortion_id_exact(label):
    if pd.isna(label):
        # NaN (결측치)이면 None 반환 → 저장 시 NULL 처리됨
        return None
    stripped = label.strip()
    return name_to_id.get(stripped, None)

df['distortion_id'] = df['Thinking Traps'].apply(get_distortion_id_exact)

# =============================================================================
# 5.── 매핑되지 않은 값이 있는지 확인 (옵션)
#     - 'Thinking Traps' 칼럼에 canonical 목록에 없는 문자열이 있다면 여기서 확인됩니다.
# =============================================================================
unmapped = df[df['distortion_id'].isna() & df['Thinking Traps'].notna()]
if not unmapped.empty:
    print("❌ 매핑되지 않은 레이블:")
    print(unmapped['Thinking Traps'].unique())

# =============================================================================
# 6.── 결과를 새로운 CSV, XLSX 파일로 저장
# =============================================================================
df.to_csv(OUTPUT_CSV_PATH, index=False)
df.to_excel(OUTPUT_EXCEL_PATH, index=False)

print(f"✅ 레이블링 완료. CSV 저장 위치: {OUTPUT_CSV_PATH}")
print(f"✅ 레이블링 완료. XLSX 저장 위치: {OUTPUT_EXCEL_PATH}")