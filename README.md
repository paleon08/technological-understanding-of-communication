클랩 모델 로드 완료

---

## 🎯 현재 목표 기반: “언어 지도 조정용 최소 절차”

### 핵심 개념

* **CLAP의 자연어 의미 공간만 사용**
* 오디오 신호 → 임베딩 변환은 나중에 (또는 생략)
* 현재 단계에서는 **텍스트 앵커 + 모델 내부 의미 좌표만으로** 종별 의미 지도 구축

---

## 🧩 간소화된 전체 절차 (불필요 단계 제거판)

```
(1) 앵커 작성 ─→ (2) 텍스트 임베딩 생성 ─→ (3) 의미 매칭 및 시각화 ─→ (4) 조정 및 지식 기반 확장
```

---

### 🧠 **1️⃣ 앵커 설계 (Anchor Definition)**

**목적:**
각 종의 대표적 행동 + 그 의미를 텍스트 문장으로 정의.
이게 사실상 “해석 기준점(semantic pivot)” 역할을 함.

**파일:**
`configs/anchors/{species}.yml`

**예시:**

```yaml
name: crestedgecko
anchors:
  - "tail wave for defense warning"
  - "slow head tilt as curiosity behavior"
  - "rapid limb vibration for stress signal"
```

---

### 🧮 **2️⃣ 텍스트 임베딩 생성 (Text Embedding via CLAP)**

**목적:**
CLAP 모델의 **언어 임베딩 공간**에서 각 앵커 문장을 벡터로 변환.
→ `artifacts/text_anchors/{species}.npy` 로 저장.

**명령어:**

```bash
python lib/scripts/embed_text_anchors.py
```

이 단계는 **“모델의 언어 의미 좌표계 확보”**를 의미.
이후 이 좌표계가 “언어 지도”의 뼈대가 된다.

---

### 🔗 **3️⃣ 의미 매칭 및 시각화 (Semantic Mapping & Visualization)**

**목적:**
동일한 공간에 인간 언어(설명/논문/지식 기반 문장)를 투영하여
**행동 ↔ 의미 간의 위치 관계를 관찰/분류**한다.

**예시 흐름:**

```
anchor_embeddings.npy
        ↓
LLM 또는 텍스트 입력(예: "stress response", "defensive vibration")
        ↓
코사인 유사도 계산 (Top-K)
        ↓
언어 지도(UMAP, t-SNE 등)로 시각화
```

> 오디오 없음.
> 대신 “텍스트 기반 의미 지도(Textual Semantic Map)”로 종별 구분·근접도 분석 가능.

---

### 🧭 **4️⃣ 조정 및 지식 기반 확장 (Knowledge-Guided Adjustment)**

**목적:**
앵커와 문헌 기반 지식(논문·사육 관찰 기록 등)을 연결해
**의미 벡터의 위치를 보정하거나 그룹화**.

**방법:**

* CLAP의 의미 벡터에 LLM을 통해 설명 문장 추가 → 확장된 앵커 세트 생성
* 동일 행동을 여러 문장으로 표현해 “언어적 다양성” 확보
* 의미 군집화로 종별 행동 패턴 차이 관찰

---

## 🚀 **이 방식의 장점**

| 항목       | CLAP 오디오 방식    | 현재(언어 지도 중심) 방식    |
| -------- | -------------- | ------------------ |
| 데이터 요구량  | 매우 높음          | 거의 없음              |
| 필요 자원    | 오디오, FFT 등 전처리 | 텍스트 기반             |
| 실험 속도    | 느림             | 매우 빠름              |
| 조정 난이도   | 높음             | 낮음                 |
| 주 목적 부합도 | 중간             | 매우 높음 (언어지도 중심 연구) |

---

## 📄 README에 들어갈 절차 요약 (최종본)

```markdown
## 🧭 Project Process Overview (Text-based Semantic Mapping)

1️⃣ **Anchor Definition**
   - 각 종의 행동 및 의미를 텍스트 문장(YAML)으로 정의.
   - 예: "rapid tail vibration as defense signal"

2️⃣ **Text Embedding (CLAP Language Space)**
   - CLAP의 언어 모듈을 사용해 각 앵커를 임베딩.
   - 결과: `artifacts/text_anchors/{species}.npy / .csv`

3️⃣ **Semantic Matching & Visualization**
   - 새로운 텍스트 입력(논문, 관찰 설명 등)을 같은 공간에 투영.
   - 코사인 유사도로 근접 앵커 탐색 및 UMAP 시각화.

4️⃣ **Knowledge-based Adjustment**
   - 관찰/논문 정보를 LLM이 해석 → 의미 기반 조정 및 확장.
   - 종별 행동 의미 지도 업데이트.

> ⚙️ 오디오 신호 단계는 제외됨.  
> 본 실험은 **CLAP의 언어 의미 공간을 직접 활용하는 텍스트 중심 의사소통 모델링**임.
```

---

1) 최상위 개요

tuc/ — 파이프라인 핵심 모듈들(모델 빌드, 조정, 검색, 입출력, 인코더 등) [CORE]

lib/ — 스크립트 모음 및 입력 임베딩 예제(예: Wav2Vec2) [SCRIPT]/[ADAPTER]

configs/ — 앵커 정의, 조정 규칙, 자연어 규칙, (선택) 페어링 정의 [CFG]

artifacts/ — 빌드/조정 후 산출물(임베딩, 리포트, 최근접 결과 등) [ARTIFACT]

src/ — 과거 래퍼/샘플(현행과 중복됨) [LEGACY]

_archive_not_used/ — 보류/미사용 코드 [LEGACY]

data/ — (선택) 정렬(Alignment)용 페어 CSV 등 학습 보조 데이터 [CFG/RAW]

2) tuc/ — 핵심 모듈
tuc/cli.py [CLI]

리포의 메인 엔트리. 서브커맨드:

build : 앵커 YAML → CLAP 텍스트 임베딩 생성

adjust: configs/adjust.yml 규칙 적용(전역/종별) 후 재정규화

query : 자연어 질의(문장) → Top-K 앵커 검색

run : build → adjust → query 순차 실행

(제안) infer : **실제 입력 벡터(.npy)**로 Top-K (아래 “PLANNED” 참조)

tuc/model_build.py [CORE]

역할: configs/anchors/*.yml을 읽어 앵커 텍스트를 CLAP으로 임베딩 → 저장.

입력: configs/anchors/<species>.yml (species, anchors: [{name,text,meaning,context}])

출력:

artifacts/text_anchors/{species}_text_anchors.npy — 앵커 임베딩

artifacts/text_anchors/{species}_text_anchors.csv — 메타(name/text/meaning/context/index)

artifacts/text_anchors/all_species_text_anchors_2d.csv — (시각화용) 전종 투영 테이블

비고: **앵커 YAML은 ‘의미 기준점 사전’**이므로 가능한 한 안정적으로 유지.

tuc/adjust.py [CORE]

역할: configs/adjust.yml(v2 스키마)을 읽어 전역/종별 조정을 적용.

지원 규칙:

global_adjust: center(평균 제거), whiten, dim_keep, temperature, dimension_weights

species_adjust: anchor_scales(개별 앵커 강/약), keyword_boosts(단어 기반 강조)

출력: 각 종의 .npy/.csv가 덮어쓰기로 갱신(최종 앵커 위치/가중이 반영됨).

tuc/search.py (파일명이 과거 serach.py였던 오타는 수정되어야 함) [CORE]

역할: 최근접 검색(자연어 질의 → 앵커 Top-K).

내부:

앵커 임베딩 스택/메타 로드 (tuc/io.py)

질의 문장 → tuc/encoder.encode_text() → 코사인 Top-K

출력: artifacts/text_anchors/nearest_overall_top{k}.csv

(제안/추가) nearest_from_vector(vec, k) : 실제 입력 벡터로 Top-K (아래 “PLANNED”)

tuc/encoder.py [CORE]

역할: CLAP(Text) 인코더 — encode_text(List[str]) -> np.ndarray[float32]

ClapProcessor, ClapModel 로드(전역 캐시), get_text_features() → L2 정규화.

체크포인트: laion/clap-htsat-fused (변경 가능)

tuc/io.py [CORE]

공용 I/O 유틸(경로 상수, YAML/CSV/NPY 로드/세이브, 앵커/메타 로딩 등).

ART = Path("artifacts/text_anchors") 등 기본 경로를 여기서 일괄 관리.

tuc/cli_input.py [ADAPTER/UTIL]

CLI 인자/입력 처리 헬퍼.

실제 임베딩 추출·검색 루틴과는 느슨하게 연결됨(확장 지점).

tuc/ingest.py [ADAPTER/UTIL]

폴더/파일 단위 입력을 읽어 전처리(간이 인제스트).

대규모 데이터 파이프라인의 초석(선택적으로 사용).

3) lib/ — 스크립트 & 입력 임베딩
lib/scripts/apply_rules.py [SCRIPT]

역할: 자연어 규칙(configs/rules/*_rules.yml)을 앵커 벡터에 직접 적용

positive_pairs: "문구" ↔ anchor 를 가깝게

negative_pairs: "문구" ↔ anchor 를 멀게

tie_groups / separate_groups: 동일/상이 그룹을 모으거나/벌리기

내부: 규칙의 text를 CLAP(Text)로 임베딩 → 해당 anchor 벡터를 소폭 이동(L2 정규화 유지)

출력: {species}_text_anchors.rules.npy (또는 --inplace 덮어쓰기)

lib/scripts/infer_behavior.py [SCRIPT]

역할: 샘플 추론(예: 텍스트/파일 입력을 받아 Top-K 확인)

현재: 텍스트 중심 예제. 실제 입력 벡터 직접 질의는 tuc/search.py에 보완 권장.

lib/scripts/embed_text_anchors.py [SCRIPT]

역할: tuc.cli build 를 감싼 래퍼. 앵커 임베딩 배치 생성.

lib/scripts/reembed_with_adjust.py [SCRIPT]

역할: 조정 후 재임베딩/리포팅 유틸(실험 편의).

lib/scripts/project_text_anchors.py [SCRIPT/REPORT]

역할: 전종 앵커 임베딩을 2D로 투영해 CSV/리포트를 만듦(시각화/QA용).

lib/scripts/embed_from_folder.py [SCRIPT]

역할: 폴더 단위 입력(오디오 등)을 읽어 배치 임베딩 생성(예제용).

lib/ops/features/base.py [ADAPTER]

역할: 입력 특징 추출의 베이스/프로토콜.

오디오 로딩, 샘플링 등 공통 유틸.

lib/ops/features/w2v2.py [ADAPTER]

역할: Wav2Vec2 기반 오디오 임베더 예제.

오디오 파일 → 고정길이 벡터(평균/Pool) 반환.

실제 센서/행동 임베더로 교체하는 접점.

4) configs/ — 설정/규칙
configs/anchors/ [CFG]

종별 앵커 정의: <species>.yml

species: cornsnake
anchors:
  - name: tail_vibration
    text: "Tail vibration as defensive warning"
    meaning: "Defensive display"
    context: "Threatened / cornered"


주의: name은 종 내부에서 유일해야 하며, 다른 규칙 파일·페어 파일에서 참조됨.

configs/adjust.yml [CFG]

전역/종별 후처리 조정(v2 스키마):

global_adjust: center/whiten/dim_keep/temperature/dimension_weights

species_adjust:

anchor_scales: 특정 앵커 가중 조정(>1 강화, <1 약화)

keyword_boosts: 텍스트/의미/컨텍스트에 특정 키워드 포함 시 가중

configs/rules/ [CFG]

자연어 규칙으로 앵커 벡터 위치를 직접 이동:

<species>_rules.yml

positive_pairs / negative_pairs / tie_groups / separate_groups

(선택) configs/linkages/ [CFG]

행동 ↔ 앵커 페어링(정렬용) YAML 버전:

species: cornsnake
pairs:
  - { behavior_id: clip_001, anchor: tail_vibration, weight: 1.0 }
behavior_embeddings:
  root: "artifacts/behaviors/cornsnake/"
  pattern: "{behavior_id}.npy"

5) artifacts/ — 산출물
artifacts/text_anchors/ [ARTIFACT]

{species}_text_anchors.npy — 앵커 임베딩 행렬

{species}_text_anchors.csv — 메타(index/name/text/meaning/context)

all_species_text_anchors_2d.csv — 전종 2D 투영(시각화/QA)

nearest_overall_top{k}.csv — 자연어 질의 Top-K 결과

{species}_text_anchors.rules.npy — 규칙 적용 결과(앵커 이동본)

(선택) behavior_to_text_W.npy — 정렬(Alignment) 가중치 행렬(행동→텍스트)

6) src/ / _archive_not_used/ — 레거시/보류
src/tuc/cli.py [LEGACY]

과거 CLI 래퍼(현행 tuc/cli.py와 중복).

실행 경로에서 제외하거나 legacy/ 이동 권장.

_archive_not_used/ [LEGACY]

보류/미사용 코드 샘플. 참고용.

정리 방법(권장):
legacy/ 폴더를 만들고 src/tuc/*, _archive_not_used/*를 이동 → 실행 경로 간소화.
(과거 코드 호환이 필요하면 src/tuc/__init__.py에 얇은 shim 추가)

7) (선택) data/ — 보조 데이터
data/pairs/ [CFG/RAW]

정렬(Alignment) 학습용 페어 CSV:

species,behavior_id,anchor,weight
cornsnake,clip_0001,tail_vibration,1.0


없다면 생략 가능. 행동 데이터 확보 후 도입.

8) 실행 흐름(요약)

앵커 임베딩 생성

python -m tuc.cli build


자연어 규칙 적용(앵커 이동) — 선택적이지만 현재 단계에서 핵심

python lib/scripts/apply_rules.py --rules configs/rules --alpha 0.15 --gamma 0.1 --out-suffix rules


후처리 조정(전역/종별)

python -m tuc.cli adjust


자연어 질의 검증

python -m tuc.cli query --query "rapid tail vibration" --k 5


(데이터 확보 후) 정렬(Alignment)

학습:

python lib/scripts/fit_alignment.py --pairs data/pairs/cornsnake_pairs.csv \
  --emb-root artifacts/behaviors --out artifacts/text_anchors/behavior_to_text_W.npy


적용/추론:

python lib/scripts/apply_alignment.py --input path/to/behavior.npy --W artifacts/text_anchors/behavior_to_text_W.npy


(제안/추가) CLI 통합: python -m tuc.cli infer --input path/to/vector.npy --k 5

9) 용어(이 프로젝트 맥락)

앵커(Anchor): 행동을 설명하는 기준 문장. 의미 공간의 기준점.

의미(Meaning): 사람이 이해하는 라벨/설명(메타데이터).

규칙(Rules): 자연어 문구로 앵커의 의미 방향을 끌어당기거나/밀어내는 조정.

조정(Adjust): 전역/종별로 가중치·정규화를 적용하는 후처리(공간 품질 안정화).

정렬(Alignment): 행동 임베딩 공간을 텍스트(앵커) 공간에 선형 변환으로 맞추는 단계.