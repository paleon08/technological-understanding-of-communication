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

원하면 위 내용을 기반으로
`README.md`에 들어갈 **최종 절차도 (mermaid 다이어그램)** 도 만들어 줄게.
텍스트 기반 파이프라인으로 딱 맞춘 형태로.
그렇게 해줄까?
