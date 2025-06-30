# G-EVAL 방식 정책 평가 시스템

이 프로젝트는 **G-EVAL** 방식을 사용하여 정책을 정량적으로 평가하는 시스템입니다.

## G-EVAL이란?

G-EVAL은 LLM의 log probabilities를 활용한 평가 방법입니다:

1. **선택지 제시**: LLM에게 "A) 우수함, B) 보통, C) 미흡" 같은 선택지를 제시
2. **Log Probability 수집**: 각 선택지에 대한 LLM의 확신도(log probability) 추출
3. **점수 계산**: 확률 기반 가중평균으로 최종 점수 산출

### 기존 방식과의 차이점

| 방식 | Chain-of-Thoughts (CoT) | G-EVAL |
|------|-------------------------|---------|
| 점수 결정 | LLM이 직접 1-10점 부여 | Log probability 기반 자동 계산 |
| 신뢰도 | 주관적 | 객관적 확률 기반 |
| 일관성 | 모델에 따라 변동 | 수학적으로 일관된 계산 |

## 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

`.env` 파일을 만들고 OpenAI API 키를 설정:

```
OPENAI_API_KEY=your-openai-api-key-here
```

## 사용법

### 방법 1: 메인 스크립트 실행

```bash
python policy_evaluation_system.py
```

실행하면 평가 방법을 선택할 수 있습니다:
- 1번: Chain-of-Thoughts (기존 방식)
- 2번: G-EVAL (새로운 방식)

### 방법 2: G-EVAL 전용 스크립트

```bash
python run_geval_evaluation.py
```

G-EVAL 방식으로 바로 평가를 시작합니다.

### 방법 3: 프로그래밍 방식

```python
from policy_evaluation_system import PolicyEvaluationSystem

# 시스템 초기화
evaluator = PolicyEvaluationSystem(api_key="your-api-key")

# G-EVAL 평가 실행
evaluator.run_full_evaluation(
    gyeonggi_file="result_gyeonggi_policy.json",
    test_mode=True,
    evaluation_method="geval"
)
```

## G-EVAL 평가 기준

각 정책을 다음 4-5개 기준으로 평가합니다:

### 1. 효과성 (Effectiveness)
- **질문**: "이 정책이 청년들의 삶에 실질적이고 긍정적인 변화를 가져올 수 있을까요?"
- **평가**: A) 우수함 (8-10점) / B) 보통 (5-7점) / C) 미흡 (1-4점)

### 2. 실현가능성 (Feasibility)
- **질문**: "이 정책이 현실적으로 실행 가능하고 제도적 기반이 충분할까요?"
- **평가**: A) 우수함 / B) 보통 / C) 미흡

### 3. 혁신성 (Innovation)
- **질문**: "이 정책이 기존 정책과 차별화된 창의적이고 혁신적인 접근법을 제시할까요?"
- **평가**: A) 우수함 / B) 보통 / C) 미흡

### 4. 지속가능성 (Sustainability)
- **질문**: "이 정책이 장기적으로 안정적이고 지속 가능하게 운영될 수 있을까요?"
- **평가**: A) 우수함 / B) 보통 / C) 미흡

### 5. 예산효율성 (Budget Efficiency) *예산 정보가 있는 경우만*
- **질문**: "이 정책이 투입 예산 대비 높은 효과를 낼 수 있을까요?"
- **평가**: A) 우수함 / B) 보통 / C) 미흡

## G-EVAL 점수 계산 방식

1. **Log Probability 추출**: OpenAI API에서 각 선택지(A, B, C)의 log probability 수집
2. **확률 정규화**: 모든 선택지의 확률 합이 1이 되도록 정규화
3. **가중평균 계산**: 
   - A(우수함): 가중치 1.0
   - B(보통): 가중치 0.6
   - C(미흡): 가중치 0.2
4. **스케일 변환**: 0-1 범위의 점수를 0-10 스케일로 변환

### 수식

```
final_score = (P(A) × 1.0 + P(B) × 0.6 + P(C) × 0.2) × 10
```

여기서 P(A), P(B), P(C)는 정규화된 확률입니다.

## 결과 파일

G-EVAL 평가 결과는 JSON 형식으로 저장됩니다:

```json
{
  "evaluation_metadata": {
    "evaluation_method": "G-EVAL (Log Probability Based)",
    "scoring_method": "각 기준별 A/B/C 선택지에 대한 log probability 기반 점수 계산",
    "score_scale": "0-10점 (log probability 가중평균을 10점 스케일로 변환)"
  },
  "policies": [
    {
      "정책명": "청년 창업 지원",
      "점수": {
        "효과성": 7.2,
        "실현가능성": 8.1,
        "혁신성": 6.8,
        "지속가능성": 7.5,
        "종합점수": 7.4
      },
      "상세피드백": "각 기준별 분석 내용..."
    }
  ]
}
```

## 주요 특징

### 🎯 객관성
- Log probability 기반으로 주관적 편향 최소화
- 수학적으로 일관된 점수 계산

### 📊 투명성
- 각 선택지별 확률을 명시적으로 사용
- 점수 계산 과정이 완전히 투명

### 🔄 재현성
- 같은 정책에 대해 일관된 결과 제공
- Temperature=0.1로 설정하여 안정성 확보

### 🚀 효율성
- 기준별로 간단한 선택지 제시
- 복잡한 텍스트 생성 없이 빠른 평가

## 제한사항

1. **API 비용**: 기준별로 별도 API 호출이 필요하여 비용이 높을 수 있음
2. **모델 의존성**: Log probability 지원 모델에서만 사용 가능
3. **선택지 제약**: 3단계(A/B/C) 선택지로 제한

## 문제 해결

### Log Probability를 찾을 수 없는 경우
시스템이 자동으로 전통적인 방식으로 fallback합니다.

### API 호출 실패
- 인터넷 연결 상태 확인
- API 키가 올바른지 확인
- API 할당량 확인

## 라이선스

MIT License 