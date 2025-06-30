import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import openai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PolicyData:
    """정책 데이터 구조를 정의하는 클래스"""

    name: str
    description: str
    budget: Optional[float] = None
    category: Optional[str] = None
    region: Optional[str] = None
    year: Optional[int] = None
    target_group: Optional[str] = None


@dataclass
class EvaluationResult:
    """평가 결과를 저장하는 클래스"""

    policy_name: str
    effectiveness_score: float
    feasibility_score: float
    innovation_score: float
    sustainability_score: float
    overall_score: float
    detailed_feedback: str
    budget_efficiency: Optional[float] = None


class PolicyEvaluator(ABC):
    """정책 평가자의 추상 클래스"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    @abstractmethod
    def get_evaluation_prompt(self, policy: PolicyData) -> str:
        """평가 프롬프트를 생성하는 추상 메서드"""
        pass

    def evaluate_policy(self, policy: PolicyData) -> EvaluationResult:
        """단일 정책을 평가하는 메서드"""
        try:
            prompt = self.get_evaluation_prompt(policy)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 정책 분석 전문가입니다. 주어진 정책을 객관적이고 체계적으로 평가해주세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            return self._parse_evaluation_response(
                policy.name, response.choices[0].message.content
            )

        except Exception as e:
            logger.error(f"정책 평가 중 오류 발생: {e}")
            return self._create_default_result(policy.name)

    def _parse_evaluation_response(
        self, policy_name: str, response: str
    ) -> EvaluationResult:
        """LLM 응답을 파싱하여 EvaluationResult 객체로 변환"""
        try:
            # 먼저 JSON 형식으로 파싱 시도
            import json

            # 응답에서 JSON 부분만 추출
            response_clean = response.strip()

            # 만약 JSON이 ```로 감싸져 있다면 제거
            if response_clean.startswith("```"):
                lines = response_clean.split("\n")
                start_idx = 0
                end_idx = len(lines)

                for i, line in enumerate(lines):
                    if line.startswith("```") and i == 0:
                        start_idx = 1
                    elif line.startswith("```") and i > 0:
                        end_idx = i
                        break

                response_clean = "\n".join(lines[start_idx:end_idx])

            # JSON 파싱
            try:
                data = json.loads(response_clean)

                # Chain-of-Thoughts 응답에서 점수 추출
                scores = {}
                feedback_parts = []

                # 정책 이해
                if "정책_이해" in data:
                    feedback_parts.append(f"【정책 이해】\n{data['정책_이해']}")

                # 각 기준별 분석
                criteria_map = {
                    "효과성": ("효과성_분석", "효과성"),
                    "실현가능성": ("실현가능성_분석", "실현가능성"),
                    "혁신성": ("혁신성_분석", "혁신성"),
                    "지속가능성": ("지속가능성_분석", "지속가능성"),
                    "예산효율성": ("예산효율성_분석", "예산효율성"),
                }

                for criterion, (analysis_key, score_key) in criteria_map.items():
                    if analysis_key in data:
                        feedback_parts.append(
                            f"【{criterion} 분석】\n{data[analysis_key]}"
                        )
                    if score_key in data:
                        scores[criterion] = float(data[score_key])

                # 개선방안
                if "개선방안" in data:
                    feedback_parts.append(f"【개선방안】\n{data['개선방안']}")

                # 상세피드백
                if "상세피드백" in data:
                    feedback_parts.append(f"【상세피드백】\n{data['상세피드백']}")

                feedback = "\n\n".join(feedback_parts)

                # 점수 추출
                effectiveness_score = scores.get("효과성", 5.0)
                feasibility_score = scores.get("실현가능성", 5.0)
                innovation_score = scores.get("혁신성", 5.0)
                sustainability_score = scores.get("지속가능성", 5.0)
                budget_efficiency = scores.get("예산효율성")

                # 종합점수 자동 계산
                scores_for_average = [
                    effectiveness_score,
                    feasibility_score,
                    innovation_score,
                    sustainability_score,
                ]

                if budget_efficiency is not None:
                    scores_for_average.append(budget_efficiency)

                calculated_overall_score = round(
                    sum(scores_for_average) / len(scores_for_average), 1
                )

                return EvaluationResult(
                    policy_name=policy_name,
                    effectiveness_score=effectiveness_score,
                    feasibility_score=feasibility_score,
                    innovation_score=innovation_score,
                    sustainability_score=sustainability_score,
                    overall_score=calculated_overall_score,
                    budget_efficiency=budget_efficiency,
                    detailed_feedback=(
                        feedback if feedback else "평가 중 오류가 발생했습니다."
                    ),
                )

            except Exception as e2:
                logger.error(f"텍스트 파싱도 실패: {str(e2)}")
                return self._create_default_result(policy_name)

        except Exception as e:
            logger.error(f"응답 파싱 중 예상치 못한 오류: {str(e)}")
            return self._create_default_result(policy_name)

    def _create_default_result(self, policy_name: str) -> EvaluationResult:
        """오류 발생 시 기본 결과 반환"""
        return EvaluationResult(
            policy_name=policy_name,
            effectiveness_score=5.0,
            feasibility_score=5.0,
            innovation_score=5.0,
            sustainability_score=5.0,
            overall_score=5.0,
            detailed_feedback="평가 중 오류가 발생했습니다.",
        )


class GEvalPolicyEvaluator(PolicyEvaluator):
    """G-EVAL 방식을 사용한 정책 평가자"""

    def evaluate_policy(self, policy: PolicyData) -> EvaluationResult:
        """G-EVAL 방식으로 정책을 평가하는 메서드"""
        try:
            # 각 평가 기준별로 G-EVAL 방식으로 점수 계산
            criteria_scores = {}
            detailed_analysis = []

            # 평가 기준 정의
            criteria = {
                "효과성": "이 정책이 청년들의 삶에 실질적이고 긍정적인 변화를 가져올 수 있을까요?",
                "실현가능성": "이 정책이 현실적으로 실행 가능하고 제도적 기반이 충분할까요?",
                "혁신성": "이 정책이 기존 정책과 차별화된 창의적이고 혁신적인 접근법을 제시할까요?",
                "지속가능성": "이 정책이 장기적으로 안정적이고 지속 가능하게 운영될 수 있을까요?",
            }

            # 예산 정보가 있는 경우 예산효율성도 추가
            if policy.budget is not None:
                criteria["예산효율성"] = (
                    "이 정책이 투입 예산 대비 높은 효과를 낼 수 있을까요?"
                )

            for criterion, question in criteria.items():
                score, analysis = self._evaluate_criterion_with_geval(
                    policy, criterion, question
                )
                criteria_scores[criterion] = score
                detailed_analysis.append(f"【{criterion} 분석】\n{analysis}")

            # 종합점수 계산
            overall_score = round(
                sum(criteria_scores.values()) / len(criteria_scores), 1
            )

            # 상세 피드백 구성
            score_breakdown = ", ".join(
                [f"{k}({v:.1f})" for k, v in criteria_scores.items()]
            )
            detailed_analysis.append(
                f"【종합점수 자동계산】\n{score_breakdown}의 평균 = {overall_score}점"
            )

            combined_feedback = "\n\n".join(detailed_analysis)

            return EvaluationResult(
                policy_name=policy.name,
                effectiveness_score=criteria_scores.get("효과성", None),
                feasibility_score=criteria_scores.get("실현가능성", None),
                innovation_score=criteria_scores.get("혁신성", None),
                sustainability_score=criteria_scores.get("지속가능성", None),
                overall_score=overall_score,
                budget_efficiency=criteria_scores.get("예산효율성"),
                detailed_feedback=combined_feedback,
            )

        except Exception as e:
            logger.error(f"G-EVAL 정책 평가 중 오류 발생: {e}")
            return self._create_default_result(policy.name)

    def _evaluate_criterion_with_geval(
        self, policy: PolicyData, criterion: str, question: str
    ) -> tuple[float, str]:
        """진정한 G-EVAL Form-filling 방식으로 단일 기준을 평가"""
        try:
            # Form-filling Paradigm에 따른 프롬프트 구성
            criteria_details = {
                "효과성": "청년의 고용, 주거, 교육, 건강 등 삶의 질 개선에 직접적으로 기여하는가? 수혜 대상이 충분히 넓은가? 정책 효과가 구체적으로 측정 가능한가?",
                "실현가능성": "기존 제도와의 충돌 여부, 법적·제도적 기반의 유무, 행정 집행 가능성과 주체의 명확성, 이해관계자 수용성",
                "혁신성": "기존 정책과의 유사성 여부, 디지털 기술·데이터 기반·청년 참여 기반 등 혁신적 요소 유무, 기존 실패 사례 보완 여부",
                "지속가능성": "예산, 인력, 제도적 기반의 지속 가능성, 사회 변화에 영향을 적게 받는 안정성, 제도화 가능성",
                "예산효율성": "예산 규모의 적절성, 비용 대비 기대 효과, 효율적 자원 집행 가능성",
            }

            detail_desc = criteria_details.get(
                criterion, "해당 기준에 대한 종합적 평가"
            )

            # G-EVAL Form-filling 프롬프트 (CoT + 단답형)
            prompt = f"""
**평가 기준 설명**: 
{criterion}: {detail_desc}

**입력 정책**:
• 정책명: {policy.name}
• 정책 내용: {policy.description}
• 카테고리: {policy.category}
• 지역: {policy.region}
• 연도: {policy.year}
{f"• 예산: {policy.budget}백만원" if policy.budget else ""}

**Chain-of-Thought 평가 과정**:
1. 정책의 핵심 특징 파악
2. {criterion} 기준에 따른 장단점 분석
3. 구체적 근거와 사례 제시
4. 개선 가능성 검토

**평가 절차**:
위 과정을 거쳐 1점부터 10점까지 중에서 가장 적절한 점수를 선택하세요.
- 1-3점: 매우 부족
- 4-6점: 보통/개선 필요
- 7-8점: 양호
- 9-10점: 우수

Chain-of-Thought 분석을 충분히 한 후, 마지막에 반드시 다음 형식으로 끝내세요:

**최종 점수: [1-10 중 하나의 숫자]**
"""

            # logprobs를 포함한 API 호출
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"당신은 청년 정책의 {criterion}을 전문적으로 평가하는 정책 분석가입니다. 체계적인 분석과 정확한 점수 부여를 해주세요.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
                logprobs=True,
                top_logprobs=20,  # 1-10 점수를 모두 볼 수 있도록 확장
            )

            content = response.choices[0].message.content

            # 분석 내용과 점수 추출
            if "**최종 점수:" in content:
                analysis = content.split("**최종 점수:")[0].strip()
                score_part = content.split("**최종 점수:")[1].strip().replace("**", "")
            else:
                analysis = content
                score_part = "None"  # 기본값

            # log probabilities에서 1-10 점수별 확률 추출
            logprobs_data = response.choices[0].logprobs
            if logprobs_data and logprobs_data.content:
                # 1-10 점수에 대한 확률 분포 수집
                score_probs = self._extract_score_probabilities(logprobs_data.content)

                if score_probs:
                    # G-EVAL 수식: score = Σ p(si) * si
                    final_score = self._calculate_weighted_score(score_probs)

                    # 확률 분포 정보를 분석에 추가
                    prob_info = ", ".join(
                        [f"{s}점({p:.3f})" for s, p in sorted(score_probs.items())]
                    )
                    analysis += f"\n\n【G-EVAL 확률 분포】\n{prob_info}\n가중평균 점수: {final_score:.2f}"
                else:
                    # 확률 추출 실패시 텍스트에서 점수 파싱
                    final_score = self._parse_score_from_text(score_part)
            else:
                final_score = self._parse_score_from_text(score_part)

            return final_score, analysis

        except Exception as e:
            logger.error(f"{criterion} G-EVAL 평가 중 오류: {e}")
            return 5.0, f"{criterion} 평가 중 오류가 발생했습니다."

    def _extract_score_probabilities(self, logprobs_content) -> Dict[int, float]:
        """log probabilities에서 1-10 점수별 확률 추출 (G-EVAL Form-filling)"""
        score_probs = {}

        # 전체 토큰에서 숫자 토큰들 찾기
        for token_data in logprobs_content[-20:]:  # 마지막 20개 토큰 확인
            if token_data.token and token_data.top_logprobs:
                for top_logprob in token_data.top_logprobs:
                    token = top_logprob.token.strip()

                    # 1-10 사이의 숫자인지 확인
                    try:
                        score_num = int(token)
                        if 1 <= score_num <= 10:
                            prob = math.exp(top_logprob.logprob)
                            # 같은 점수가 여러 번 나타나면 최고 확률로 업데이트
                            if (
                                score_num not in score_probs
                                or prob > score_probs[score_num]
                            ):
                                score_probs[score_num] = prob
                    except (ValueError, TypeError):
                        continue

        return score_probs

    def _calculate_weighted_score(self, score_probs: Dict[int, float]) -> float:
        """G-EVAL 수식: score = Σ p(si) * si 계산"""
        if not score_probs:
            return 5.0

        # 확률 정규화
        total_prob = sum(score_probs.values())
        if total_prob == 0:
            return 5.0

        # 가중평균 계산: score = Σ p(si) * si
        weighted_sum = 0
        for score, prob in score_probs.items():
            normalized_prob = prob / total_prob
            weighted_sum += score * normalized_prob

        return round(weighted_sum, 2)

    def _parse_score_from_text(self, score_text: str) -> float:
        """텍스트에서 점수 파싱 (fallback)"""
        import re

        # 숫자 패턴 찾기
        numbers = re.findall(r"\b([1-9]|10)\b", score_text.strip())
        if numbers:
            try:
                score = int(numbers[0])
                return float(score) if 1 <= score <= 10 else 5.0
            except (ValueError, TypeError):
                pass

        return 5.0  # 기본값

    def get_evaluation_prompt(self, policy: PolicyData) -> str:
        """G-EVAL은 직접 프롬프트를 사용하지 않고 개별 기준별로 평가"""
        return "G-EVAL 방식으로 평가합니다."


class BudgetIncludedEvaluator(PolicyEvaluator):
    """예산 정보를 포함한 정책 평가자"""

    def get_evaluation_prompt(self, policy: PolicyData) -> str:
        prompt = f"""
다음 경기도 청년 정책을 평가하고 반드시 아래 JSON 형식으로만 응답해주세요:

정책명: {policy.name}
정책 내용: {policy.description}
예산: {policy.budget}백만원
카테고리: {policy.category}
지역: {policy.region}
연도: {policy.year}

청년 정책 정량 평가 기준 (각 항목 1~10점 척도):

1. 효과성 (Effectiveness)
정책이 청년의 삶에 실질적인 긍정적 변화를 유도할 수 있는 정도를 평가한다.
- 청년의 고용, 주거, 교육, 건강 등 삶의 질 개선에 직접적으로 기여하는가?
- 수혜 대상이 충분히 넓은가?
- 정책 효과가 구체적으로 측정 가능한가?
- 1~3점: 대상이 협소하고 효과가 모호함
- 4~6점: 일부 긍정적 효과 기대 가능, 일부 측정 지표 존재
- 7~10점: 명확하고 광범위한 효과 기대 가능, 측정 가능성 높음

2. 실현가능성 (Feasibility)
정책이 법적, 제도적, 행정적으로 현실에서 실행 가능한지를 평가한다.
- 기존 제도와의 충돌 여부
- 법적·제도적 기반의 유무
- 행정 집행 가능성과 주체의 명확성
- 이해관계자 수용성
- 1~3점: 제약 많고 실행 불가능에 가까움
- 4~6점: 실행 가능하나 일부 제약 있음
- 7~10점: 법적·행정적으로 충분히 가능, 주체와 절차가 명확함

3. 혁신성 (Innovativeness)
정책의 창의성과 기존 정책과의 차별성, 새로운 접근법 도입 정도를 평가한다.
- 기존 정책과의 유사성 여부
- 디지털 기술, 데이터 기반, 청년 참여 기반 등 혁신적 요소 유무
- 기존 실패 사례 보완 여부
- 1~3점: 기존 정책의 반복, 창의성 없음
- 4~6점: 일부 차별적 요소 있음
- 7~10점: 새로운 문제 접근법, 창의적 설계 다수 포함

4. 지속가능성 (Sustainability)
정책이 단기 이벤트가 아닌 중장기적으로 안정적이고 지속 가능한지를 평가한다.
- 예산, 인력, 제도적 기반의 지속 가능성
- 사회 변화에 영향을 적게 받는 안정성
- 제도화 가능성
- 1~3점: 일회성, 중단 가능성 높음
- 4~6점: 중기 지속 가능하나 제도화 미흡
- 7~10점: 장기적 운영 가능, 제도화 기반 확실함

5. 예산효율성 (Cost-effectiveness)
정책이 투입된 예산 대비 얼마나 효과적으로 성과를 낼 수 있는지를 평가한다.
- 수혜 인원 대비 예산 적정성
- 자원 낭비 여부
- 다른 대안 대비 효율성
- 1~3점: 고비용 대비 낮은 효과, 비효율적 구조
- 4~6점: 일정 수준의 성과 있으나 예산 구조 개선 필요
- 7~10점: 낮은 비용으로 높은 효과, 효율적 자원 집행 가능

⚠️ 중요: Chain-of-Thoughts 방식으로 단계별 사고 과정을 포함하여 반드시 아래 JSON 형식으로만 응답하세요.

**주의:** 종합점수는 시스템에서 자동으로 각 영역 점수의 평균으로 계산되므로, 종합점수는 입력하지 마세요.

**평가 절차:**
1. 먼저 정책을 종합적으로 이해하고 핵심 특징을 파악
2. 각 평가 기준별로 단계적 분석 수행
3. 분석 근거를 바탕으로 객관적 점수 산정
4. 종합적 평가 및 개선 방안 제시

{{
  "정책_이해": "정책의 핵심 목적, 대상, 방법에 대한 종합적 이해",
  "효과성_분석": "효과성 평가를 위한 단계별 사고 과정",
  "효과성": 1-10 사이의 점수,
  "실현가능성_분석": "실현가능성 평가를 위한 단계별 사고 과정", 
  "실현가능성": 1-10 사이의 점수,
  "혁신성_분석": "혁신성 평가를 위한 단계별 사고 과정",
  "혁신성": 1-10 사이의 점수,
  "지속가능성_분석": "지속가능성 평가를 위한 단계별 사고 과정",
  "지속가능성": 1-10 사이의 점수,
  "예산효율성_분석": "예산효율성 평가를 위한 단계별 사고 과정",
  "예산효율성": 1-10 사이의 점수,
  "개선방안": "정책 개선을 위한 구체적 제안사항",
  "상세피드백": "평가 결과 종합 및 최종 의견 (위 각 영역 점수들을 종합적으로 고려)"
}}
"""
        return prompt


class BudgetExcludedEvaluator(PolicyEvaluator):
    """예산 정보를 제외한 정책 평가자"""

    def get_evaluation_prompt(self, policy: PolicyData) -> str:
        prompt = f"""
다음 경기도 청년 정책을 평가하고 반드시 아래 JSON 형식으로만 응답해주세요:

정책명: {policy.name}
정책 내용: {policy.description}
카테고리: {policy.category}
지역: {policy.region}
연도: {policy.year}

청년 정책 정량 평가 기준 (각 항목 1~10점 척도):

1. 효과성 (Effectiveness)
정책이 청년의 삶에 실질적인 긍정적 변화를 유도할 수 있는 정도를 평가한다.
- 청년의 고용, 주거, 교육, 건강 등 삶의 질 개선에 직접적으로 기여하는가?
- 수혜 대상이 충분히 넓은가?
- 정책 효과가 구체적으로 측정 가능한가?
- 1~3점: 대상이 협소하고 효과가 모호함
- 4~6점: 일부 긍정적 효과 기대 가능, 일부 측정 지표 존재
- 7~10점: 명확하고 광범위한 효과 기대 가능, 측정 가능성 높음

2. 실현가능성 (Feasibility)
정책이 법적, 제도적, 행정적으로 현실에서 실행 가능한지를 평가한다.
- 기존 제도와의 충돌 여부
- 법적·제도적 기반의 유무
- 행정 집행 가능성과 주체의 명확성
- 이해관계자 수용성
- 1~3점: 제약 많고 실행 불가능에 가까움
- 4~6점: 실행 가능하나 일부 제약 있음
- 7~10점: 법적·행정적으로 충분히 가능, 주체와 절차가 명확함

3. 혁신성 (Innovativeness)
정책의 창의성과 기존 정책과의 차별성, 새로운 접근법 도입 정도를 평가한다.
- 기존 정책과의 유사성 여부
- 디지털 기술, 데이터 기반, 청년 참여 기반 등 혁신적 요소 유무
- 기존 실패 사례 보완 여부
- 1~3점: 기존 정책의 반복, 창의성 없음
- 4~6점: 일부 차별적 요소 있음
- 7~10점: 새로운 문제 접근법, 창의적 설계 다수 포함

4. 지속가능성 (Sustainability)
정책이 단기 이벤트가 아닌 중장기적으로 안정적이고 지속 가능한지를 평가한다.
- 예산, 인력, 제도적 기반의 지속 가능성
- 사회 변화에 영향을 적게 받는 안정성
- 제도화 가능성
- 1~3점: 일회성, 중단 가능성 높음
- 4~6점: 중기 지속 가능하나 제도화 미흡
- 7~10점: 장기적 운영 가능, 제도화 기반 확실함

⚠️ 중요: Chain-of-Thoughts 방식으로 단계별 사고 과정을 포함하여 반드시 아래 JSON 형식으로만 응답하세요.

**주의:** 종합점수는 시스템에서 자동으로 각 영역 점수의 평균으로 계산되므로, 종합점수는 입력하지 마세요.

**평가 절차:**
1. 먼저 정책을 종합적으로 이해하고 핵심 특징을 파악
2. 각 평가 기준별로 단계적 분석 수행
3. 분석 근거를 바탕으로 객관적 점수 산정
4. 종합적 평가 및 개선 방안 제시

{{
  "정책_이해": "정책의 핵심 목적, 대상, 방법에 대한 종합적 이해",
  "효과성_분석": "효과성 평가를 위한 단계별 사고 과정",
  "효과성": 1-10 사이의 점수,
  "실현가능성_분석": "실현가능성 평가를 위한 단계별 사고 과정",
  "실현가능성": 1-10 사이의 점수,
  "혁신성_분석": "혁신성 평가를 위한 단계별 사고 과정", 
  "혁신성": 1-10 사이의 점수,
  "지속가능성_분석": "지속가능성 평가를 위한 단계별 사고 과정",
  "지속가능성": 1-10 사이의 점수,
  "개선방안": "정책 개선을 위한 구체적 제안사항",
  "상세피드백": "평가 결과 종합 및 최종 의견 (위 각 영역 점수들을 종합적으로 고려)"
}}
"""
        return prompt


class PolicyDataLoader:
    """정책 데이터 로더"""

    @staticmethod
    def load_gyeonggi_data(file_path: str) -> List[PolicyData]:
        """경기도 정책 데이터 로드 (예산 정보 포함)"""
        policies = []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for region, region_data in data.items():
            year = region_data.get("년도", None)
            policy_categories = region_data.get("정책수행", {})

            for category, category_data in policy_categories.items():
                if category == "합계":
                    continue

                for project in category_data.get("세부사업", []):
                    policy = PolicyData(
                        name=project.get("사업명", ""),
                        description=project.get("주요내용", ""),
                        budget=project.get("예산"),
                        category=category,
                        region=region,
                        year=year,
                        target_group="청년",
                    )
                    policies.append(policy)

        return policies


class PolicyEvaluationSystem:
    """정책 평가 시스템 메인 클래스"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        self.budget_included_evaluator = BudgetIncludedEvaluator(api_key, model)
        self.budget_excluded_evaluator = BudgetExcludedEvaluator(api_key, model)
        self.geval_evaluator = GEvalPolicyEvaluator(api_key, model)
        self.data_loader = PolicyDataLoader()

    def evaluate_policies_with_budget(
        self, policies: List[PolicyData]
    ) -> List[EvaluationResult]:
        """예산 정보를 포함한 정책 평가"""
        results = []

        for i, policy in enumerate(policies):
            logger.info(
                f"예산 포함 평가 진행 중: {i+1}/{len(policies)} - {policy.name}"
            )

            if policy.budget is not None:
                result = self.budget_included_evaluator.evaluate_policy(policy)
                results.append(result)

                # API 호출 제한을 위한 대기
                time.sleep(1)

        return results

    def evaluate_policies_without_budget(
        self, policies: List[PolicyData]
    ) -> List[EvaluationResult]:
        """예산 정보를 제외한 정책 평가"""
        results = []

        for i, policy in enumerate(policies):
            logger.info(
                f"예산 제외 평가 진행 중: {i+1}/{len(policies)} - {policy.name}"
            )

            result = self.budget_excluded_evaluator.evaluate_policy(policy)
            results.append(result)

            # API 호출 제한을 위한 대기
            time.sleep(1)

        return results

    def save_results_to_json(self, results: List[EvaluationResult], filename: str):
        """평가 결과를 JSON으로 저장"""
        data = {
            "evaluation_metadata": {
                "total_policies": len(results),
                "model_used": getattr(
                    self.budget_included_evaluator, "model", "unknown"
                ),
            },
            "policies": [],
        }

        for result in results:
            policy_data = {
                "정책명": result.policy_name,
                "점수": {
                    "효과성": result.effectiveness_score,
                    "실현가능성": result.feasibility_score,
                    "혁신성": result.innovation_score,
                    "지속가능성": result.sustainability_score,
                    "종합점수": result.overall_score,
                },
                "상세피드백": result.detailed_feedback,
            }

            # 예산효율성이 있는 경우에만 추가
            if result.budget_efficiency is not None:
                policy_data["점수"]["예산효율성"] = result.budget_efficiency

            data["policies"].append(policy_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"평가 결과가 {filename}에 저장되었습니다.")

    def save_results_to_json_geval(
        self, results: List[EvaluationResult], filename: str
    ):
        """G-EVAL 평가 결과를 JSON으로 저장"""
        data = {
            "evaluation_metadata": {
                "total_policies": len(results),
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": self.model,
                "evaluation_method": "G-EVAL Form-filling Paradigm",
                "scoring_method": "score = Σ p(si) * si (각 점수별 log probability 가중합)",
                "score_scale": "1-10점 (Chain-of-Thought 분석 후 확률 기반 가중평균)",
                "paradigm": "평가 기준 설명 + CoT + 입력 문맥 + 출력 결과(단답)",
            },
            "policies": [],
        }

        for result in results:
            policy_data = {
                "정책명": result.policy_name,
                "점수": {
                    "효과성": result.effectiveness_score,
                    "실현가능성": result.feasibility_score,
                    "혁신성": result.innovation_score,
                    "지속가능성": result.sustainability_score,
                    "종합점수": result.overall_score,
                },
                "상세피드백": result.detailed_feedback,
            }

            # 예산효율성이 있는 경우에만 추가
            if result.budget_efficiency is not None:
                policy_data["점수"]["예산효율성"] = result.budget_efficiency

            data["policies"].append(policy_data)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"G-EVAL 평가 결과가 {filename}에 저장되었습니다.")

    def evaluate_policies_with_geval(
        self, policies: List[PolicyData]
    ) -> List[EvaluationResult]:
        """G-EVAL 방식으로 정책 평가"""
        results = []

        for i, policy in enumerate(policies):
            logger.info(f"G-EVAL 평가 진행 중: {i+1}/{len(policies)} - {policy.name}")

            result = self.geval_evaluator.evaluate_policy(policy)
            results.append(result)

            # API 호출 제한을 위한 대기
            time.sleep(3)  # G-EVAL은 더 많은 API 호출을 하므로 조금 더 긴 대기

        return results

    def run_full_evaluation(
        self,
        gyeonggi_file: str,
        output_dir: str = "evaluation_results",
        test_mode: bool = False,
        evaluation_method: str = "cot",
    ):
        """전체 평가 프로세스 실행"""

        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 경기도 데이터 평가
        if os.path.exists(gyeonggi_file):
            logger.info("경기도 정책 데이터 로드 중...")
            gyeonggi_policies = self.data_loader.load_gyeonggi_data(gyeonggi_file)
            logger.info(f"로드된 정책 수: {len(gyeonggi_policies)}")

            # 테스트 모드일 경우 5개만 처리, 아니면 전체 처리
            if test_mode:
                policies_to_evaluate = gyeonggi_policies[:5]
                logger.info("테스트 모드: 처음 5개 정책만 평가합니다.")
            else:
                policies_to_evaluate = gyeonggi_policies
                logger.info(
                    f"전체 모드: {len(policies_to_evaluate)}개 정책을 평가합니다."
                )

            if evaluation_method.lower() == "geval":
                # G-EVAL 방식 평가
                logger.info("G-EVAL 방식으로 정책 평가 시작...")
                geval_results = self.evaluate_policies_with_geval(policies_to_evaluate)

                # 결과 저장
                geval_filename = "gyeonggi_evaluation_geval.json"
                if test_mode:
                    geval_filename = "test_" + geval_filename

                self.save_results_to_json_geval(
                    geval_results, os.path.join(output_dir, geval_filename)
                )

                logger.info("G-EVAL 평가 완료!")

            else:
                # 기존 Chain-of-Thoughts 방식 평가
                # 예산 포함 평가
                logger.info("경기도 정책 평가 시작 (예산 포함)...")
                gyeonggi_budget_results = self.evaluate_policies_with_budget(
                    policies_to_evaluate
                )

                # 예산 제외 평가
                logger.info("경기도 정책 평가 시작 (예산 제외)...")
                gyeonggi_no_budget_results = self.evaluate_policies_without_budget(
                    policies_to_evaluate
                )

                # 결과 저장
                budget_filename = "gyeonggi_evaluation_with_budget.json"
                no_budget_filename = "gyeonggi_evaluation_without_budget.json"

                if test_mode:
                    budget_filename = "test_" + budget_filename
                    no_budget_filename = "test_" + no_budget_filename

                self.save_results_to_json(
                    gyeonggi_budget_results,
                    os.path.join(output_dir, budget_filename),
                )
                self.save_results_to_json(
                    gyeonggi_no_budget_results,
                    os.path.join(output_dir, no_budget_filename),
                )

                logger.info("Chain-of-Thoughts 평가 완료!")

        else:
            logger.error(f"파일을 찾을 수 없습니다: {gyeonggi_file}")


def main():
    """메인 실행 함수"""

    # API 키 설정 (환경변수 또는 직접 입력)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("OpenAI API 키를 입력하세요: ")

    # 평가 방법 선택
    print("평가 방법을 선택하세요:")
    print("1. Chain-of-Thoughts (CoT) - 기존 방식")
    print("2. G-EVAL - Log Probability 기반")

    choice = input("선택 (1 또는 2): ").strip()
    evaluation_method = "geval" if choice == "2" else "cot"

    method_name = "G-EVAL" if evaluation_method == "geval" else "Chain-of-Thoughts"
    print(f"\n{method_name} 방식으로 평가를 진행합니다.")

    # 평가 시스템 초기화
    evaluation_system = PolicyEvaluationSystem(api_key)

    # 데이터 파일 경로
    gyeonggi_file = "result_gyeonggi_policy.json"

    # 전체 평가 실행
    evaluation_system.run_full_evaluation(
        gyeonggi_file=gyeonggi_file,
        test_mode=False,  # 전체 모드로 실행 (모든 정책)
        evaluation_method=evaluation_method,
    )


if __name__ == "__main__":
    main()
