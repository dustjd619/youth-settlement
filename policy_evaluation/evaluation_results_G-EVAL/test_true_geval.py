#!/usr/bin/env python3
"""
G-EVAL Form-filling Paradigm 테스트 스크립트

G-EVAL 수식을 기반으로:
score = Σ p(si) * si
- 평가 기준 설명 + Chain-of-Thought + 입력 문맥 + 출력 결과(단답)
- 1-10점 범위에서 log probability 기반 가중평균 계산
"""

import json
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from policy_evaluation_system import (
    GEvalPolicyEvaluator,
    PolicyData,
    PolicyEvaluationSystem,
)


def save_single_test_results(policy, results, detailed_analyses, overall_score):
    """단일 정책 테스트 결과를 JSON 파일로 저장"""

    # 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_geval_test_{timestamp}.json"

    # 저장할 데이터 구조
    save_data = {
        "test_metadata": {
            "test_type": "Single Policy G-EVAL Test",
            "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_method": "G-EVAL Form-filling Paradigm",
            "scoring_formula": "score = Σ p(si) * si",
            "model_used": "gpt-4o-mini",
        },
        "policy_info": {
            "정책명": policy.name,
            "정책내용": policy.description,
            "예산": f"{policy.budget}백만원" if policy.budget else "N/A",
            "카테고리": policy.category,
            "지역": policy.region,
            "연도": policy.year,
            "대상그룹": policy.target_group,
        },
        "evaluation_results": {
            "점수": {
                criterion: round(score, 2) for criterion, score in results.items()
            },
            "종합점수": overall_score,
            "상세분석": detailed_analyses,
        },
        "summary": {
            "최고점수": f"{max(results.keys(), key=results.get)} ({max(results.values()):.2f}점)",
            "최저점수": f"{min(results.keys(), key=results.get)} ({min(results.values()):.2f}점)",
            "점수범위": f"{min(results.values()):.2f} ~ {max(results.values()):.2f}",
            "표준편차": round(
                (sum((x - overall_score) ** 2 for x in results.values()) / len(results))
                ** 0.5,
                2,
            ),
        },
    }

    # 파일 저장
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 결과가 저장되었습니다: {filename}")
        print(f"📁 파일 크기: {os.path.getsize(filename)} bytes")

    except Exception as e:
        print(f"❌ 파일 저장 중 오류: {e}")


def test_single_policy():
    """단일 정책으로 G-EVAL 테스트"""

    # 환경변수 로드
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 필요합니다.")
        return

    # 테스트용 정책 데이터
    test_policy = PolicyData(
        name="청년 창업 지원 프로그램",
        description="만 18~39세 청년을 대상으로 창업 아이디어 개발부터 사업화까지 단계별 맞춤형 지원을 제공하는 프로그램. 창업교육, 멘토링, 자금지원, 공간제공 등을 통해 청년 창업 생태계를 조성한다.",
        budget=150.0,
        category="창업지원",
        region="경기도",
        year=2024,
        target_group="청년",
    )

    print("🧪 G-EVAL Form-filling Paradigm 테스트")
    print("=" * 60)
    print(f"테스트 정책: {test_policy.name}")
    print(f"설명: {test_policy.description[:50]}...")
    print(f"예산: {test_policy.budget}백만원")
    print("=" * 60)

    # G-EVAL 평가자 초기화
    evaluator = GEvalPolicyEvaluator(api_key, model="gpt-4o-mini")

    # 각 기준별로 개별 테스트
    criteria = ["효과성", "실현가능성", "혁신성", "지속가능성", "예산효율성"]
    results = {}
    detailed_analyses = {}

    for i, criterion in enumerate(criteria, 1):
        print(f"\n📊 [{i}/{len(criteria)}] {criterion} 평가 중...")

        try:
            score, analysis = evaluator._evaluate_criterion_with_geval(
                test_policy, criterion, f"{criterion} 기준으로 평가"
            )

            results[criterion] = score
            detailed_analyses[criterion] = analysis

            print(f"✅ {criterion} 점수: {score:.2f}점")

            # 확률 분포 정보가 있다면 출력
            if "【G-EVAL 확률 분포】" in analysis:
                prob_section = analysis.split("【G-EVAL 확률 분포】")[1].split("\n")[1]
                print(f"📈 확률 분포: {prob_section}")

            print("-" * 40)

        except Exception as e:
            print(f"❌ {criterion} 평가 중 오류: {e}")
            results[criterion] = 5.0
            detailed_analyses[criterion] = (
                f"{criterion} 평가 중 오류가 발생했습니다: {e}"
            )

        # API 호출 제한
        if i < len(criteria):
            time.sleep(2)

    # 종합점수 계산
    overall_score = round(sum(results.values()) / len(results), 2)

    print("\n🎯 G-EVAL Form-filling Paradigm 테스트 완료!")
    print(f"📊 종합점수: {overall_score}점")
    print("\n📋 특징:")
    print("• Form-filling: 평가 기준 설명 + CoT + 단답형 점수")
    print("• 수식: score = Σ p(si) * si")
    print("• 범위: 1-10점 log probability 가중평균")
    print("• 장점: LLM 불확실성 반영, 연속적 점수 분포")

    # 결과를 파일로 저장
    save_single_test_results(test_policy, results, detailed_analyses, overall_score)


def test_full_evaluation():
    """전체 평가 시스템으로 테스트 (소수 정책)"""

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 필요합니다.")
        return

    print("\n🚀 전체 G-EVAL 시스템 테스트")
    print("=" * 60)

    # 평가 시스템 초기화
    system = PolicyEvaluationSystem(api_key, model="gpt-4o-mini")

    # 전체 G-EVAL 평가 실행 (전체 모드)
    try:
        system.run_full_evaluation(
            gyeonggi_file="result_gyeonggi_policy.json",
            test_mode=False,  # 전체 정책 평가
            evaluation_method="geval",
        )

        print("✅ 전체 시스템 테스트 완료!")

        # 생성된 파일 확인
        import glob

        geval_files = glob.glob("*geval*.json")
        if geval_files:
            print(f"\n💾 생성된 G-EVAL 결과 파일:")
            for file in sorted(geval_files)[-3:]:  # 최근 3개 파일만
                print(f"   📄 {file} ({os.path.getsize(file)} bytes)")

    except Exception as e:
        print(f"❌ 전체 시스템 테스트 중 오류: {e}")


if __name__ == "__main__":
    print("G-EVAL Form-filling Paradigm 테스트")
    print("이미지 기반 진정한 G-EVAL 구현")
    print("=" * 60)
    print("💾 모든 테스트 결과는 자동으로 JSON 파일로 저장됩니다.")
    print("=" * 60)

    choice = input(
        "테스트 방법을 선택하세요:\n1. 단일 정책 상세 테스트 (결과 파일: single_geval_test_*.json)\n2. 전체 시스템 테스트 (결과 파일: test_gyeonggi_evaluation_geval_*.json)\n선택 (1 또는 2): "
    ).strip()

    if choice == "1":
        test_single_policy()
    elif choice == "2":
        test_full_evaluation()
    else:
        print("1 또는 2를 선택해주세요.")

    print("\n🎯 테스트가 완료되었습니다!")
    print("📋 결과 파일에는 다음이 포함됩니다:")
    print("   • 테스트 메타데이터 (날짜, 모델, 평가방법)")
    print("   • 정책 정보 (이름, 내용, 예산 등)")
    print("   • 평가 결과 (각 기준별 점수, 종합점수)")
    print("   • 상세 분석 (CoT 과정, 확률 분포)")
    print("   • 통계 요약 (최고/최저 점수, 표준편차 등)")
