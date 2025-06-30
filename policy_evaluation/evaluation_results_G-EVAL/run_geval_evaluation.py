#!/usr/bin/env python3
"""
G-EVAL 방식 정책 평가 실행 스크립트
"""

import os

from dotenv import load_dotenv
from policy_evaluation_system import PolicyEvaluationSystem


def main():
    """G-EVAL 방식으로 정책 평가 실행"""

    # 환경변수 로드
    load_dotenv()

    # API 키 설정
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print(".env 파일에 OPENAI_API_KEY=your-api-key를 추가해주세요.")
        return

    print("🚀 G-EVAL 방식 정책 평가 시작")
    print("=" * 50)

    # 평가 시스템 초기화
    evaluation_system = PolicyEvaluationSystem(api_key, model="gpt-4o-mini")

    # 데이터 파일 경로
    gyeonggi_file = "result_gyeonggi_policy.json"

    # 파일 존재 확인
    if not os.path.exists(gyeonggi_file):
        print(f"Error: {gyeonggi_file} 파일을 찾을 수 없습니다.")
        return

    # G-EVAL 평가 실행
    try:
        result_file = evaluation_system.run_full_evaluation(
            gyeonggi_file=gyeonggi_file,
            test_mode=False,  # 전체 모드 (모든 정책)
            evaluation_method="geval",
        )

        print("=" * 50)
        print("✅ G-EVAL 평가 완료!")
        print(f"📊 결과 파일이 저장되었습니다.")

    except Exception as e:
        print("=" * 50)
        print(f"❌ 평가 중 오류 발생: {e}")
        print("API 키가 올바른지, 인터넷 연결이 정상인지 확인해주세요.")


if __name__ == "__main__":
    main()
