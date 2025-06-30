# 재정자립도 고려 ver 2.
# 전체 예산을 고려함
# 실제 세출예산 데이터를 사용하여 정확한 청년정책예산 비율 계산

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_evaluation_results(evaluation_file_path):
    """기존 정책 평가 결과 데이터를 로드합니다."""
    df = pd.read_csv(evaluation_file_path, encoding="utf-8")
    return df


def load_finance_autonomy_data(finance_file_path):
    """재정자립도 데이터를 로드합니다."""
    df = pd.read_csv(finance_file_path, encoding="utf-8")
    return df


def load_budget_data(budget_file_path):
    """기초자치단체 세출예산 데이터를 로드합니다."""
    df = pd.read_csv(budget_file_path, encoding="utf-8")
    return df


def normalize_region_name(region_name):
    """지역명을 정규화하여 매칭 정확도를 높입니다."""
    # 공통적인 변환 규칙
    normalized = region_name.strip()

    # 특별시/광역시 처리
    normalized = normalized.replace("특별시 ", "")
    normalized = normalized.replace("광역시 ", "")
    normalized = normalized.replace("도 ", "")

    # 경상남도 특별 처리
    if "경상남도" in normalized:
        normalized = normalized.replace("경상남도 ", "")

    return normalized


def find_matching_budget(region_name, budget_data):
    """지역명에 해당하는 예산 데이터를 찾습니다."""
    # 1. 정확한 매칭 시도
    exact_match = budget_data[budget_data["자치단체명"] == region_name]
    if len(exact_match) > 0:
        return exact_match["세출총계"].iloc[0]

    # 2. 정규화된 이름으로 매칭 시도
    normalized_region = normalize_region_name(region_name)

    for _, row in budget_data.iterrows():
        budget_region = normalize_region_name(row["자치단체명"])
        if normalized_region in budget_region or budget_region in normalized_region:
            return row["세출총계"]

    # 3. 부분 문자열 매칭 시도 (경상남도 지역의 경우)
    if "경상남도" in region_name:
        city_name = region_name.replace("경상남도 ", "")
        for _, row in budget_data.iterrows():
            if city_name in row["자치단체명"]:
                return row["세출총계"]

    # 매칭되지 않으면 None 반환
    return None


def calculate_final_indices_with_real_budget(
    evaluation_data, finance_data, budget_data
):
    """
    실제 총예산 데이터를 사용하여 최종지수1과 최종지수2를 계산합니다.
    """

    # 기존 데이터 복사
    result_df = evaluation_data.copy()

    # 경상남도 전체 청년인구 (첫 번째 행이 경상남도 전체)
    total_youth_population = evaluation_data.loc[0, "청년인구"]

    final_index_1_list = []
    final_index_2_list = []
    actual_budget_list = []
    budget_found_list = []
    A_ratio_list = []
    B_ratio_list = []
    concentration_index_list = []

    print("=== 지역별 예산 매칭 결과 ===")

    for _, row in evaluation_data.iterrows():
        city_name = row["지역명"]
        youth_policy_budget = row["총_예산_백만원"]  # 청년정책 예산
        youth_population = row["청년인구"]

        # 실제 총예산 찾기
        actual_total_budget = find_matching_budget(city_name, budget_data)

        if actual_total_budget is not None:
            total_budget = actual_total_budget
            budget_found = True
            print(f"✓ {city_name}: 실제예산 {total_budget:,.0f}백만원")
        else:
            # 실제 예산을 찾지 못한 경우 기존 방식 사용
            if youth_policy_budget > 0:
                if "시" in city_name:
                    multiplier = 500
                elif "군" in city_name:
                    multiplier = 300
                else:  # 경상남도
                    multiplier = 1000
                total_budget = youth_policy_budget * multiplier
            else:
                total_budget = 100000  # 기본값
            budget_found = False
            print(
                f"✗ {city_name}: 추정예산 {total_budget:,.0f}백만원 (실제 데이터 없음)"
            )

        # A: 청년정책예산 비율 = 청년정책예산 / 총예산
        A = youth_policy_budget / total_budget if total_budget > 0 else 0

        # B: 청년인구 비율 = 해당 시군 청년인구 / 경상남도 전체 청년인구
        B = (
            youth_population / total_youth_population
            if total_youth_population > 0
            else 0
        )

        # 청년 예산 집중도 지수 = A/B
        concentration_index = A / B if B > 0 else 0

        # 재정자립도 찾기
        finance_row = finance_data[finance_data["지자체명"] == city_name]
        if len(finance_row) > 0:
            finance_autonomy = finance_row["재정자립도"].iloc[0] / 100
        else:
            # 만약 정확한 매칭이 안되면 유사한 이름으로 찾기
            matching_rows = finance_data[
                finance_data["지자체명"].str.contains(
                    city_name.replace("경상남도 ", ""), na=False
                )
            ]
            if len(matching_rows) > 0:
                finance_autonomy = matching_rows["재정자립도"].iloc[0] / 100
            else:
                finance_autonomy = 0.337  # 경상남도 평균값으로 대체

        # 최종 지수 계산
        # 최종지수1 = 집중도지수 × (1-재정자립도)
        final_index_1 = concentration_index * (1 - finance_autonomy)

        # 최종지수2 = ln(집중도지수/재정자립도+1)
        if finance_autonomy > 0:
            final_index_2 = math.log(concentration_index / finance_autonomy + 1)
        else:
            final_index_2 = 0

        # 결과 저장
        final_index_1_list.append(final_index_1)
        final_index_2_list.append(final_index_2)
        actual_budget_list.append(total_budget)
        budget_found_list.append(budget_found)
        A_ratio_list.append(A)
        B_ratio_list.append(B)
        concentration_index_list.append(concentration_index)

    # 새로운 컬럼 추가
    result_df["지자체_전체예산_백만원"] = actual_budget_list
    result_df["예산데이터_존재"] = budget_found_list
    result_df["A_청년정책예산비율"] = A_ratio_list
    result_df["B_청년인구비율"] = B_ratio_list
    result_df["집중도지수"] = concentration_index_list
    result_df["최종지수1"] = final_index_1_list
    result_df["최종지수2"] = final_index_2_list

    # 기존 컬럼명도 더 명확하게 변경
    result_df = result_df.rename(columns={"총_예산_백만원": "청년정책_예산_백만원"})

    return result_df


def analyze_budget_impact(result_df):
    """실제 예산 데이터 사용 여부에 따른 영향 분석"""

    real_budget_df = result_df[result_df["예산데이터_존재"] == True]
    estimated_budget_df = result_df[result_df["예산데이터_존재"] == False]

    print(f"\n=== 예산 데이터 분석 ===")
    print(f"실제 예산 데이터 사용: {len(real_budget_df)}개 지역")
    print(f"추정 예산 데이터 사용: {len(estimated_budget_df)}개 지역")

    if len(real_budget_df) > 0:
        print(
            f"\n실제 예산 사용 지역의 평균 최종지수1: {real_budget_df['최종지수1'].mean():.6f}"
        )
        print(
            f"실제 예산 사용 지역의 평균 A비율: {real_budget_df['A_청년정책예산비율'].mean():.6f}"
        )

    if len(estimated_budget_df) > 0:
        print(
            f"\n추정 예산 사용 지역의 평균 최종지수1: {estimated_budget_df['최종지수1'].mean():.6f}"
        )
        print(
            f"추정 예산 사용 지역의 평균 A비율: {estimated_budget_df['A_청년정책예산비율'].mean():.6f}"
        )


def main():
    """메인 실행 함수"""

    # 파일 경로 설정
    base_path = Path(__file__).parent
    evaluation_file = base_path.parent / "policy_evaluation_results-1.csv"
    finance_file = (
        base_path.parent.parent.parent
        / "data/policy/재정자립도/finance_autonomy_processed.csv"
    )
    budget_file = (
        base_path.parent.parent.parent / "data/budget/세출예산_기초자치단체.csv"
    )

    # 데이터 로드
    print("=== 데이터 로딩 중 ===")
    evaluation_data = load_evaluation_results(evaluation_file)
    finance_data = load_finance_autonomy_data(finance_file)
    budget_data = load_budget_data(budget_file)

    print(f"평가 대상 지역 수: {len(evaluation_data)}")
    print(f"예산 데이터 지역 수: {len(budget_data)}")

    # 실제 예산 데이터를 사용한 최종 지수 계산
    print("\n=== 실제 예산 데이터를 사용한 최종지수 계산 중 ===")
    result_df = calculate_final_indices_with_real_budget(
        evaluation_data, finance_data, budget_data
    )

    # 예산 데이터 영향 분석
    analyze_budget_impact(result_df)

    # 결과 미리보기
    print("\n=== 결과 미리보기 ===")
    display_columns = [
        "지역명",
        "청년정책_예산_백만원",
        "지자체_전체예산_백만원",
        "예산데이터_존재",
        "A_청년정책예산비율",
        "최종지수1",
        "최종지수2",
    ]
    display_df = result_df[display_columns].copy()
    display_df["A_청년정책예산비율"] = display_df["A_청년정책예산비율"].round(8)
    display_df["최종지수1"] = display_df["최종지수1"].round(6)
    display_df["최종지수2"] = display_df["최종지수2"].round(6)
    print(display_df.to_string(index=False))

    # 상위 5개 지역 확인 (실제 예산 데이터 사용 지역만)
    real_budget_df = result_df[result_df["예산데이터_존재"] == True]
    if len(real_budget_df) > 0:
        print("\n=== 실제 예산 데이터 기준 최종지수1 상위 5개 지역 ===")
        top5_real = real_budget_df.nlargest(5, "최종지수1")[
            ["지역명", "최종지수1", "A_청년정책예산비율"]
        ].round(6)
        print(top5_real.to_string(index=False))

    # 전체 상위 5개 지역 확인
    print("\n=== 전체 최종지수1 기준 상위 5개 지역 ===")
    top5_index1 = result_df.nlargest(5, "최종지수1")[
        ["지역명", "최종지수1", "예산데이터_존재"]
    ].round(6)
    print(top5_index1.to_string(index=False))

    print("\n=== 전체 최종지수2 기준 상위 5개 지역 ===")
    top5_index2 = result_df.nlargest(5, "최종지수2")[
        ["지역명", "최종지수2", "예산데이터_존재"]
    ].round(6)
    print(top5_index2.to_string(index=False))

    # 결과 저장
    print("\n=== 결과 저장 ===")
    output_file = base_path / "재정자립도_고려점수-2.csv"
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"실제 지자체 전체예산 데이터를 반영한 결과가 저장되었습니다: {output_file}")

    print("\n=== 추가된 컬럼 정보 ===")
    print("청년정책_예산_백만원: 각 지역의 청년 관련 정책 예산 총합")
    print("지자체_전체예산_백만원: 세출예산 데이터에서 가져온 지자체 전체 예산")
    print("예산데이터_존재: 실제 예산 데이터 사용 여부 (True/False)")
    print("A_청년정책예산비율: 청년정책예산 / 지자체전체예산")
    print("B_청년인구비율: 해당 지역 청년인구 / 전체 청년인구")
    print("집중도지수: A / B")
    print("최종지수1: 집중도지수 × (1-재정자립도)")
    print("최종지수2: ln(집중도지수/재정자립도+1)")


if __name__ == "__main__":
    main()
