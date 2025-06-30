# 재정자립도 고려 ver 1.
# 예산이 전체 예산이 아니고, '청년 정책 한정' 예산을 고려함
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def load_policy_data(policy_file_path):
    """경상남도 정책 데이터를 로드하고 총 예산을 계산합니다."""
    with open(policy_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_budget = 0
    policy_count = 0

    for region, info in data.items():
        if "정책수행" in info:
            for category, policies in info["정책수행"].items():
                if isinstance(policies, dict) and "세부사업" in policies:
                    for policy in policies["세부사업"]:
                        budget = policy.get("예산", 0)
                        total_budget += budget
                        policy_count += 1

    return {"total_budget": total_budget, "policy_count": policy_count, "data": data}


def load_youth_ratio_data(youth_ratio_file_path):
    """청년인구 비율 데이터를 로드합니다."""
    df = pd.read_csv(youth_ratio_file_path, encoding="utf-8")
    return df


def load_finance_autonomy_data(finance_file_path):
    """재정자립도 데이터를 로드합니다."""
    df = pd.read_csv(finance_file_path, encoding="utf-8")
    return df


def calculate_gyeongnam_youth_concentration_index(
    policy_data, youth_data, finance_data, total_budget_assumption=1000000
):
    """
    경상남도의 청년 예산 집중도 지수를 계산합니다.

    Args:
        policy_data: 정책 데이터
        youth_data: 청년인구 비율 데이터
        finance_data: 재정자립도 데이터
        total_budget_assumption: 경상남도 총 예산 가정값 (백만원 단위)
    """

    # 1. 경상남도 청년정책 총예산 계산 (A의 분자)
    youth_policy_budget = policy_data["total_budget"]  # 백만원 단위

    # 2. 경상남도 총예산 (가정값 사용) (A의 분모)
    total_budget = total_budget_assumption

    # 3. A: 청년정책예산 비율 = 청년정책 총예산 / 총예산
    A = youth_policy_budget / total_budget

    # 4. B: 경상남도 청년인구 비율 계산
    # 경상남도 전체 청년인구 비율 (광역코드 38)
    gyeongnam_youth_data = youth_data[youth_data["광역코드"] == 38]
    total_youth_population = gyeongnam_youth_data["시군구_청년인구"].sum()

    # 경상남도 총 인구 계산을 위해 청년인구와 청년비율을 이용
    # 첫 번째 행의 광역_청년인구를 사용 (모든 행이 동일)
    total_youth_in_gyeongnam = gyeongnam_youth_data["광역_청년인구"].iloc[0]

    # 전국 청년인구 대비 경상남도 청년인구 비율
    B = (
        total_youth_in_gyeongnam
        / youth_data["광역_청년인구"].sum()
        * len(youth_data["광역코드"].unique())
    )

    # 실제로는 경상남도 내에서의 청년인구 비율을 사용해야 함
    # 경상남도의 평균 청년비율 계산
    B = gyeongnam_youth_data["청년비율"].mean()

    # 5. 청년 예산 집중도 지수 = A/B
    concentration_index = A / B if B != 0 else 0

    # 6. 경상남도 재정자립도
    finance_autonomy = (
        finance_data[finance_data["지자체명"] == "경상남도"]["재정자립도"].iloc[0] / 100
    )

    # 7. 최종 지수 계산
    # 최종지수1 = 집중도지수 × (1-재정자립도)
    final_index_1 = concentration_index * (1 - finance_autonomy)

    # 최종지수2 = ln(집중도지수/재정자립도+1)
    final_index_2 = (
        math.log(concentration_index / finance_autonomy + 1)
        if finance_autonomy != 0
        else 0
    )

    return {
        "youth_policy_budget": youth_policy_budget,
        "total_budget_assumption": total_budget,
        "A_youth_budget_ratio": A,
        "B_youth_population_ratio": B,
        "concentration_index": concentration_index,
        "finance_autonomy": finance_autonomy,
        "final_index_1": final_index_1,
        "final_index_2": final_index_2,
    }


def calculate_detailed_analysis_by_city(policy_data, youth_data, finance_data):
    """
    경상남도 내 각 시군별 상세 분석 (정책 예산이 있는 경우에만)
    """
    results = []

    # 경상남도 시군 목록
    gyeongnam_cities = youth_data[youth_data["광역코드"] == 38]

    for _, city_data in gyeongnam_cities.iterrows():
        city_name = city_data["지자체명"]
        youth_ratio = city_data["청년비율"]

        # 재정자립도 찾기
        finance_row = finance_data[finance_data["지자체명"] == city_name]
        if len(finance_row) > 0:
            finance_autonomy = finance_row["재정자립도"].iloc[0] / 100

            # 실제 정책 데이터가 있는 경우에만 계산
            # 현재는 경상남도 전체 데이터만 있으므로 예시로 계산
            if "시" in city_name or "군" in city_name:
                # 가상의 청년정책 예산 (실제로는 각 시군별 데이터가 필요)
                estimated_youth_budget = 0  # 실제 데이터 없음
                estimated_total_budget = 100000  # 가정값

                if estimated_total_budget > 0:
                    A = estimated_youth_budget / estimated_total_budget
                    B = youth_ratio
                    concentration_index = A / B if B != 0 else 0

                    final_index_1 = concentration_index * (1 - finance_autonomy)
                    final_index_2 = (
                        math.log(concentration_index / finance_autonomy + 1)
                        if finance_autonomy != 0
                        else 0
                    )

                    results.append(
                        {
                            "city": city_name,
                            "youth_ratio": youth_ratio,
                            "finance_autonomy": finance_autonomy,
                            "concentration_index": concentration_index,
                            "final_index_1": final_index_1,
                            "final_index_2": final_index_2,
                        }
                    )

    return results


def load_evaluation_results(evaluation_file_path):
    """기존 정책 평가 결과 데이터를 로드합니다."""
    df = pd.read_csv(evaluation_file_path, encoding="utf-8")
    return df


def calculate_city_concentration_indices(
    evaluation_data, finance_data, total_budget_assumptions=None
):
    """
    경상남도 내 각 시군별 청년 예산 집중도 지수를 계산합니다.

    Args:
        evaluation_data: 기존 정책 평가 결과 데이터
        finance_data: 재정자립도 데이터
        total_budget_assumptions: 각 시군별 총예산 가정값 딕셔너리 (없으면 자동 계산)
    """

    results = []

    # 경상남도 전체 청년인구 (첫 번째 행이 경상남도 전체)
    total_youth_population = evaluation_data.loc[0, "청년인구"]

    # 기본 총예산 배율 (청년정책예산 대비)
    default_budget_multiplier = 1000  # 청년정책예산의 1000배를 총예산으로 가정

    for _, row in evaluation_data.iterrows():
        city_name = row["지역명"]
        youth_policy_budget = row["총_예산_백만원"]  # 청년정책 예산
        youth_population = row["청년인구"]
        policy_count = row["총_정책수"]

        # 총예산 가정값 설정
        if total_budget_assumptions and city_name in total_budget_assumptions:
            total_budget = total_budget_assumptions[city_name]
        else:
            # 청년정책 예산의 배수로 총예산 추정
            if youth_policy_budget > 0:
                # 시군 규모에 따라 다른 배율 적용
                if "시" in city_name:
                    multiplier = 500  # 시는 500배
                elif "군" in city_name:
                    multiplier = 300  # 군은 300배
                else:  # 경상남도
                    multiplier = 1000  # 도는 1000배
                total_budget = youth_policy_budget * multiplier
            else:
                total_budget = 100000  # 기본값: 10만백만원

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

        results.append(
            {
                "지역명": city_name,
                "청년정책예산(백만원)": youth_policy_budget,
                "총예산가정(백만원)": total_budget,
                "청년인구": youth_population,
                "A_청년정책예산비율": A,
                "B_청년인구비율": B,
                "집중도지수": concentration_index,
                "재정자립도": finance_autonomy,
                "최종지수1": final_index_1,
                "최종지수2": final_index_2,
                "정책수": policy_count,
                "전략적_강도_엔트로피": row.get("전략적_강도_엔트로피", 0),
                "행정적_강도_가중치": row.get("행정적_강도_가중치", 0),
                "정책_분야수": row.get("정책_분야수", 0),
            }
        )

    return results


def create_comprehensive_analysis(results):
    """종합적인 분석 결과를 생성합니다."""

    df = pd.DataFrame(results)

    # 순위 계산 (높은 값이 좋은 순위)
    df["최종지수1_순위"] = df["최종지수1"].rank(ascending=False, method="min")
    df["최종지수2_순위"] = df["최종지수2"].rank(ascending=False, method="min")
    df["집중도지수_순위"] = df["집중도지수"].rank(ascending=False, method="min")

    # 정규화 (0-1 스케일)
    for col in ["최종지수1", "최종지수2", "집중도지수"]:
        max_val = df[col].max()
        min_val = df[col].min()
        if max_val > min_val:
            df[f"{col}_정규화"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"{col}_정규화"] = 0

    return df


def add_final_indices(evaluation_data, finance_data):
    """
    기존 평가 데이터에 최종지수1과 최종지수2를 추가합니다.
    """

    # 기존 데이터 복사
    result_df = evaluation_data.copy()

    # 경상남도 전체 청년인구 (첫 번째 행이 경상남도 전체)
    total_youth_population = evaluation_data.loc[0, "청년인구"]

    final_index_1_list = []
    final_index_2_list = []

    for _, row in evaluation_data.iterrows():
        city_name = row["지역명"]
        youth_policy_budget = row["총_예산_백만원"]  # 청년정책 예산
        youth_population = row["청년인구"]

        # 총예산 가정값 설정
        if youth_policy_budget > 0:
            # 시군 규모에 따라 다른 배율 적용
            if "시" in city_name:
                multiplier = 500  # 시는 500배
            elif "군" in city_name:
                multiplier = 300  # 군은 300배
            else:  # 경상남도
                multiplier = 1000  # 도는 1000배
            total_budget = youth_policy_budget * multiplier
        else:
            total_budget = 100000  # 기본값: 10만백만원

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

        final_index_1_list.append(final_index_1)
        final_index_2_list.append(final_index_2)

    # 새로운 컬럼 추가
    result_df["최종지수1"] = final_index_1_list
    result_df["최종지수2"] = final_index_2_list

    return result_df


def main():
    """메인 실행 함수"""

    # 파일 경로 설정
    base_path = Path(__file__).parent
    evaluation_file = base_path / "policy_evaluation_results.csv"
    finance_file = (
        base_path.parent.parent
        / "data/policy/재정자립도/finance_autonomy_processed.csv"
    )

    # 데이터 로드
    print("=== 데이터 로딩 중 ===")
    evaluation_data = load_evaluation_results(evaluation_file)
    finance_data = load_finance_autonomy_data(finance_file)

    print(f"평가 대상 지역 수: {len(evaluation_data)}")

    # 최종 지수 계산 및 추가
    print("\n=== 최종지수1, 최종지수2 계산 중 ===")
    result_df = add_final_indices(evaluation_data, finance_data)

    # 결과 미리보기
    print("\n=== 결과 미리보기 ===")
    display_columns = ["지역명", "총_예산_백만원", "청년인구", "최종지수1", "최종지수2"]
    display_df = result_df[display_columns].copy()
    display_df["최종지수1"] = display_df["최종지수1"].round(6)
    display_df["최종지수2"] = display_df["최종지수2"].round(6)
    print(display_df.to_string(index=False))

    # 상위 5개 지역 확인
    print("\n=== 최종지수1 기준 상위 5개 지역 ===")
    top5_index1 = result_df.nlargest(5, "최종지수1")[["지역명", "최종지수1"]].round(6)
    print(top5_index1.to_string(index=False))

    print("\n=== 최종지수2 기준 상위 5개 지역 ===")
    top5_index2 = result_df.nlargest(5, "최종지수2")[["지역명", "최종지수2"]].round(6)
    print(top5_index2.to_string(index=False))

    # 결과 저장
    print("\n=== 결과 저장 ===")
    output_file = base_path / "policy_evaluation_results_with_final_indices.csv"
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"최종지수가 추가된 결과가 저장되었습니다: {output_file}")

    print("\n=== 추가된 컬럼 정보 ===")
    print("최종지수1: 집중도지수 × (1-재정자립도)")
    print("최종지수2: ln(집중도지수/재정자립도+1)")


if __name__ == "__main__":
    main()
