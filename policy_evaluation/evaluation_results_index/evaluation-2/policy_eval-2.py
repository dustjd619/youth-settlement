# 전국 청년정책 종합 평가 시스템
# 전략적 강도: 엔트로피 지수 (정책 분야별 균형성과 다양성)
# 행정적 강도: ln(집중도지수/재정자립도+1) (집중도지수와 재정자립도 고려)

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class NationalPolicyEvaluator:
    """전국 청년정책 종합 평가 클래스"""

    def __init__(self):
        """초기화"""
        self.base_path = Path(__file__).parent
        self.policy_data = {}
        self.youth_population_data = None
        self.finance_data = None
        self.basic_budget_data = None
        self.metro_budget_data = None

        # 데이터 로드
        self._load_all_data()

    def _load_all_data(self):
        """모든 필요한 데이터를 로드합니다."""
        # 1. 모든 정책 데이터 로드 (각 시/군별로 분리)
        policy_folder = self.base_path.parent.parent / "data/policy/정책책자"
        for file_path in policy_folder.glob("*_정책_최종본.json"):
            if file_path.name != "empty":
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    # 각 JSON 파일 안의 모든 지역을 별도로 추가
                    for region_name, region_data in raw_data.items():
                        if isinstance(region_data, dict) and "정책수행" in region_data:
                            self.policy_data[region_name] = region_data

        # 2. 청년인구 데이터 로드
        youth_population_path = (
            self.base_path.parent.parent / "data/policy/청년비율_시군구_기준.csv"
        )
        self.youth_population_data = pd.read_csv(
            youth_population_path, encoding="utf-8"
        )

        # 3. 재정자립도 데이터 로드
        finance_path = (
            self.base_path.parent.parent
            / "data/policy/재정자립도/finance_autonomy_processed.csv"
        )
        self.finance_data = pd.read_csv(finance_path, encoding="utf-8")

        # 4. 기초자치단체 예산 데이터 로드
        basic_budget_path = (
            self.base_path.parent.parent / "data/budget/세출예산_기초자치단체.csv"
        )
        self.basic_budget_data = pd.read_csv(basic_budget_path, encoding="utf-8")

        # 5. 광역자치단체 예산 데이터 로드
        metro_budget_path = (
            self.base_path.parent.parent / "data/budget/세출예산_광역자치단체.csv"
        )
        self.metro_budget_data = pd.read_csv(metro_budget_path, encoding="utf-8")

        print(f"✓ 정책 데이터: {len(self.policy_data)}개 지역")
        print(f"✓ 청년인구 데이터: {len(self.youth_population_data)}개 지역")
        print(f"✓ 재정자립도 데이터: {len(self.finance_data)}개 지역")
        print(f"✓ 기초자치단체 예산 데이터: {len(self.basic_budget_data)}개 지역")
        print(f"✓ 광역자치단체 예산 데이터: {len(self.metro_budget_data)}개 지역")

    def calculate_strategic_intensity(self, region_data: Dict) -> float:
        """
        전략적 강도 계산 (엔트로피 지수)
        Formula: (Entropy / log(N)) * (실제 분야수 / 최대 분야수)
        """
        policy_areas = region_data.get("정책수행", {})
        max_possible_areas = 5  # 최대 가능한 정책 분야수

        if not policy_areas:
            return 0.0

        # 각 분야별 사업 수 계산
        area_counts = {}
        total_count = 0

        for area, data in policy_areas.items():
            # 세부사업 리스트에서 사업 수 계산
            policies = data.get("세부사업", [])
            count = len(policies)
            if count > 0:
                area_counts[area] = count
                total_count += count

        if total_count == 0 or len(area_counts) <= 1:
            return 0.0

        # 엔트로피 계산
        entropy = 0.0
        for count in area_counts.values():
            p_i = count / total_count
            if p_i > 0:
                entropy -= p_i * math.log(p_i)

        # 정규화 (0~1 사이값으로)
        max_entropy = math.log(len(area_counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # 다양성 페널티 적용: 실제 분야수 / 최대 가능 분야수
        diversity_penalty = len(area_counts) / max_possible_areas

        # 최종 전략적 강도 = 균형성 × 다양성
        strategic_intensity = normalized_entropy * diversity_penalty

        return strategic_intensity

    def get_youth_population(self, region_name: str) -> int:
        """지역의 청년인구를 조회합니다."""
        # 정확한 매칭 시도
        for _, row in self.youth_population_data.iterrows():
            if region_name in row["지자체명"] or row["지자체명"] in region_name:
                return row["시군구_청년인구"]

        # 광역자치단체의 경우 전체 청년인구 조회
        if any(keyword in region_name for keyword in ["도", "특별시", "광역시"]):
            filtered_data = self.youth_population_data[
                self.youth_population_data["지자체명"].str.contains(
                    region_name, na=False
                )
            ]
            if len(filtered_data) > 0:
                return filtered_data["시군구_청년인구"].sum()

        # 기본값 반환
        default_populations = {
            "서울특별시": 2500000,
            "부산광역시": 800000,
            "대구광역시": 600000,
            "인천광역시": 700000,
            "광주광역시": 350000,
            "대전광역시": 350000,
            "울산광역시": 250000,
            "경상남도": 524436,
            "경상북도": 600000,
            "전라남도": 400000,
            "전북특별자치도": 400000,
            "제주특별자치도": 150000,
        }

        return default_populations.get(region_name, 30000)  # 기초자치단체 기본값을 낮춤

    def get_finance_autonomy(self, region_name: str) -> float:
        """지역의 재정자립도를 조회합니다."""
        # 정확한 매칭 시도
        exact_match = self.finance_data[self.finance_data["지자체명"] == region_name]
        if len(exact_match) > 0:
            return exact_match["재정자립도"].iloc[0] / 100

        # 부분 매칭 시도
        partial_matches = []
        for _, row in self.finance_data.iterrows():
            if region_name in row["지자체명"] or row["지자체명"] in region_name:
                partial_matches.append(row["재정자립도"] / 100)

        if partial_matches:
            return partial_matches[0]

        # 기본값 (전국 평균 수준)
        return 0.35

    def get_total_budget(self, region_name: str) -> float:
        """지역의 총예산을 조회합니다."""
        # 1. 광역자치단체인지 확인
        if self.is_metropolitan_government(region_name):
            # 광역자치단체 예산 데이터에서 조회
            # 데이터에서 "본청" 제거하여 매칭
            for _, row in self.metro_budget_data.iterrows():
                budget_region = row["자치단체명"].replace(" 본청", "")

                # 정확한 매칭
                if region_name == budget_region:
                    return row["세출총계"]

                # 특별 케이스 매칭
                if (
                    ("전북특별자치도" in region_name and "전라북도" in budget_region)
                    or ("전라북도" in region_name and "전라북도" in budget_region)
                    or ("대전" in region_name and "대전" in budget_region)
                    or (
                        region_name.replace("특별자치도", "도")
                        == budget_region.replace("특별자치도", "도")
                    )
                    or (
                        region_name.replace("특별시", "")
                        == budget_region.replace("특별시", "")
                    )
                    or (
                        region_name.replace("광역시", "")
                        == budget_region.replace("광역시", "")
                    )
                ):
                    return row["세출총계"]

            # 기본 추정값 (광역자치단체)
            default_budgets = {
                "서울특별시": 52387838.2,
                "부산광역시": 16690868.3,
                "대구광역시": 11649756.4,
                "인천광역시": 15379013.7,
                "광주광역시": 7467959.8,
                "대전광역시": 7540100.2,
                "울산광역시": 5194230.3,
                "경상남도": 12742649.6,
                "경상북도": 13372746.4,
                "전라남도": 11809873.2,
                "전북특별자치도": 9861808.1,
                "제주특별자치도": 8070231.1,
            }
            return default_budgets.get(region_name, 10000000)

        else:
            # 기초자치단체 예산 데이터에서 조회
            # 정확한 매칭 시도
            exact_match = self.basic_budget_data[
                self.basic_budget_data["자치단체명"] == region_name
            ]
            if len(exact_match) > 0:
                return exact_match["세출총계"].iloc[0]

            # 부분 매칭 시도
            for _, row in self.basic_budget_data.iterrows():
                if region_name in row["자치단체명"] or row["자치단체명"] in region_name:
                    return row["세출총계"]

            # 기본값 (기초자치단체)
            return 1000000

    def calculate_total_policy_budget(self, region_data: Dict) -> float:
        """지역의 총 청년정책 예산을 계산합니다."""
        policy_areas = region_data.get("정책수행", {})
        total_budget = 0.0

        for area, data in policy_areas.items():
            # 세부사업들의 예산 합계 계산
            policies = data.get("세부사업", [])
            for policy in policies:
                budget = policy.get("예산", 0)
                total_budget += budget

        return total_budget / 100  # 백만원 단위로 변환

    def calculate_administrative_intensity(
        self, region_name: str, region_data: Dict, total_youth_population: int
    ) -> float:
        """
        행정적 강도 계산 (최종지수2)
        Formula: ln(집중도지수/재정자립도+1)
        집중도지수 = (청년정책예산/총예산) / (해당지역청년인구/전체청년인구)
        """
        # 청년정책 예산
        youth_policy_budget = self.calculate_total_policy_budget(region_data)

        # 총예산
        total_budget = self.get_total_budget(region_name)

        # 청년인구
        youth_population = self.get_youth_population(region_name)

        # 재정자립도
        finance_autonomy = self.get_finance_autonomy(region_name)

        if total_budget == 0 or total_youth_population == 0 or finance_autonomy == 0:
            return 0.0

        # A: 청년정책예산 비율
        A = youth_policy_budget / total_budget

        # B: 청년인구 비율
        B = youth_population / total_youth_population

        # 집중도 지수
        concentration_index = A / B if B > 0 else 0

        # 최종지수2 = ln(집중도지수/재정자립도+1)
        if finance_autonomy > 0:
            administrative_intensity = math.log(
                concentration_index / finance_autonomy + 1
            )
        else:
            administrative_intensity = 0

        return administrative_intensity

    def evaluate_all_regions(self) -> pd.DataFrame:
        """모든 지역에 대한 종합 평가를 수행합니다."""
        results = []

        # 전체 청년인구 계산
        total_youth_population = sum(
            self.get_youth_population(region) for region in self.policy_data.keys()
        )

        print(f"\n=== 전국 {len(self.policy_data)}개 지역 정책 평가 중 ===")

        for region_name, region_data in self.policy_data.items():
            print(f"분석 중: {region_name}")

            # 전략적 강도 계산
            strategic_intensity = self.calculate_strategic_intensity(region_data)

            # 행정적 강도 계산
            administrative_intensity = self.calculate_administrative_intensity(
                region_name, region_data, total_youth_population
            )

            # 추가 메트릭 계산
            total_policies = self._count_total_policies(region_data)
            total_budget = self.calculate_total_policy_budget(region_data)
            policy_areas = len(region_data.get("정책수행", {}))
            youth_population = self.get_youth_population(region_name)
            finance_autonomy = self.get_finance_autonomy(region_name)
            total_region_budget = self.get_total_budget(region_name)

            results.append(
                {
                    "지역명": region_name,
                    "전략적_강도_엔트로피": round(strategic_intensity, 4),
                    "행정적_강도_최종지수2": round(administrative_intensity, 6),
                    "총_정책수": total_policies,
                    "청년정책_예산_백만원": round(total_budget, 2),
                    "지자체_전체예산_백만원": total_region_budget,
                    "정책_분야수": policy_areas,
                    "청년인구": youth_population,
                    "재정자립도": round(finance_autonomy, 3),
                    "A_청년정책예산비율": (
                        round(total_budget / total_region_budget, 8)
                        if total_region_budget > 0
                        else 0
                    ),
                    "B_청년인구비율": (
                        round(youth_population / total_youth_population, 6)
                        if total_youth_population > 0
                        else 0
                    ),
                    "집중도지수": (
                        round(
                            (total_budget / total_region_budget)
                            / (youth_population / total_youth_population),
                            6,
                        )
                        if total_region_budget > 0
                        and total_youth_population > 0
                        and youth_population > 0
                        else 0
                    ),
                }
            )

        return pd.DataFrame(results)

    def _count_total_policies(self, region_data: Dict) -> int:
        """총 정책 수 계산"""
        total = 0
        for area_data in region_data.get("정책수행", {}).values():
            # 세부사업 리스트에서 사업 수 계산
            policies = area_data.get("세부사업", [])
            total += len(policies)
        return total

    def analyze_top_performers(self, results_df: pd.DataFrame) -> Dict:
        """상위 성과 지역 분석"""
        analysis = {}

        # 전략적 강도 상위 5개 지역
        top_strategic = results_df.nlargest(5, "전략적_강도_엔트로피")
        analysis["전략적_강도_상위5"] = top_strategic[
            ["지역명", "전략적_강도_엔트로피", "정책_분야수", "총_정책수"]
        ].to_dict("records")

        # 행정적 강도 상위 5개 지역
        top_administrative = results_df.nlargest(5, "행정적_강도_최종지수2")
        analysis["행정적_강도_상위5"] = top_administrative[
            ["지역명", "행정적_강도_최종지수2", "집중도지수", "재정자립도"]
        ].to_dict("records")

        # 종합 점수 (전략적 강도 + 행정적 강도의 정규화 합계)
        max_strategic = results_df["전략적_강도_엔트로피"].max()
        min_strategic = results_df["전략적_강도_엔트로피"].min()
        max_admin = results_df["행정적_강도_최종지수2"].max()
        min_admin = results_df["행정적_강도_최종지수2"].min()

        if max_strategic > min_strategic:
            results_df["전략적_강도_정규화"] = (
                results_df["전략적_강도_엔트로피"] - min_strategic
            ) / (max_strategic - min_strategic)
        else:
            results_df["전략적_강도_정규화"] = 0

        if max_admin > min_admin:
            results_df["행정적_강도_정규화"] = (
                results_df["행정적_강도_최종지수2"] - min_admin
            ) / (max_admin - min_admin)
        else:
            results_df["행정적_강도_정규화"] = 0

        results_df["종합점수"] = (
            results_df["전략적_강도_정규화"] + results_df["행정적_강도_정규화"]
        ) / 2

        top_comprehensive = results_df.nlargest(5, "종합점수")
        analysis["종합점수_상위5"] = top_comprehensive[
            ["지역명", "종합점수", "전략적_강도_엔트로피", "행정적_강도_최종지수2"]
        ].to_dict("records")

        return analysis

    def generate_regional_comparison(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """지역별 비교 분석 테이블 생성"""
        comparison_df = results_df.copy()

        # 순위 계산
        comparison_df["전략적강도_순위"] = comparison_df["전략적_강도_엔트로피"].rank(
            ascending=False, method="min"
        )
        comparison_df["행정적강도_순위"] = comparison_df["행정적_강도_최종지수2"].rank(
            ascending=False, method="min"
        )
        comparison_df["종합점수_순위"] = comparison_df["종합점수"].rank(
            ascending=False, method="min"
        )

        # 결과 정렬 (종합점수 기준)
        comparison_df = comparison_df.sort_values("종합점수", ascending=False)

        return comparison_df

    def save_results(self, results_df: pd.DataFrame, analysis: Dict):
        """결과를 파일로 저장합니다."""
        output_path = self.base_path / "전국_청년정책_종합평가결과.csv"
        results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        # 분석 결과를 JSON으로 저장
        analysis_path = self.base_path / "전국_청년정책_분석결과.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 결과 저장 완료:")
        print(f"  - {output_path}")
        print(f"  - {analysis_path}")

    def is_metropolitan_government(self, region_name: str) -> bool:
        """광역자치단체인지 확인합니다."""
        metro_keywords = ["특별시", "광역시", "특별자치시", "특별자치도"]
        metro_provinces = [
            "경기도",
            "강원도",
            "충청북도",
            "충청남도",
            "전라북도",
            "전북특별자치도",
            "전라남도",
            "경상북도",
            "경상남도",
            "제주특별자치도",
        ]

        # 키워드 기반 확인
        for keyword in metro_keywords:
            if keyword in region_name:
                return True

        # 도 단위 확인
        for province in metro_provinces:
            if region_name == province or province in region_name:
                return True

        return False


def main():
    """메인 실행 함수"""
    print("=== 전국 청년정책 종합 평가 시스템 ===\n")

    try:
        # 평가 시스템 초기화
        evaluator = NationalPolicyEvaluator()

        # 전체 지역 평가
        results_df = evaluator.evaluate_all_regions()

        # 상위 성과 지역 분석
        analysis = evaluator.analyze_top_performers(results_df)

        # 지역별 비교 테이블 생성
        comparison_df = evaluator.generate_regional_comparison(results_df)

        # 결과 출력
        print("\n=== 전국 청년정책 평가 결과 ===")
        print(
            comparison_df[
                [
                    "지역명",
                    "전략적_강도_엔트로피",
                    "행정적_강도_최종지수2",
                    "종합점수",
                    "종합점수_순위",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

        print("\n=== 전략적 강도 상위 5개 지역 ===")
        for i, region in enumerate(analysis["전략적_강도_상위5"], 1):
            print(
                f"{i}. {region['지역명']}: {region['전략적_강도_엔트로피']:.4f} (분야수: {region['정책_분야수']}, 정책수: {region['총_정책수']})"
            )

        print("\n=== 행정적 강도 상위 5개 지역 ===")
        for i, region in enumerate(analysis["행정적_강도_상위5"], 1):
            print(
                f"{i}. {region['지역명']}: {region['행정적_강도_최종지수2']:.6f} (집중도: {region['집중도지수']:.6f}, 재정자립도: {region['재정자립도']:.3f})"
            )

        print("\n=== 종합 점수 상위 5개 지역 ===")
        for i, region in enumerate(analysis["종합점수_상위5"], 1):
            print(
                f"{i}. {region['지역명']}: {region['종합점수']:.4f} (전략적: {region['전략적_강도_엔트로피']:.4f}, 행정적: {region['행정적_강도_최종지수2']:.6f})"
            )

        # 결과 저장
        evaluator.save_results(comparison_df, analysis)

        print("\n=== 평가 완료 ===")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
