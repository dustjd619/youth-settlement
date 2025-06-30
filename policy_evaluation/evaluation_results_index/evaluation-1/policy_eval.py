# 전략적+행정적지표 모두 평가(행정적강도 - 가중합평균 > 실패 버전)
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class PolicyIndexEvaluator:
    """
    정량적 지표를 통한 정책 평가 클래스
    - 전략적 강도: 엔트로피 지수를 통한 정책 분야별 균형성 측정
    - 행정적 강도: 가중치 조정을 통한 정책 수행 강도 측정
    """

    def __init__(self, policy_data_path: str, youth_population_path: str):
        """
        Args:
            policy_data_path: 정책 데이터 JSON 파일 경로
            youth_population_path: 청년 인구 데이터 CSV 파일 경로
        """
        self.policy_data = self._load_policy_data(policy_data_path)
        self.youth_population = self._load_youth_population_data(youth_population_path)

        # 사업별 가중치 기준 (이미지에서 제시된 기준 기반)
        self.weight_criteria = {
            # 단일성 행사/홍보 (축제 or 캠페인) - 0.5
            "홍보": 0.5,
            "축제": 0.5,
            "행사": 0.5,
            "캠페인": 0.5,
            "박람회": 0.5,
            # 복지/지원 (교통비, 대여, 활동비) - 1.5 (수정: 현금성 지원 중요도 반영)
            "지원": 1.5,
            "대여": 1.0,
            "비용": 1.5,
            "수당": 1.5,
            "교통비": 1.5,
            "활동비": 1.0,
            "장려금": 1.5,
            "통장": 1.5,
            "축하금": 1.5,
            "장학": 1.5,
            # 행정 연계 (인턴, 채용, 제도 운영) - 1.5
            "인턴": 1.5,
            "채용": 1.5,
            "일자리": 1.5,
            "취업": 1.5,
            "교육": 1.5,
            "운영": 1.5,
            "위원회": 1.5,
            "정책": 1.5,
            "아카데미": 1.5,
            # 인프라/제도 구축 (공간 조성, 시스템 설계) - 2.0
            "센터": 2.0,
            "플랫폼": 2.0,
            "공간": 2.0,
            "시설": 2.0,
            "구축": 2.0,
            "조성": 2.0,
            "설치": 2.0,
            "창업": 2.0,
            "인프라": 2.0,
            "시스템": 2.0,
        }

    def _load_policy_data(self, path: str) -> Dict:
        """정책 데이터 로드"""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_youth_population_data(self, path: str) -> pd.DataFrame:
        """청년 인구 데이터 로드"""
        return pd.read_csv(path)

    def _get_policy_weight(self, policy_name: str, policy_content: str) -> float:
        """
        정책의 가중치를 계산
        사업명과 주요내용을 분석하여 가중치 부여
        """
        text = (policy_name + " " + policy_content).lower()

        # 각 가중치 기준별로 점수 계산
        weights = []
        for keyword, weight in self.weight_criteria.items():
            if keyword in text:
                weights.append(weight)

        # 매칭된 가중치가 있으면 최대값 사용, 없으면 기본값 1.0
        return max(weights) if weights else 1.0

    def calculate_entropy_index(self, region_data: Dict) -> float:
        """
        엔트로피 지수 계산 (전략적 강도)

        Formula: (Entropy / log(N)) * (실제 분야수 / 최대 분야수)
        - 분야별 균형성과 다양성을 모두 고려
        - 정책 분야수가 적으면 페널티 적용

        Args:
            region_data: 특정 지역의 정책 데이터

        Returns:
            정규화된 엔트로피 지수 (0~1 사이값, 다양성 페널티 포함)
        """
        policy_areas = region_data.get("정책수행", {})
        max_possible_areas = (
            5  # 최대 가능한 정책 분야수 (일자리, 주거, 교육, 복지·문화, 참여·권리)
        )

        if not policy_areas:
            return 0.0

        # 각 분야별 사업 수 계산
        area_counts = {}
        total_count = 0

        for area, data in policy_areas.items():
            count = data.get("사업수", 0)
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

    def calculate_administrative_intensity(
        self, region_data: Dict, region_name: str
    ) -> float:
        """
        행정적 강도 계산 (가중치 조정)

        Formula: 행정적 강도 = Σ(weight_i) / 청년 인구

        Args:
            region_data: 특정 지역의 정책 데이터
            region_name: 지역명

        Returns:
            청년 1인당 가중치 조정된 정책 강도
        """
        policy_areas = region_data.get("정책수행", {})

        if not policy_areas:
            return 0.0

        # 청년 인구 데이터 조회
        youth_pop = self._get_youth_population(region_name)
        if youth_pop == 0:
            return 0.0

        # 전체 가중치 합계 계산
        total_weighted_score = 0.0

        for area, data in policy_areas.items():
            policies = data.get("세부사업", [])
            for policy in policies:
                policy_name = policy.get("사업명", "")
                policy_content = policy.get("주요내용", "")
                weight = self._get_policy_weight(policy_name, policy_content)
                total_weighted_score += weight

        # 청년 1인당 행정적 강도
        administrative_intensity = total_weighted_score / youth_pop

        return administrative_intensity

    def _get_youth_population(self, region_name: str) -> int:
        """
        지역명으로 청년 인구 조회

        Args:
            region_name: 지역명 (예: "경상남도", "진주시", "창원시")

        Returns:
            청년 인구 수
        """
        # 경상남도의 경우 시군구별 청년 인구 합계
        if region_name == "경상남도":
            gyeongnam_cities = [
                "창원시",
                "진주시",
                "통영시",
                "사천시",
                "김해시",
                "밀양시",
                "거제시",
                "양산시",
                "의령군",
                "함안군",
                "창녕군",
                "고성군",
                "남해군",
                "하동군",
                "산청군",
                "함양군",
                "거창군",
                "합천군",
            ]

            total_youth = 0
            for _, row in self.youth_population.iterrows():
                city_name = row["지자체명"]
                for city in gyeongnam_cities:
                    if city in city_name:
                        total_youth += row["시군구_청년인구"]
                        break

            return total_youth if total_youth > 0 else 521162  # 경상남도 전체 청년인구

            # 창원시의 경우 특별 처리 (5개 구 합계)
        if region_name == "창원시":
            changwon_total = 0
            for _, row in self.youth_population.iterrows():
                if "창원시" in row["지자체명"] and "경상남도" in row["지자체명"]:
                    changwon_total += row["시군구_청년인구"]
            return (
                changwon_total if changwon_total > 0 else 179486
            )  # 창원시 전체 청년인구 합계

        # 개별 시군구의 경우 정확한 매칭
        for _, row in self.youth_population.iterrows():
            city_name = row["지자체명"]

            # 정확한 지역명 매칭
            if region_name in city_name and "경상남도" in city_name:
                return row["시군구_청년인구"]

        # 매칭되지 않는 경우 기본값 반환
        return 10000  # 기본값을 더 현실적으로 조정

    def evaluate_all_regions(self) -> pd.DataFrame:
        """
        모든 지역에 대한 정량적 지표 평가

        Returns:
            평가 결과 DataFrame
        """
        results = []

        for region_name, region_data in self.policy_data.items():
            if isinstance(region_data, dict) and "정책수행" in region_data:
                # 전략적 강도 계산
                strategic_intensity = self.calculate_entropy_index(region_data)

                # 행정적 강도 계산
                administrative_intensity = self.calculate_administrative_intensity(
                    region_data, region_name
                )

                # 추가 메트릭 계산
                total_policies = self._count_total_policies(region_data)
                total_budget = self._calculate_total_budget(region_data)
                policy_areas = len(region_data.get("정책수행", {}))

                results.append(
                    {
                        "지역명": region_name,
                        "전략적_강도_엔트로피": round(strategic_intensity, 4),
                        "행정적_강도_가중치": round(
                            administrative_intensity * 10000, 4
                        ),  # 10000배하여 가독성 향상
                        "총_정책수": total_policies,
                        "총_예산_백만원": round(total_budget / 100, 2),  # 백만원 단위
                        "정책_분야수": policy_areas,
                        "청년인구": self._get_youth_population(region_name),
                    }
                )

        return pd.DataFrame(results)

    def _count_total_policies(self, region_data: Dict) -> int:
        """총 정책 수 계산"""
        total = 0
        for area_data in region_data.get("정책수행", {}).values():
            total += area_data.get("사업수", 0)
        return total

    def _calculate_total_budget(self, region_data: Dict) -> float:
        """총 예산 계산 (만원 단위)"""
        total = 0.0
        for area_data in region_data.get("정책수행", {}).values():
            total += area_data.get("총예산", 0)
        return total

    def analyze_policy_distribution(self, region_name: str) -> Dict:
        """
        특정 지역의 정책 분야별 분포 분석

        Args:
            region_name: 분석할 지역명

        Returns:
            분석 결과 딕셔너리
        """
        if region_name not in self.policy_data:
            return {}

        region_data = self.policy_data[region_name]
        policy_areas = region_data.get("정책수행", {})

        distribution = {}
        total_policies = 0
        total_budget = 0.0

        for area, data in policy_areas.items():
            policies = data.get("사업수", 0)
            budget = data.get("총예산", 0)

            distribution[area] = {
                "정책수": policies,
                "예산_만원": budget,
                "평균_예산_만원": round(budget / policies, 2) if policies > 0 else 0,
            }

            total_policies += policies
            total_budget += budget

        # 비율 계산
        for area in distribution:
            distribution[area]["정책수_비율"] = (
                round(distribution[area]["정책수"] / total_policies * 100, 2)
                if total_policies > 0
                else 0
            )

            distribution[area]["예산_비율"] = (
                round(distribution[area]["예산_만원"] / total_budget * 100, 2)
                if total_budget > 0
                else 0
            )

        return {
            "지역명": region_name,
            "분야별_분포": distribution,
            "총_정책수": total_policies,
            "총_예산_만원": total_budget,
        }

    def generate_policy_weight_analysis(self, region_name: str) -> pd.DataFrame:
        """
        특정 지역의 정책별 가중치 분석

        Args:
            region_name: 분석할 지역명

        Returns:
            정책별 가중치 분석 결과 DataFrame
        """
        if region_name not in self.policy_data:
            return pd.DataFrame()

        region_data = self.policy_data[region_name]
        policy_areas = region_data.get("정책수행", {})

        analysis_results = []

        for area, data in policy_areas.items():
            policies = data.get("세부사업", [])
            for policy in policies:
                policy_name = policy.get("사업명", "")
                policy_content = policy.get("주요내용", "")
                budget = policy.get("예산", 0)

                weight = self._get_policy_weight(policy_name, policy_content)

                analysis_results.append(
                    {
                        "정책분야": area,
                        "사업명": policy_name,
                        "예산_만원": budget,
                        "가중치": weight,
                        "가중치_점수": round(weight * budget, 2),
                    }
                )

        return pd.DataFrame(analysis_results)


def main():
    """메인 실행 함수"""
    # 파일 경로 설정
    policy_data_path = "../../data/policy/정책책자/경상남도_정책_최종본.json"
    youth_population_path = "../../data/policy/청년비율_시군구_기준.csv"

    try:
        # 평가자 인스턴스 생성
        evaluator = PolicyIndexEvaluator(policy_data_path, youth_population_path)

        # 전체 지역 평가
        print("=== 정량적 지표 기반 정책 평가 결과 ===")
        results_df = evaluator.evaluate_all_regions()
        print(results_df.to_string(index=False))

        print("\n=== 경상남도 정책 분야별 분포 분석 ===")
        distribution = evaluator.analyze_policy_distribution("경상남도")
        if distribution:
            for area, data in distribution["분야별_분포"].items():
                print(f"\n[{area}]")
                print(f"  정책수: {data['정책수']}개 ({data['정책수_비율']}%)")
                print(f"  예산: {data['예산_만원']}만원 ({data['예산_비율']}%)")
                print(f"  평균예산: {data['평균_예산_만원']}만원")

        print("\n=== 경상남도 정책별 가중치 분석 (상위 10개) ===")
        weight_analysis = evaluator.generate_policy_weight_analysis("경상남도")
        if not weight_analysis.empty:
            top_policies = weight_analysis.nlargest(10, "가중치_점수")
            print(
                top_policies[
                    ["사업명", "정책분야", "예산_만원", "가중치", "가중치_점수"]
                ].to_string(index=False)
            )

        # 결과를 CSV로 저장
        results_df.to_csv(
            "policy_evaluation_results.csv", index=False, encoding="utf-8-sig"
        )
        weight_analysis.to_csv(
            "policy_weight_analysis.csv", index=False, encoding="utf-8-sig"
        )

        print(f"\n결과가 저장되었습니다:")
        print(f"- policy_evaluation_results.csv")
        print(f"- policy_weight_analysis.csv")

    except FileNotFoundError as e:
        print(f"파일을 찾을 수 없습니다: {e}")
        print("현재 디렉토리에서 실행하거나 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
