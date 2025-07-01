# 전국 청년정책 종합 평가 시스템 v2
# 전략적 강도: 엔트로피 지수 (정책 분야별 균형성과 다양성)
# 행정적 강도: ln(집중도지수/재정자립도+1) (집중도지수와 재정자립도 고려)

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # Z-score 정규화에 사용


class YouthPolicyEvaluationSystemV2:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent.parent

        # 광역자치단체 목록 정의
        self.metropolitan_areas = {
            "강원도",
            "경기도",
            "경상남도",
            "경상북도",
            "광주광역시",
            "대구광역시",
            "대전광역시",
            "부산광역시",
            "서울특별시",
            "세종특별자치시",
            "울산광역시",
            "인천광역시",
            "전라남도",
            "전라북도",
            "제주특별자치도",
            "충청남도",
            "충청북도",
        }

        # 데이터 저장용
        self.policy_data = {}
        self.youth_population_data = None
        self.finance_autonomy_data = None
        self.metropolitan_budget_data = None
        self.basic_budget_data = None

    def load_all_data(self):
        """모든 필요한 데이터를 로드합니다."""
        print("=== 데이터 로딩 시작 ===")

        # 1. 정책 데이터 로드
        self._load_policy_data()

        # 2. 청년인구 데이터 로드
        self._load_youth_population_data()

        # 3. 재정자립도 데이터 로드
        self._load_finance_autonomy_data()

        # 4. 예산 데이터 로드
        self._load_budget_data()

        print("✅ 모든 데이터 로딩 완료")

    def _load_policy_data(self):
        """정책 데이터를 로드합니다."""
        policy_dir = self.base_path / "data/policy/정책책자"
        self.policy_data = {}

        for policy_file in policy_dir.glob("*_정책_최종본.json"):
            try:
                with open(policy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 파일별로 모든 지역 데이터를 policy_data에 추가
                for region_name, region_data in data.items():
                    if isinstance(region_data, dict) and "정책수행" in region_data:
                        self.policy_data[region_name] = region_data

            except Exception as e:
                print(f"정책 파일 로드 오류 {policy_file}: {e}")

        print(f"✓ 정책 데이터: {len(self.policy_data)}개 지역")

    def _load_youth_population_data(self):
        """청년인구 데이터를 로드합니다."""
        file_path = self.base_path / "data/policy/청년인구/시군구_청년비율_2023.csv"
        self.youth_population_data = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✓ 청년인구 데이터 로드: {len(self.youth_population_data)}개 지역")

    def _load_finance_autonomy_data(self):
        """재정자립도 데이터를 로드합니다."""
        file_path = (
            self.base_path / "data/policy/재정자립도/finance_autonomy_processed.csv"
        )
        self.finance_autonomy_data = pd.read_csv(file_path, encoding="utf-8-sig")
        print(f"✓ 재정자립도 데이터 로드: {len(self.finance_autonomy_data)}개 지역")

    def _load_budget_data(self):
        """예산 데이터를 로드합니다."""
        # 광역자치단체 예산
        metro_file = self.base_path / "data/budget/세출예산_광역자치단체.csv"
        self.metropolitan_budget_data = pd.read_csv(metro_file, encoding="utf-8-sig")

        # 기초자치단체 예산
        basic_file = self.base_path / "data/budget/세출예산_기초자치단체.csv"
        self.basic_budget_data = pd.read_csv(basic_file, encoding="utf-8-sig")

        print(f"✓ 광역자치단체 예산 데이터: {len(self.metropolitan_budget_data)}개")
        print(f"✓ 기초자치단체 예산 데이터: {len(self.basic_budget_data)}개")

    def is_metropolitan_area(self, region_name):
        """광역자치단체인지 판별합니다."""
        return region_name in self.metropolitan_areas

    def get_youth_population_ratio(self, region_name):
        """청년인구 비율을 조회합니다."""
        exact_match = self.youth_population_data[
            self.youth_population_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0:
            return exact_match["청년비율"].iloc[0]

        # 기본값 (전국 평균 청년비율 약 20%)
        return 0.20

    def get_finance_autonomy(self, region_name):
        """재정자립도를 조회합니다."""
        exact_match = self.finance_autonomy_data[
            self.finance_autonomy_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0:
            return exact_match["재정자립도"].iloc[0] / 100.0  # 퍼센트를 비율로 변환

        # 기본값 (전국 평균 재정자립도 약 25%)
        return 0.25

    def get_total_budget(self, region_name):
        """총예산을 조회합니다."""
        if self.is_metropolitan_area(region_name):
            # 광역자치단체 예산 조회
            budget_data = self.metropolitan_budget_data
            exact_match = budget_data[budget_data["자치단체명"] == region_name]
            if len(exact_match) > 0:
                return exact_match["세출총계"].iloc[0]
        else:
            # 기초자치단체 예산 조회
            budget_data = self.basic_budget_data
            exact_match = budget_data[budget_data["자치단체명"] == region_name]
            if len(exact_match) > 0:
                return exact_match["세출총계"].iloc[0]

        # 기본 추정값 (단위: 백만원)
        if self.is_metropolitan_area(region_name):
            return 10000000  # 광역자치단체 기본값: 1조원
        else:
            return 1000000  # 기초자치단체 기본값: 1000억원

    def get_youth_population(self, region_name):
        """특정 지역의 청년 인구수(절대값)를 조회합니다."""
        col_name = "청년인구"  # 제공해주신 컬럼명으로 수정
        exact_match = self.youth_population_data[
            self.youth_population_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0 and col_name in exact_match.columns:
            population = exact_match[col_name].iloc[0]
            # 이미 숫자 형식이므로 추가 변환 불필요
            return int(population)

        # 데이터가 없을 경우 기본값 반환
        print(
            f"  [경고] {region_name}의 청년 인구 데이터를 찾을 수 없어 기본값을 사용합니다."
        )
        return 200000 if self.is_metropolitan_area(region_name) else 10000

    def get_total_population(self, region_name):
        """특정 지역의 전체 인구수(절대값)를 조회합니다."""
        col_name = "전체인구"  # 제공해주신 컬럼명으로 수정
        exact_match = self.youth_population_data[
            self.youth_population_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0 and col_name in exact_match.columns:
            population = exact_match[col_name].iloc[0]
            return int(population)

        # 데이터가 없을 경우 기본값 반환
        print(
            f"  [경고] {region_name}의 전체 인구 데이터를 찾을 수 없어 기본값을 사용합니다."
        )
        return 1000000 if self.is_metropolitan_area(region_name) else 50000

    def calculate_youth_policy_budget(self, region_name):
        """청년정책 예산을 계산합니다."""
        if region_name not in self.policy_data:
            return 0

        region_data = self.policy_data[region_name]
        policy_execution = region_data.get("정책수행", {})

        total_budget = 0
        for category, category_data in policy_execution.items():
            if isinstance(category_data, dict):
                category_budget = 0

                # 1단계: 세부사업 내부 개별 예산 합산 시도
                if "세부사업" in category_data:
                    detailed_projects = category_data["세부사업"]
                    if isinstance(detailed_projects, list):
                        for project in detailed_projects:
                            if isinstance(project, dict) and "예산" in project:
                                budget_value = project["예산"]
                                try:
                                    # 숫자로 변환하여 합산
                                    if isinstance(budget_value, (int, float)):
                                        category_budget += float(budget_value)
                                    else:
                                        # 문자열인 경우 숫자만 추출
                                        budget_str = str(budget_value)
                                        numeric_value = float(
                                            "".join(
                                                filter(
                                                    str.isdigit,
                                                    budget_str.replace(".", ""),
                                                )
                                            )
                                        )
                                        category_budget += numeric_value
                                except (ValueError, TypeError):
                                    continue

                # 2단계: 세부사업에서 예산을 찾지 못했다면 '총예산' 사용
                if category_budget == 0 and "총예산" in category_data:
                    total_budget_value = category_data["총예산"]
                    try:
                        if isinstance(total_budget_value, (int, float)):
                            category_budget = float(total_budget_value)
                        else:
                            # 문자열인 경우 숫자만 추출
                            budget_str = str(total_budget_value)
                            category_budget = float(
                                "".join(
                                    filter(str.isdigit, budget_str.replace(".", ""))
                                )
                            )
                    except (ValueError, TypeError):
                        category_budget = 0

                total_budget += category_budget

        return total_budget

    def calculate_administrative_intensity(self, region_name):
        """
        [최종본] '1인당 예산'을 기반으로 행정적 강도를 계산합니다.
        """
        total_budget = self.get_total_budget(region_name) * 1000000
        youth_policy_budget = self.calculate_youth_policy_budget(region_name) * 1000000
        youth_population = self.get_youth_population(region_name)
        total_population = self.get_total_population(region_name)
        finance_autonomy = self.get_finance_autonomy(region_name)

        budget_per_youth = (
            youth_policy_budget / youth_population if youth_population > 0 else 0
        )
        budget_per_capita = (
            total_budget / total_population if total_population > 0 else 0
        )

        concentration_index = (
            budget_per_youth / budget_per_capita if budget_per_capita > 0 else 0
        )

        if finance_autonomy > 0:
            administrative_intensity = math.log(
                concentration_index / finance_autonomy + 1
            )
        else:
            administrative_intensity = math.log(concentration_index + 1)

        # [핵심] 반환되는 딕셔너리의 키들입니다.
        return {
            "행정적_강도": administrative_intensity,
            "집중도_지수": concentration_index,
            "청년1인당_정책예산_원": budget_per_youth,
            "전체1인당_총예산_원": budget_per_capita,
            "재정자립도": finance_autonomy,
            "총예산_백만원": total_budget / 1000000,
            "청년정책예산_백만원": youth_policy_budget / 1000000,
            "청년인구": youth_population,
            "전체인구": total_population,
        }

    def calculate_strategic_intensity(self, region_name, override_region_type=None):
        """
        [최종+예외처리] 전략적 강도를 계산합니다.
        - override_region_type ('metro'/'basic')을 통해 제주/세종의 평가 그룹을 외부에서 지정할 수 있습니다.
        - 기존의 평가 방식 및 페널티 완화 설정 기능은 그대로 유지합니다.
        """
        # ======================================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 1. 기본 평가 방식 선택 ('percentile' 또는 'z_score')
        # method = 'percentile'
        method = "z_score"

        # 2. 페널티 완화 방식 선택 ('sigmoid', 'root', 또는 'none')
        scaling_method = "sigmoid"
        # scaling_method = 'root'
        # scaling_method = 'none'

        # 3. 페널티 완화 강도 조절
        SIGMOID_K = 5
        ROOT_N = 2
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # ======================================================================
        policy_categories = ["일자리", "주거", "교육", "복지·문화", "참여·권리"]

        if not hasattr(self, "_category_stats_df"):
            print("\n[최초 실행] 모든 지역의 영역별 정책 수를 계산하고 캐싱합니다...")
            all_regions_data = []
            for name, data in self.policy_data.items():
                region_row = {
                    "region_name": name,
                    "is_metro": self.is_metropolitan_area(name),
                }
                policy_execution = data.get("정책수행", {})
                for category in policy_categories:
                    count = 0
                    category_data = policy_execution.get(category, {})
                    if isinstance(category_data, dict):
                        if (
                            "사업수" in category_data
                            and isinstance(category_data["사업수"], (int, float))
                            and category_data["사업수"] > 0
                        ):
                            count = int(category_data["사업수"])
                        elif "세부사업" in category_data and isinstance(
                            category_data.get("세부사업"), list
                        ):
                            count = len(category_data["세부사업"])
                    region_row[f"{category}_정책수"] = count
                all_regions_data.append(region_row)
            self._category_stats_df = pd.DataFrame(all_regions_data)
            print("✓ 영역별 데이터 캐싱 완료.")

        stats_df = self._category_stats_df
        try:
            current_region_data = stats_df.loc[
                stats_df["region_name"] == region_name
            ].iloc[0]
        except IndexError:
            result = {"전략적_강도": 0, "엔트로피": 0, "정규화_엔트로피": 0}
            for cat in policy_categories:
                result[f"{cat}_점수"], result[f"{cat}_정책수"] = 0, 0
            return result

        # [핵심 수정] 외부에서 지정한 평가 그룹을 우선적으로 사용
        if override_region_type == "metro":
            is_metro = True
        elif override_region_type == "basic":
            is_metro = False
        else:
            is_metro = self.is_metropolitan_area(region_name)

        group_df = stats_df[stats_df["is_metro"] == is_metro]

        # 이하 점수 계산 로직은 제공해주신 코드와 완전히 동일하게 유지
        category_total_score = 0
        final_result = {}
        for category in policy_categories:
            count_col, score_col = f"{category}_정책수", f"{category}_점수"
            current_value = current_region_data[count_col]
            distribution = group_df[count_col]

            raw_score = 0.0
            if method == "percentile":
                sorted_dist = np.sort(distribution.values)
                if len(sorted_dist) > 0:
                    raw_score = np.searchsorted(
                        sorted_dist, current_value, side="right"
                    ) / len(sorted_dist)
            elif method == "z_score":
                mean, std = distribution.mean(), distribution.std()
                if std > 0:
                    raw_score = stats.norm.cdf((current_value - mean) / std)
                elif len(distribution) > 0:
                    raw_score = 0.5

            scaled_score = raw_score
            if scaling_method == "sigmoid":
                scaled_score = 1 / (1 + math.exp(-SIGMOID_K * (raw_score - 0.5)))
            elif scaling_method == "root":
                scaled_score = raw_score ** (1 / ROOT_N)

            final_result[score_col] = scaled_score
            final_result[count_col] = current_value
            category_total_score += scaled_score

        policy_counts = {
            cat: final_result[f"{cat}_정책수"] for cat in policy_categories
        }
        total_policies = sum(policy_counts.values())
        entropy = 0.0
        active_categories = sum(1 for c in policy_counts.values() if c > 0)
        if total_policies > 0:
            for count in policy_counts.values():
                if count > 0:
                    entropy -= (count / total_policies) * math.log2(
                        count / total_policies
                    )

        entropy_score = 0.0
        if active_categories > 1:
            max_entropy = math.log2(active_categories)
            if max_entropy > 0:
                entropy_score = entropy / max_entropy

        strategic_intensity = category_total_score + entropy_score

        final_result.update(
            {
                "전략적_강도": strategic_intensity,
                "엔트로피": entropy,
                "정규화_엔트로피": entropy_score,
            }
        )
        return final_result

    def evaluate_all_regions(self):

        print("\n=== 전국 청년정책 평가 시작 ===")
        results = []
        special_dual_role_regions = {"제주특별자치도", "세종특별자치시"}

        if self.policy_data:
            _ = self.calculate_strategic_intensity(list(self.policy_data.keys())[0])

        for region_name in self.policy_data.keys():
            print(f"평가 중: {region_name}")

            if region_name in special_dual_role_regions:
                print(
                    f"  -> 특별자치시/도({region_name}) 감지. 광역/기초 이중 평가를 수행합니다."
                )

                admin_result = self.calculate_administrative_intensity(region_name)

                # (1) 광역으로서 평가
                strategic_metro = self.calculate_strategic_intensity(
                    region_name, override_region_type="metro"
                )
                result_metro = {
                    "지역명": region_name,
                    "지역유형": "광역자치단체",
                    **admin_result,  # admin_result의 모든 내용을 여기에 복사
                    **strategic_metro,  # strategic_metro의 모든 내용을 여기에 복사
                }
                results.append(result_metro)

                # (2) 기초로서 평가
                strategic_basic = self.calculate_strategic_intensity(
                    region_name, override_region_type="basic"
                )
                result_basic = {
                    "지역명": region_name,
                    "지역유형": "기초자치단체",
                    **admin_result,  # 행정적 강도 결과는 동일하게 사용
                    **strategic_basic,
                }
                results.append(result_basic)

            else:  # 일반 지역의 경우
                admin_result = self.calculate_administrative_intensity(region_name)
                strategic_result = self.calculate_strategic_intensity(region_name)
                region_type = (
                    "광역자치단체"
                    if self.is_metropolitan_area(region_name)
                    else "기초자치단체"
                )

                result = {
                    "지역명": region_name,
                    "지역유형": region_type,
                    **admin_result,
                    **strategic_result,
                }
                results.append(result)

        print("\n✅ 모든 지역 평가 완료")
        return results

    def calculate_comprehensive_scores(self, results):
        """종합점수를 계산합니다."""
        df = pd.DataFrame(results)

        if len(df) == 0:
            return df

        # 정규화 (Min-Max Scaling)
        admin_min, admin_max = df["행정적_강도"].min(), df["행정적_강도"].max()
        strategic_min, strategic_max = df["전략적_강도"].min(), df["전략적_강도"].max()

        if admin_max > admin_min:
            df["행정적_강도_정규화"] = (df["행정적_강도"] - admin_min) / (
                admin_max - admin_min
            )
        else:
            df["행정적_강도_정규화"] = 0

        if strategic_max > strategic_min:
            df["전략적_강도_정규화"] = (df["전략적_강도"] - strategic_min) / (
                strategic_max - strategic_min
            )
        else:
            df["전략적_강도_정규화"] = 0

        # 종합점수 계산 (50:50 비율)
        df["종합점수"] = (df["행정적_강도_정규화"] + df["전략적_강도_정규화"]) / 2

        return df

    def add_rankings(self, df):
        """종합점수를 바탕으로 순위를 추가합니다."""
        print(f"\n📊 총 {len(df)}개 지역에 순위 추가 중...")

        # 1. 전체 순위 (종합점수 기준 내림차순)
        df = df.sort_values("종합점수", ascending=False).reset_index(drop=True)
        df["전체순위"] = range(1, len(df) + 1)

        # 2. 지역유형별 순위
        df["광역순위"] = 0
        df["기초순위"] = 0

        # 광역자치단체 순위
        metro_df = df[df["지역유형"] == "광역자치단체"].copy()
        metro_df = metro_df.sort_values("종합점수", ascending=False).reset_index(
            drop=True
        )
        metro_df["광역순위"] = range(1, len(metro_df) + 1)

        # 기초자치단체 순위
        basic_df = df[df["지역유형"] == "기초자치단체"].copy()
        basic_df = basic_df.sort_values("종합점수", ascending=False).reset_index(
            drop=True
        )
        basic_df["기초순위"] = range(1, len(basic_df) + 1)

        # 순위 정보를 원본 데이터프레임에 병합
        for idx, row in metro_df.iterrows():
            df.loc[df["지역명"] == row["지역명"], "광역순위"] = row["광역순위"]

        for idx, row in basic_df.iterrows():
            df.loc[df["지역명"] == row["지역명"], "기초순위"] = row["기초순위"]

        # 3. 컬럼 순서 재정렬 (순위 컬럼들을 앞쪽으로)
        columns = ["전체순위", "광역순위", "기초순위"] + [
            col for col in df.columns if col not in ["전체순위", "광역순위", "기초순위"]
        ]
        df = df[columns]

        print("✅ 순위 추가 완료")
        return df

    def save_results(self, df):
        """
        [최종 확장본]
        1. 기존 방식대로 평가 결과를 '광역'과 '기초'로 나누어 각각 저장합니다.
        2. 추가적으로, 광역 점수를 기초에 반영한 최종 연계 점수를 계산하여 별도의 단일 파일로 저장합니다.
        """
        if df.empty:
            print("평가 결과가 없어 파일을 저장하지 않습니다.")
            return

        # ======================================================================
        # 1. 기존 로직: 광역/기초 분리하여 각각 파일로 저장 (유지)
        # ======================================================================
        metro_df = df[df["지역유형"] == "광역자치단체"].copy()
        basic_df = df[df["지역유형"] == "기초자치단체"].copy()
        print(f"\n결과 분리 중: 광역 {len(metro_df)}개, 기초 {len(basic_df)}개")

        base_output_path = (
            self.base_path / "policy_evaluation/evaluation_results_index/evaluation-3"
        )
        base_output_path.mkdir(parents=True, exist_ok=True)

        metro_csv_file = base_output_path / "광역_청년정책_종합평가결과.csv"
        basic_csv_file = base_output_path / "기초_청년정책_종합평가결과.csv"

        if not metro_df.empty:
            metro_df.to_csv(metro_csv_file, index=False, encoding="utf-8-sig")
        if not basic_df.empty:
            basic_df.to_csv(basic_csv_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ 1차 개별 평가 결과 저장 완료:")
        print(f"   [광역] {metro_csv_file}")
        print(f"   [기초] {basic_csv_file}")

        # ======================================================================
        # 2. 새로운 로직: 광역-기초 연계 점수 계산 및 추가 파일 저장
        # ======================================================================
        print("\n=== 광역-기초 연계 점수 계산 시작... ===")

        # (2-1) 광역 점수 조회용 맵 생성
        #       여기서의 '종합점수'는 광역/기초가 모두 포함된 전체 데이터에서 정규화된 점수입니다.
        metro_scores_map = metro_df.set_index("지역명")["종합점수"]

        # (2-2) 기초 데이터프레임에 소속 광역 매핑
        def get_metro_region(basic_region_name):
            # 서울특별시, 부산광역시 등 광역/특별시 이름으로 시작하는지 확인
            for metro_name in self.metropolitan_areas:
                if basic_region_name.startswith(metro_name[:2]):
                    # "경상", "전라", "충청" 이름 충돌 방지
                    if metro_name.endswith("남도") and "북도" in basic_region_name:
                        continue
                    if metro_name.endswith("북도") and "남도" in basic_region_name:
                        continue
                    return metro_name
            return None

        basic_df["소속_광역"] = basic_df["지역명"].apply(get_metro_region)

        # 제주/세종은 소속 광역이 자기 자신
        basic_df.loc[basic_df["지역명"] == "세종특별자치시", "소속_광역"] = (
            "세종특별자치시"
        )
        basic_df.loc[basic_df["지역명"] == "제주특별자치도", "소속_광역"] = (
            "제주특별자치도"
        )

        # (2-3) 소속 광역의 점수를 기초 데이터프레임에 추가
        basic_df["광역_종합점수"] = (
            basic_df["소속_광역"].map(metro_scores_map).fillna(0)
        )

        # (2-4) 최종 연계 점수 계산 (가중 평균)
        basic_weight = 0.7  # 기초 자체 노력 가중치
        metro_weight = 0.3  # 광역 지원 노력 가중치

        # '종합점수'는 basic_df의 개별 종합점수를 의미
        basic_df["최종_연계점수"] = (basic_df["종합점수"] * basic_weight) + (
            basic_df["광역_종합점수"] * metro_weight
        )

        # (2-5) 최종 순위 매기기 및 컬럼 정리
        final_linked_df = basic_df.sort_values(
            "최종_연계점수", ascending=False
        ).reset_index(drop=True)
        final_linked_df["최종순위"] = final_linked_df.index + 1

        output_columns = [
            "최종순위",
            "지역명",
            "소속_광역",
            "최종_연계점수",
            "종합점수",
            "광역_종합점수",
            "행정적_강도",
            "전략적_강도",
            "재정자립도",
            "청년1인당_정책예산_원",
        ]
        # 없는 컬럼이 있어도 오류나지 않도록 처리
        final_linked_df = final_linked_df[
            [col for col in output_columns if col in final_linked_df.columns]
        ]

        # (2-6) 최종 연계 분석 결과 파일 저장
        linked_csv_file = base_output_path / "기초_최종평가결과(광역연계).csv"
        final_linked_df.to_csv(linked_csv_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ 2차 연계 평가 결과 저장 완료:")
        print(f"   [연계] {linked_csv_file}")

        # 이전 버전과의 호환성을 위해 기존 파일 경로 반환
        return metro_csv_file, None, basic_csv_file, None

    def print_summary(self, df):
        """결과 요약을 출력합니다."""
        print(f"\n=== 평가 결과 요약 ===")
        print(f"총 평가 지역: {len(df)}개")

        # 지역 유형별 통계
        type_stats = df.groupby("지역유형").size()
        print(f"\n📍 지역 유형별 분포:")
        for region_type, count in type_stats.items():
            print(f"   {region_type}: {count}개")

        # 상위 10개 지역 (종합점수 기준)
        print(f"\n🏆 **청년정책 종합평가 순위 결과**")
        print("=" * 60)

        print(f"\n📍 **전체 순위 TOP 10**")
        print("-" * 40)
        top10 = df.head(10)[["전체순위", "지역명", "지역유형", "종합점수"]]
        for _, row in top10.iterrows():
            type_icon = "🏛️" if row["지역유형"] == "광역자치단체" else "🏘️"
            print(
                f"{row['전체순위']:2d}위. {type_icon} {row['지역명']} ({row['종합점수']:.4f})"
            )

        # 광역자치단체 상위 5개
        print(f"\n🏛️ **광역자치단체 순위 TOP 5**")
        print("-" * 40)
        metro_top5 = df[df["지역유형"] == "광역자치단체"].head(5)[
            ["광역순위", "지역명", "종합점수"]
        ]
        for _, row in metro_top5.iterrows():
            print(f"{row['광역순위']:2d}위. {row['지역명']} ({row['종합점수']:.4f})")

        # 기초자치단체 상위 10개
        print(f"\n🏘️ **기초자치단체 순위 TOP 10**")
        print("-" * 40)
        basic_top10 = df[df["지역유형"] == "기초자치단체"].head(10)[
            ["기초순위", "지역명", "종합점수"]
        ]
        for _, row in basic_top10.iterrows():
            print(f"{row['기초순위']:2d}위. {row['지역명']} ({row['종합점수']:.4f})")

        print(f"\n📊 **통계 요약**")
        print("-" * 40)
        print(f"• 총 평가 지역: {len(df):,}개")
        print(f"• 광역자치단체: {len(df[df['지역유형'] == '광역자치단체']):,}개")
        print(f"• 기초자치단체: {len(df[df['지역유형'] == '기초자치단체']):,}개")
        print(f"• 평균 종합점수: {df['종합점수'].mean():.4f}")
        print(f"• 최고 종합점수: {df['종합점수'].max():.4f} ({df.iloc[0]['지역명']})")
        print(f"• 최저 종합점수: {df['종합점수'].min():.4f} ({df.iloc[-1]['지역명']})")

    def run_evaluation(self):
        """전체 평가 프로세스를 실행합니다."""
        print("=== 전국 청년정책 종합 평가 시스템 v2 시작 ===")

        # 1. 데이터 로딩
        self.load_all_data()

        # 2. 모든 지역 평가
        results = self.evaluate_all_regions()

        # 3. 종합점수 계산
        df = self.calculate_comprehensive_scores(results)

        # 4. 순위 추가
        df = self.add_rankings(df)

        # 5. 결과 저장
        self.save_results(df)

        # 6. 요약 출력
        self.print_summary(df)

        return df


if __name__ == "__main__":
    evaluator = YouthPolicyEvaluationSystemV2()
    results_df = evaluator.run_evaluation()
