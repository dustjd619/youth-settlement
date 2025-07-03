# Sigmoid 함수만 활용하는 청년정책 평가 시스템 (Z-Score 제외)
# migration_plot/eval-3_result에 결과 저장

import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # GUI 없이 파일 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class YouthPolicyEvaluationSigmoidOnly:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent

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

        # 결과 저장 디렉토리 생성
        self.result_dir = self.base_path / "migration_plot/eval-3_result"
        os.makedirs(self.result_dir, exist_ok=True)

        # 데이터 저장용
        self.policy_data = {}
        self.youth_population_data = None
        self.finance_autonomy_data = None
        self.metropolitan_budget_data = None
        self.basic_budget_data = None
        self.migration_data = None

        # 분석 기간 설정 (2023년 8월 ~ 2024년 7월)
        self.start_year_month = 202308
        self.end_year_month = 202407

    def load_all_data(self):
        """모든 필요한 데이터를 로드합니다."""
        print("=== 데이터 로딩 시작 (Sigmoid Only) ===")

        # 1. 정책 데이터 로드
        self._load_policy_data()

        # 2. 청년인구 데이터 로드
        self._load_youth_population_data()

        # 3. 재정자립도 데이터 로드
        self._load_finance_autonomy_data()

        # 4. 예산 데이터 로드
        self._load_budget_data()

        # 5. 마이그레이션 데이터 로드
        self._load_migration_data()

        print("✅ 모든 데이터 로딩 완료")

    def _load_policy_data(self):
        """정책 데이터를 로드합니다."""
        policy_dir = self.base_path / "data/policy/정책책자"
        self.policy_data = {}

        for policy_file in policy_dir.glob("*_정책_최종본.json"):
            try:
                with open(policy_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

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
        metro_file = self.base_path / "data/budget/세출예산_광역자치단체.csv"
        self.metropolitan_budget_data = pd.read_csv(metro_file, encoding="utf-8-sig")

        basic_file = self.base_path / "data/budget/세출예산_기초자치단체.csv"
        self.basic_budget_data = pd.read_csv(basic_file, encoding="utf-8-sig")

        print(f"✓ 광역자치단체 예산 데이터: {len(self.metropolitan_budget_data)}개")
        print(f"✓ 기초자치단체 예산 데이터: {len(self.basic_budget_data)}개")

    def _load_migration_data(self):
        """마이그레이션 데이터를 로드하고 순유입률을 계산합니다."""
        print("📊 마이그레이션 데이터 로드 중...")

        migration_dir = self.base_path / "data/migration/청년 인구 이동량_consolidated"
        if not migration_dir.exists():
            print("❌ 마이그레이션 데이터 디렉토리를 찾을 수 없습니다.")
            return False

        # 분석 기간에 해당하는 파일들 찾기
        target_files = []
        current_ym = self.start_year_month

        while current_ym <= self.end_year_month:
            file_path = migration_dir / f"youth_total_migration_{current_ym}.csv"
            if file_path.exists():
                target_files.append(file_path)

            # 다음 월로 증가
            current_month = current_ym % 100
            current_year = current_ym // 100

            if current_month == 12:
                current_ym = (current_year + 1) * 100 + 1
            else:
                current_ym = current_year * 100 + (current_month + 1)

        if not target_files:
            print("❌ 분석 기간에 해당하는 마이그레이션 데이터를 찾을 수 없습니다.")
            return False

        # 순이동 계산
        self._preprocess_migration_data(target_files)
        print(f"✓ 마이그레이션 데이터 로드: {len(self.migration_data)}개 지역")

    def _preprocess_migration_data(self, target_files):
        """파일별로 각 지역의 컬럼합(전입), row합(전출) 누적 방식으로 순이동 계산"""
        # 모든 지역명 집합
        all_regions = set()
        for file in target_files:
            df = pd.read_csv(file, encoding="utf-8-sig")
            all_regions.update(df.columns[1:])  # 첫 컬럼은 지역명(row)
            all_regions.update(df.iloc[:, 0].unique())
        all_regions = sorted(all_regions)

        # 누적용 dict
        inflow_dict = {region: 0 for region in all_regions}
        outflow_dict = {region: 0 for region in all_regions}

        # 파일별로 누적
        for file in target_files:
            df = pd.read_csv(file, encoding="utf-8-sig")
            df = df.fillna(0)

            # 컬럼: [지역명, ...]
            col_regions = df.columns[1:]
            row_regions = df.iloc[:, 0]

            # 전입: 각 지역별 컬럼 합
            for region in col_regions:
                inflow_dict[region] += df[region].sum()

            # 전출: 각 지역별 row 합
            for idx, region in enumerate(row_regions):
                outflow_dict[region] += df.iloc[idx, 1:].sum()

        # 결과 DataFrame
        result = []
        for region in all_regions:
            inflow = inflow_dict[region]
            outflow = outflow_dict[region]
            net_migration = inflow - outflow

            result.append(
                {
                    "지역명": region,
                    "전입": inflow,
                    "전출": outflow,
                    "순이동": net_migration,
                }
            )

        self.migration_data = pd.DataFrame(result)

    def get_net_migration_rate(self, region_name):
        """특정 지역의 순유입률(청년인구 대비 %)을 조회합니다."""
        if self.migration_data is None:
            return 0.0

        def normalize_region_name(name):
            """지역명 정규화"""
            if pd.isna(name):
                return ""
            name = str(name).strip()

            # 특별/광역시 제거
            prefixes = [
                "서울특별시",
                "부산광역시",
                "대구광역시",
                "인천광역시",
                "광주광역시",
                "대전광역시",
                "울산광역시",
                "세종특별자치시",
                "경기도",
                "강원도",
                "충청북도",
                "충청남도",
                "전라북도",
                "전라남도",
                "경상북도",
                "경상남도",
                "제주특별자치도",
            ]

            for prefix in prefixes:
                if name.startswith(prefix):
                    name = name[len(prefix) :].strip()
                    break

            return name

        # 지역명 정규화하여 매칭
        normalized_region = normalize_region_name(region_name)

        # 마이그레이션 데이터에서 해당 지역 찾기
        migration_match = None
        for _, row in self.migration_data.iterrows():
            if normalize_region_name(row["지역명"]) == normalized_region:
                migration_match = row
                break

        if migration_match is not None:
            net_migration = migration_match["순이동"]
            youth_pop = self.get_youth_population(region_name)

            # 순유입률 = (순이동 / 청년인구) * 100
            if youth_pop > 0:
                return (net_migration / youth_pop) * 100

        return 0.0

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

        return 0.20

    def get_finance_autonomy(self, region_name):
        """재정자립도를 조회합니다."""
        exact_match = self.finance_autonomy_data[
            self.finance_autonomy_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0:
            return exact_match["재정자립도"].iloc[0] / 100.0

        return 0.25

    def get_total_budget(self, region_name):
        """총예산을 조회합니다."""
        if self.is_metropolitan_area(region_name):
            budget_data = self.metropolitan_budget_data
            exact_match = budget_data[budget_data["자치단체명"] == region_name]
            if len(exact_match) > 0:
                return exact_match["세출총계"].iloc[0]
        else:
            budget_data = self.basic_budget_data
            exact_match = budget_data[budget_data["자치단체명"] == region_name]
            if len(exact_match) > 0:
                return exact_match["세출총계"].iloc[0]

        if self.is_metropolitan_area(region_name):
            return 10000000  # 광역자치단체 기본값: 1조원
        else:
            return 1000000  # 기초자치단체 기본값: 1000억원

    def get_youth_population(self, region_name):
        """특정 지역의 청년 인구수(절대값)를 조회합니다."""
        col_name = "청년인구"
        exact_match = self.youth_population_data[
            self.youth_population_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0 and col_name in exact_match.columns:
            population = exact_match[col_name].iloc[0]
            return int(population)

        return 200000 if self.is_metropolitan_area(region_name) else 10000

    def get_total_population(self, region_name):
        """특정 지역의 전체 인구수(절대값)를 조회합니다."""
        col_name = "전체인구"
        exact_match = self.youth_population_data[
            self.youth_population_data["지자체명"] == region_name
        ]
        if len(exact_match) > 0 and col_name in exact_match.columns:
            population = exact_match[col_name].iloc[0]
            return int(population)

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
                                    if isinstance(budget_value, (int, float)):
                                        category_budget += float(budget_value)
                                    else:
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
        """행정적 강도를 계산합니다."""
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
        전략적 강도를 계산합니다. (Percentile + Sigmoid 적용, Z-Score 제외)
        """
        # 정책 카테고리 정의
        policy_categories = ["일자리", "주거", "교육", "복지·문화", "참여·권리"]

        # Sigmoid 강도 조절 (K값이 클수록 더 가파른 S-curve)
        SIGMOID_K = 5

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

        # 외부에서 지정한 평가 그룹을 우선적으로 사용
        if override_region_type == "metro":
            is_metro = True
        elif override_region_type == "basic":
            is_metro = False
        else:
            is_metro = self.is_metropolitan_area(region_name)

        group_df = stats_df[stats_df["is_metro"] == is_metro]

        # Percentile + Sigmoid 방식으로 점수 계산
        category_total_score = 0
        final_result = {}
        for category in policy_categories:
            count_col, score_col = f"{category}_정책수", f"{category}_점수"
            current_value = current_region_data[count_col]
            distribution = group_df[count_col]

            # Percentile 계산
            raw_score = 0.0
            sorted_dist = np.sort(distribution.values)
            if len(sorted_dist) > 0:
                raw_score = np.searchsorted(
                    sorted_dist, current_value, side="right"
                ) / len(sorted_dist)

            # Sigmoid 적용
            scaled_score = 1 / (1 + math.exp(-SIGMOID_K * (raw_score - 0.5)))

            final_result[score_col] = scaled_score
            final_result[count_col] = current_value
            category_total_score += scaled_score

        # 엔트로피 계산
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
        """모든 지역을 평가합니다."""
        print("\n=== 전국 청년정책 평가 시작 (Sigmoid Only) ===")
        results = []
        special_dual_role_regions = {"제주특별자치도", "세종특별자치시"}

        # 전략적 강도 계산을 위한 초기화
        if self.policy_data:
            _ = self.calculate_strategic_intensity(list(self.policy_data.keys())[0])

        for region_name in self.policy_data.keys():
            print(f"평가 중: {region_name}")

            if region_name in special_dual_role_regions:
                admin_result = self.calculate_administrative_intensity(region_name)
                # 순유입률 계산
                net_migration_rate = self.get_net_migration_rate(region_name)

                # 광역으로서 평가
                strategic_metro = self.calculate_strategic_intensity(
                    region_name, override_region_type="metro"
                )
                result_metro = {
                    "지역명": region_name,
                    "지역유형": "광역자치단체",
                    "순이동률_인구대비": net_migration_rate,
                    **admin_result,
                    **strategic_metro,
                }
                results.append(result_metro)

                # 기초로서 평가
                strategic_basic = self.calculate_strategic_intensity(
                    region_name, override_region_type="basic"
                )
                result_basic = {
                    "지역명": region_name,
                    "지역유형": "기초자치단체",
                    "순이동률_인구대비": net_migration_rate,
                    **admin_result,
                    **strategic_basic,
                }
                results.append(result_basic)

            else:
                admin_result = self.calculate_administrative_intensity(region_name)
                strategic_result = self.calculate_strategic_intensity(region_name)
                region_type = (
                    "광역자치단체"
                    if self.is_metropolitan_area(region_name)
                    else "기초자치단체"
                )

                # 순유입률 계산
                net_migration_rate = self.get_net_migration_rate(region_name)

                result = {
                    "지역명": region_name,
                    "지역유형": region_type,
                    "순이동률_인구대비": net_migration_rate,
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
            df["행정적_강도_정규화"] = 0.5

        if strategic_max > strategic_min:
            df["전략적_강도_정규화"] = (df["전략적_강도"] - strategic_min) / (
                strategic_max - strategic_min
            )
        else:
            df["전략적_강도_정규화"] = 0.5

        # 종합점수 계산 (50:50 비율)
        df["종합점수"] = (df["행정적_강도_정규화"] + df["전략적_강도_정규화"]) / 2

        return df

    def add_rankings(self, df):
        """종합점수를 바탕으로 순위를 추가합니다."""
        print(f"\n📊 총 {len(df)}개 지역에 순위 추가 중...")

        # 전체 순위
        df = df.sort_values("종합점수", ascending=False).reset_index(drop=True)
        df["전체순위"] = range(1, len(df) + 1)

        # 지역유형별 순위
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

        # 컬럼 순서 조정
        columns = ["전체순위", "광역순위", "기초순위"] + [
            col for col in df.columns if col not in ["전체순위", "광역순위", "기초순위"]
        ]
        df = df[columns]

        print("✅ 순위 추가 완료")
        return df

    def save_results(self, df):
        """결과를 migration_plot/eval-3_result에 저장합니다."""
        if df.empty:
            print("평가 결과가 없어 파일을 저장하지 않습니다.")
            return

        # 광역/기초 분리
        metro_df = df[df["지역유형"] == "광역자치단체"].copy()
        basic_df = df[df["지역유형"] == "기초자치단체"].copy()
        print(f"\n결과 분리 중: 광역 {len(metro_df)}개, 기초 {len(basic_df)}개")

        # 파일 저장
        metro_csv_file = self.result_dir / "sigmoid_only_광역_청년정책_종합평가결과.csv"
        basic_csv_file = self.result_dir / "sigmoid_only_기초_청년정책_종합평가결과.csv"

        if not metro_df.empty:
            metro_df.to_csv(metro_csv_file, index=False, encoding="utf-8-sig")
        if not basic_df.empty:
            basic_df.to_csv(basic_csv_file, index=False, encoding="utf-8-sig")

        print(f"\n✅ Sigmoid Only 평가 결과 저장 완료:")
        print(f"   [광역] {metro_csv_file}")
        print(f"   [기초] {basic_csv_file}")

        # 광역-기초 연계 점수 계산
        metro_scores_map = metro_df.set_index("지역명")["종합점수"]

        def get_metro_region(basic_region_name):
            for metro_name in self.metropolitan_areas:
                if basic_region_name.startswith(metro_name[:2]):
                    if metro_name.endswith("남도") and "북도" in basic_region_name:
                        continue
                    if metro_name.endswith("북도") and "남도" in basic_region_name:
                        continue
                    return metro_name
            return None

        basic_df["소속_광역"] = basic_df["지역명"].apply(get_metro_region)
        basic_df.loc[basic_df["지역명"] == "세종특별자치시", "소속_광역"] = (
            "세종특별자치시"
        )
        basic_df.loc[basic_df["지역명"] == "제주특별자치도", "소속_광역"] = (
            "제주특별자치도"
        )

        basic_df["광역_종합점수"] = (
            basic_df["소속_광역"].map(metro_scores_map).fillna(0)
        )

        # 최종 연계 점수 계산
        basic_weight = 0.5
        metro_weight = 0.5
        basic_df["최종_연계점수"] = (basic_df["종합점수"] * basic_weight) + (
            basic_df["광역_종합점수"] * metro_weight
        )

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
        final_linked_df = final_linked_df[
            [col for col in output_columns if col in final_linked_df.columns]
        ]

        linked_csv_file = (
            self.result_dir / "sigmoid_only_기초_최종평가결과(광역연계).csv"
        )
        final_linked_df.to_csv(linked_csv_file, index=False, encoding="utf-8-sig")

        print(f"   [연계] {linked_csv_file}")

        return metro_csv_file, basic_csv_file, linked_csv_file

    def create_comprehensive_analysis_plots(self, df):
        """종합 분석 플롯 생성 (Sigmoid Only 버전)"""
        print("\n📊 종합 분석 플롯 생성 중...")

        # 4개 서브플롯 생성 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 종합점수 vs 행정적 강도 산점도
        colors = ["red" if x == "광역자치단체" else "blue" for x in df["지역유형"]]
        axes[0, 0].scatter(df["행정적_강도"], df["종합점수"], c=colors, alpha=0.6, s=60)

        # 회귀선 추가
        if len(df) > 2:
            valid_data = df[["행정적_강도", "종합점수"]].dropna()
            if len(valid_data) > 2:
                z = np.polyfit(valid_data["행정적_강도"], valid_data["종합점수"], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(
                    valid_data["행정적_강도"],
                    p(valid_data["행정적_강도"]),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                )

                # 상관계수 계산 및 표시
                corr_coef, p_value = stats.pearsonr(
                    valid_data["행정적_강도"], valid_data["종합점수"]
                )
                significance = (
                    "***"
                    if p_value < 0.001
                    else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                )

                axes[0, 0].text(
                    0.05,
                    0.95,
                    f"r = {corr_coef:.3f}{significance}",
                    transform=axes[0, 0].transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

        axes[0, 0].set_xlabel("행정적 강도")
        axes[0, 0].set_ylabel("종합점수")
        axes[0, 0].set_title("행정적 강도 vs 종합점수\n(Percentile + Sigmoid 적용)")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 전략적 강도 vs 청년1인당 정책예산 산점도
        axes[0, 1].scatter(
            df["전략적_강도"], df["청년1인당_정책예산_원"], c=colors, alpha=0.6, s=60
        )

        # 상위/하위 5개 지역 라벨링
        sorted_data = df.sort_values("청년1인당_정책예산_원", ascending=False)
        top5 = sorted_data.head(5)
        bottom5 = sorted_data.tail(5)

        for _, row in pd.concat([top5, bottom5]).iterrows():
            axes[0, 1].annotate(
                row["지역명"],
                (row["전략적_강도"], row["청년1인당_정책예산_원"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    facecolor="white",
                    edgecolor="blue",
                    alpha=0.7,
                    boxstyle="round,pad=0.3",
                ),
            )

        axes[0, 1].set_xlabel("전략적 강도 (Sigmoid 적용)")
        axes[0, 1].set_ylabel("청년1인당 정책예산 (원)")
        axes[0, 1].set_title("전략적 강도 vs 청년1인당 정책예산")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 재정자립도 vs 종합점수 산점도
        axes[1, 0].scatter(df["재정자립도"], df["종합점수"], c=colors, alpha=0.6, s=60)
        axes[1, 0].set_xlabel("재정자립도")
        axes[1, 0].set_ylabel("종합점수")
        axes[1, 0].set_title("재정자립도 vs 종합점수")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 지역유형별 종합점수 분포 박스플롯
        metro_data = df[df["지역유형"] == "광역자치단체"]["종합점수"]
        basic_data = df[df["지역유형"] == "기초자치단체"]["종합점수"]

        box_data = [metro_data, basic_data]
        labels = ["광역자치단체", "기초자치단체"]

        axes[1, 1].boxplot(box_data, labels=labels)
        axes[1, 1].set_ylabel("종합점수")
        axes[1, 1].set_title("지역유형별 종합점수 분포")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        # 범례 추가
        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color="red", label="광역자치단체")
        blue_patch = mpatches.Patch(color="blue", label="기초자치단체")
        fig.legend(
            handles=[red_patch, blue_patch],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

        plt.suptitle(
            "청년정책 평가 종합 분석 (Percentile + Sigmoid 적용)", fontsize=16, y=0.98
        )
        plt.tight_layout()

        # 저장
        comprehensive_path = self.result_dir / "sigmoid_only_comprehensive_analysis.png"
        try:
            plt.savefig(comprehensive_path, dpi=300, bbox_inches="tight")
            print(f"✅ 종합 분석 플롯 저장 완료: {comprehensive_path}")
        except Exception as e:
            print(f"❌ 종합 분석 플롯 저장 실패: {e}")
        plt.close()

    def create_policy_effectiveness_analysis(self, df):
        """정책 효과성 분석 플롯 생성"""
        print("\n📊 정책 효과성 분석 플롯 생성 중...")

        # 3개 서브플롯 생성 (1x3)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 1. 정책 분야별 점수 분석 (방사형 차트)
        policy_categories = ["일자리", "주거", "교육", "복지·문화", "참여·권리"]
        category_cols = [f"{cat}_점수" for cat in policy_categories]

        # 상위 5개 지역의 평균 vs 하위 5개 지역의 평균
        top5_regions = df.nlargest(5, "종합점수")
        bottom5_regions = df.nsmallest(5, "종합점수")

        # 해당 컬럼들이 존재하는지 확인
        existing_cols = [col for col in category_cols if col in df.columns]

        if existing_cols:
            top5_avg = top5_regions[existing_cols].mean()
            bottom5_avg = bottom5_regions[existing_cols].mean()

            # 막대 차트로 표시
            x_pos = np.arange(len(existing_cols))
            width = 0.35

            axes[0].bar(
                x_pos - width / 2,
                top5_avg,
                width,
                label="상위 5개 지역 평균",
                color="steelblue",
                alpha=0.8,
            )
            axes[0].bar(
                x_pos + width / 2,
                bottom5_avg,
                width,
                label="하위 5개 지역 평균",
                color="lightcoral",
                alpha=0.8,
            )

            axes[0].set_xlabel("정책 분야")
            axes[0].set_ylabel("평균 점수 (Sigmoid 적용)")
            axes[0].set_title("정책 분야별 효과성 비교\n(상위 vs 하위 지역)")
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(
                [col.replace("_점수", "") for col in existing_cols], rotation=45
            )
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

        # 2. 정책 강도 균형 분석
        balance = abs(df["행정적_강도_정규화"] - df["전략적_강도_정규화"])
        df_temp = df.copy()
        df_temp["강도균형"] = balance

        # 균형 수준별 그룹핑
        df_temp["균형등급"] = pd.cut(
            balance,
            bins=[0, 0.1, 0.3, 0.5, 1.0],
            labels=["매우균형", "균형", "불균형", "매우불균형"],
        )

        balance_stats = df_temp.groupby("균형등급")["종합점수"].agg(["mean", "count"])

        # 막대 차트
        balance_stats.plot(
            kind="bar", y="mean", ax=axes[1], color="forestgreen", alpha=0.8
        )
        axes[1].set_xlabel("정책 강도 균형 등급")
        axes[1].set_ylabel("평균 종합점수")
        axes[1].set_title("정책 강도 균형과 종합점수")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)

        # 개수 정보 추가
        for i, (idx, row) in enumerate(balance_stats.iterrows()):
            axes[1].text(
                i,
                row["mean"] + 0.01,
                f"n={int(row['count'])}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # 3. Sigmoid 적용 효과 비교
        # 원래 값과 Sigmoid 적용 후 값의 분포 비교
        if "전략적_강도" in df.columns:
            # 원래 전략적 강도 분포
            axes[2].hist(
                df["전략적_강도"],
                bins=20,
                alpha=0.6,
                label="원래 전략적 강도",
                color="lightblue",
                density=True,
            )

            # 정규화된 전략적 강도 분포
            axes[2].hist(
                df["전략적_강도_정규화"],
                bins=20,
                alpha=0.6,
                label="정규화된 전략적 강도",
                color="lightcoral",
                density=True,
            )

            axes[2].set_xlabel("전략적 강도 값")
            axes[2].set_ylabel("밀도")
            axes[2].set_title("Sigmoid 적용 전후 분포 비교")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.suptitle("정책 효과성 및 Sigmoid 적용 효과 분석", fontsize=16, y=1.02)
        plt.tight_layout()

        # 저장
        effectiveness_path = self.result_dir / "sigmoid_only_policy_effectiveness.png"
        try:
            plt.savefig(effectiveness_path, dpi=300, bbox_inches="tight")
            print(f"✅ 정책 효과성 분석 플롯 저장 완료: {effectiveness_path}")
        except Exception as e:
            print(f"❌ 정책 효과성 분석 플롯 저장 실패: {e}")
        plt.close()

    def create_visualizations(self, df):
        """Sigmoid Only 결과 시각화를 생성합니다."""
        print("\n📊 Sigmoid Only 시각화 생성 중...")

        # 1. 기본 분석 플롯 (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 전체 분포
        axes[0, 0].hist(df["종합점수"], bins=20, alpha=0.7, color="steelblue")
        axes[0, 0].set_title("종합점수 분포 (Sigmoid Only)", fontsize=14)
        axes[0, 0].set_xlabel("종합점수")
        axes[0, 0].set_ylabel("빈도")
        axes[0, 0].grid(True, alpha=0.3)

        # 지역유형별 분포
        metro_data = df[df["지역유형"] == "광역자치단체"]["종합점수"]
        basic_data = df[df["지역유형"] == "기초자치단체"]["종합점수"]

        axes[0, 1].hist(
            [metro_data, basic_data],
            bins=15,
            alpha=0.7,
            label=["광역자치단체", "기초자치단체"],
            color=["red", "blue"],
        )
        axes[0, 1].set_title("지역유형별 종합점수 분포", fontsize=14)
        axes[0, 1].set_xlabel("종합점수")
        axes[0, 1].set_ylabel("빈도")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 행정적 vs 전략적 강도 산점도
        colors = ["red" if x == "광역자치단체" else "blue" for x in df["지역유형"]]
        axes[1, 0].scatter(df["행정적_강도"], df["전략적_강도"], c=colors, alpha=0.6)
        axes[1, 0].set_title("행정적 강도 vs 전략적 강도", fontsize=14)
        axes[1, 0].set_xlabel("행정적_강도")
        axes[1, 0].set_ylabel("전략적_강도")
        axes[1, 0].grid(True, alpha=0.3)

        # 상위 10개 지역 바 차트
        top10 = df.head(10)
        bars = axes[1, 1].barh(range(len(top10)), top10["종합점수"])
        axes[1, 1].set_yticks(range(len(top10)))
        axes[1, 1].set_yticklabels(top10["지역명"], fontsize=9)
        axes[1, 1].set_title("상위 10개 지역 (Sigmoid Only)", fontsize=14)
        axes[1, 1].set_xlabel("종합점수")
        axes[1, 1].grid(True, alpha=0.3)

        # 막대 색상 구분
        for i, bar in enumerate(bars):
            if top10.iloc[i]["지역유형"] == "광역자치단체":
                bar.set_color("red")
            else:
                bar.set_color("blue")

        plt.suptitle("청년정책 평가 결과 (Sigmoid Only)", fontsize=16)
        plt.tight_layout()

        # 저장
        plot_path = self.result_dir / "sigmoid_only_evaluation_plots.png"
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"✅ 기본 시각화 저장 완료: {plot_path}")
        except Exception as e:
            print(f"❌ 기본 시각화 저장 실패: {e}")
        plt.close()

        # 2. 산점도 모음 (3x2)
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        colors = ["red" if x == "광역자치단체" else "blue" for x in df["지역유형"]]

        # 종합점수 vs 재정자립도
        axes[0, 0].scatter(df["재정자립도"], df["종합점수"], c=colors, alpha=0.6)
        axes[0, 0].set_title("재정자립도 vs 종합점수", fontsize=14)
        axes[0, 0].set_xlabel("재정자립도")
        axes[0, 0].set_ylabel("종합점수")
        axes[0, 0].grid(True, alpha=0.3)

        # 종합점수 vs 청년1인당 정책예산
        axes[0, 1].scatter(
            df["청년1인당_정책예산_원"], df["종합점수"], c=colors, alpha=0.6
        )
        axes[0, 1].set_title("청년1인당 정책예산 vs 종합점수", fontsize=14)
        axes[0, 1].set_xlabel("청년1인당_정책예산_원")
        axes[0, 1].set_ylabel("종합점수")
        axes[0, 1].grid(True, alpha=0.3)

        # 행정적 강도 vs 재정자립도
        axes[1, 0].scatter(df["재정자립도"], df["행정적_강도"], c=colors, alpha=0.6)
        axes[1, 0].set_title("재정자립도 vs 행정적 강도", fontsize=14)
        axes[1, 0].set_xlabel("재정자립도")
        axes[1, 0].set_ylabel("행정적_강도")
        axes[1, 0].grid(True, alpha=0.3)

        # 전략적 강도 vs 청년인구
        axes[1, 1].scatter(df["청년인구"], df["전략적_강도"], c=colors, alpha=0.6)
        axes[1, 1].set_title("청년인구 vs 전략적 강도", fontsize=14)
        axes[1, 1].set_xlabel("청년인구")
        axes[1, 1].set_ylabel("전략적_강도")
        axes[1, 1].grid(True, alpha=0.3)

        # 청년1인당 정책예산 vs 재정자립도
        axes[2, 0].scatter(
            df["재정자립도"], df["청년1인당_정책예산_원"], c=colors, alpha=0.6
        )
        axes[2, 0].set_title("재정자립도 vs 청년1인당 정책예산", fontsize=14)
        axes[2, 0].set_xlabel("재정자립도")
        axes[2, 0].set_ylabel("청년1인당_정책예산_원")
        axes[2, 0].grid(True, alpha=0.3)

        # 청년인구 vs 전체인구 (비율 시각화)
        axes[2, 1].scatter(df["전체인구"], df["청년인구"], c=colors, alpha=0.6)
        axes[2, 1].set_title("전체인구 vs 청년인구", fontsize=14)
        axes[2, 1].set_xlabel("전체인구")
        axes[2, 1].set_ylabel("청년인구")
        axes[2, 1].grid(True, alpha=0.3)

        # 범례 추가
        import matplotlib.patches as mpatches

        red_patch = mpatches.Patch(color="red", label="광역자치단체")
        blue_patch = mpatches.Patch(color="blue", label="기초자치단체")
        fig.legend(
            handles=[red_patch, blue_patch],
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

        plt.suptitle("청년정책 평가 지표 간 관계 분석 (산점도)", fontsize=16)
        plt.tight_layout()

        # 산점도 저장
        scatter_path = self.result_dir / "sigmoid_only_scatter_plots.png"
        try:
            plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
            print(f"✅ 산점도 시각화 저장 완료: {scatter_path}")
        except Exception as e:
            print(f"❌ 산점도 시각화 저장 실패: {e}")
        plt.close()

        # 3. 종합 분석 플롯 추가
        self.create_comprehensive_analysis_plots(df)

        # 4. 정책 효과성 분석 플롯 추가
        self.create_policy_effectiveness_analysis(df)

        # 5. 상관관계 히트맵 추가
        self.create_correlation_heatmap(df)

        # 6. 정착 유도 스타일 플롯 추가
        self.create_settlement_style_plots(df)

    def create_correlation_heatmap(self, df):
        """상관관계 히트맵 생성"""
        print("\n📊 상관관계 히트맵 생성 중...")

        plt.figure(figsize=(12, 10))

        # 분석할 지표들
        correlation_cols = [
            "종합점수",
            "행정적_강도",
            "전략적_강도",
            "행정적_강도_정규화",
            "전략적_강도_정규화",
            "재정자립도",
            "청년1인당_정책예산_원",
            "청년인구",
        ]

        # 존재하는 컬럼만 선택
        existing_cols = [col for col in correlation_cols if col in df.columns]

        if len(existing_cols) > 1:
            correlation_data = df[existing_cols].corr()

            # 히트맵 생성
            mask = np.triu(
                np.ones_like(correlation_data, dtype=bool)
            )  # 상삼각형 마스크

            sns.heatmap(
                correlation_data,
                mask=mask,  # 상삼각형 숨기기
                annot=True,
                cmap="RdBu_r",
                center=0,
                square=True,
                fmt=".3f",
                cbar_kws={"shrink": 0.8},
                linewidths=0.5,
            )

            plt.title(
                "청년정책 평가 지표 상관관계 (Percentile + Sigmoid 적용)",
                fontsize=14,
                pad=20,
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 저장
            heatmap_path = self.result_dir / "sigmoid_only_correlation_heatmap.png"
            try:
                plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
                print(f"✅ 상관관계 히트맵 저장 완료: {heatmap_path}")
            except Exception as e:
                print(f"❌ 상관관계 히트맵 저장 실패: {e}")
            plt.close()

            # 상관관계 분석 결과 출력
            print(f"\n📈 주요 상관관계 분석 결과:")

            # 종합점수와 다른 지표들 간의 상관관계
            if "종합점수" in correlation_data.columns:
                comprehensive_corr = (
                    correlation_data["종합점수"]
                    .drop("종합점수")
                    .sort_values(key=abs, ascending=False)
                )
                print(f"\n🎯 종합점수와의 상관관계 (절댓값 기준 정렬):")
                for idx, corr_val in comprehensive_corr.head(5).items():
                    significance = (
                        "***"
                        if abs(corr_val) > 0.7
                        else (
                            "**"
                            if abs(corr_val) > 0.5
                            else "*" if abs(corr_val) > 0.3 else ""
                        )
                    )
                    print(f"  - {idx}: {corr_val:.3f}{significance}")

        else:
            print("❌ 상관관계 분석을 위한 충분한 데이터가 없습니다.")

    def create_settlement_style_plots(self, df):
        """정착 유도 스타일 플롯 생성 (순유입률 vs 정책 종합점수)"""
        print("\n📊 정착 유도 스타일 플롯 생성 중...")

        # 정책 종합점수 vs 순유입률 플롯 생성 (광역/기초/전체)
        if "종합점수" in df.columns and "순이동률_인구대비" in df.columns:

            # 지역유형별로 데이터 분리
            metropolitan_data = (
                df[df["지역유형"] == "광역자치단체"].copy()
                if "지역유형" in df.columns
                else pd.DataFrame()
            )

            municipal_data = (
                df[df["지역유형"] == "기초자치단체"].copy()
                if "지역유형" in df.columns
                else pd.DataFrame()
            )

            # 3x1 서브플롯 생성
            fig, axes = plt.subplots(1, 3, figsize=(30, 8))

            # 1. 광역자치단체 플롯
            if len(metropolitan_data) > 0:
                valid_metro = metropolitan_data[
                    ["종합점수", "순이동률_인구대비", "지역명"]
                ].dropna()

                if len(valid_metro) > 0:
                    x_metro = valid_metro["종합점수"]
                    y_metro = valid_metro["순이동률_인구대비"]

                    # 산점도
                    axes[0].scatter(
                        x_metro,
                        y_metro,
                        alpha=0.7,
                        s=120,
                        c="steelblue",
                        edgecolors="white",
                        linewidth=1,
                    )

                    # 회귀선 추가
                    if len(valid_metro) > 2:
                        z = np.polyfit(x_metro, y_metro, 1)
                        p = np.poly1d(z)
                        axes[0].plot(
                            x_metro,
                            p(x_metro),
                            "r--",
                            alpha=0.8,
                            linewidth=2,
                            label=f"회귀선: y = {z[0]:.0f}x + {z[1]:.0f}",
                        )

                        # 상관계수 계산 및 표시
                        corr_coef, p_value = stats.pearsonr(x_metro, y_metro)

                        # 유의성 표시
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = ""

                        axes[0].text(
                            0.05,
                            0.95,
                            f"상관계수: r = {corr_coef:.3f}{significance}\np-value = {p_value:.4f}\nn = {len(valid_metro)}",
                            transform=axes[0].transAxes,
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.5", facecolor="white", alpha=0.9
                            ),
                            verticalalignment="top",
                        )

                    # 지역명 라벨 추가
                    for idx, row in valid_metro.iterrows():
                        axes[0].annotate(
                            row["지역명"],
                            (row["종합점수"], row["순이동률_인구대비"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=9,
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightblue",
                                alpha=0.7,
                            ),
                        )

                    # 축 설정
                    axes[0].set_xlabel("정책 종합점수", fontsize=12)
                    axes[0].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                    axes[0].set_title(
                        f"광역자치단체 - 정책 종합점수 vs 순유입률\n(Sigmoid Only, n={len(valid_metro)})",
                        fontsize=14,
                        pad=20,
                    )
                    axes[0].grid(True, alpha=0.3)

                    if len(valid_metro) > 2:
                        axes[0].legend(loc="upper left")

            # 2. 기초자치단체 플롯
            if len(municipal_data) > 0:
                valid_muni = municipal_data[
                    ["종합점수", "순이동률_인구대비", "지역명"]
                ].dropna()

                if len(valid_muni) > 0:
                    x_muni = valid_muni["종합점수"]
                    y_muni = valid_muni["순이동률_인구대비"]

                    # 산점도
                    axes[1].scatter(
                        x_muni,
                        y_muni,
                        alpha=0.6,
                        s=60,
                        c="forestgreen",
                        edgecolors="white",
                        linewidth=0.5,
                    )

                    # 회귀선 추가
                    if len(valid_muni) > 2:
                        z = np.polyfit(x_muni, y_muni, 1)
                        p = np.poly1d(z)
                        axes[1].plot(
                            x_muni,
                            p(x_muni),
                            "r--",
                            alpha=0.8,
                            linewidth=2,
                            label=f"회귀선: y = {z[0]:.0f}x + {z[1]:.0f}",
                        )

                        # 상관계수 계산 및 표시
                        corr_coef, p_value = stats.pearsonr(x_muni, y_muni)

                        # 유의성 표시
                        if p_value < 0.001:
                            significance = "***"
                        elif p_value < 0.01:
                            significance = "**"
                        elif p_value < 0.05:
                            significance = "*"
                        else:
                            significance = ""

                        axes[1].text(
                            0.05,
                            0.95,
                            f"상관계수: r = {corr_coef:.3f}{significance}\np-value = {p_value:.4f}\nn = {len(valid_muni)}",
                            transform=axes[1].transAxes,
                            fontsize=12,
                            bbox=dict(
                                boxstyle="round,pad=0.5", facecolor="white", alpha=0.9
                            ),
                            verticalalignment="top",
                        )

                    # 상위/하위 10개 지역만 라벨 추가
                    sorted_muni = valid_muni.sort_values("종합점수")
                    top_bottom_muni = pd.concat(
                        [sorted_muni.head(5), sorted_muni.tail(5)]
                    )

                    for idx, row in top_bottom_muni.iterrows():
                        axes[1].annotate(
                            row["지역명"],
                            (row["종합점수"], row["순이동률_인구대비"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                            alpha=0.8,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="lightgreen",
                                alpha=0.7,
                            ),
                        )

                    # 축 설정
                    axes[1].set_xlabel("정책 종합점수", fontsize=12)
                    axes[1].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                    axes[1].set_title(
                        f"기초자치단체 - 정책 종합점수 vs 순유입률\n(Sigmoid Only, n={len(valid_muni)})",
                        fontsize=14,
                        pad=20,
                    )
                    axes[1].grid(True, alpha=0.3)

                    if len(valid_muni) > 2:
                        axes[1].legend(loc="upper left")

            # 3. 전체(광역+기초) 플롯
            valid_all = df[
                ["종합점수", "순이동률_인구대비", "지역유형", "지역명"]
            ].dropna()
            if len(valid_all) > 0:
                color_map = {"광역자치단체": "steelblue", "기초자치단체": "forestgreen"}
                colors = valid_all["지역유형"].map(color_map).fillna("gray")
                axes[2].scatter(
                    valid_all["종합점수"],
                    valid_all["순이동률_인구대비"],
                    c=colors,
                    alpha=0.6,
                    s=60,
                    edgecolors="white",
                    linewidth=0.5,
                    label=None,
                )

                # 회귀선 추가
                if len(valid_all) > 2:
                    z = np.polyfit(
                        valid_all["종합점수"], valid_all["순이동률_인구대비"], 1
                    )
                    p = np.poly1d(z)
                    axes[2].plot(
                        valid_all["종합점수"],
                        p(valid_all["종합점수"]),
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                        label=f"회귀선: y = {z[0]:.2f}x + {z[1]:.2f}",
                    )

                    # 상관계수 계산 및 표시
                    corr_coef, p_value = stats.pearsonr(
                        valid_all["종합점수"], valid_all["순이동률_인구대비"]
                    )

                    # 유의성 표시
                    if p_value < 0.001:
                        significance = "***"
                    elif p_value < 0.01:
                        significance = "**"
                    elif p_value < 0.05:
                        significance = "*"
                    else:
                        significance = ""

                    axes[2].text(
                        0.05,
                        0.95,
                        f"상관계수: r = {corr_coef:.3f}{significance}\np-value = {p_value:.4f}\nn = {len(valid_all)}",
                        transform=axes[2].transAxes,
                        fontsize=12,
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9
                        ),
                        verticalalignment="top",
                    )

                # 범례
                from matplotlib.lines import Line2D

                legend_elements = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label="광역자치단체",
                        markerfacecolor="steelblue",
                        markersize=10,
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label="기초자치단체",
                        markerfacecolor="forestgreen",
                        markersize=10,
                    ),
                ]

                # 축 설정
                axes[2].set_xlabel("정책 종합점수", fontsize=12)
                axes[2].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[2].set_title(
                    f"전체(광역+기초) - 정책 종합점수 vs 순유입률\n(Sigmoid Only, n={len(valid_all)})",
                    fontsize=14,
                    pad=20,
                )
                axes[2].grid(True, alpha=0.3)

                if len(valid_all) > 2:
                    axes[2].legend(
                        handles=legend_elements
                        + [
                            Line2D(
                                [0], [0], color="red", linestyle="--", label="회귀선"
                            )
                        ],
                        loc="upper left",
                    )

            plt.suptitle(
                "정책 종합점수 vs 순유입률 (광역 vs 기초 vs 전체)\n(Percentile + Sigmoid 적용, 순유입률 = 순이동/청년인구 × 100)",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout()

            # 저장
            save_path = self.result_dir / "sigmoid_only_settlement_style_plot.png"
            try:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"✅ 정착 유도 스타일 플롯 저장 완료: {save_path}")
            except Exception as e:
                print(f"❌ 정착 유도 스타일 플롯 저장 실패: {e}")
            plt.close()

            print("✅ 정책 종합점수 vs 순유입률 플롯 생성 완료 (광역/기초/전체)")

            # 간단한 분석 결과 출력
            print(f"\n📊 분석 결과:")
            if len(metropolitan_data) > 0:
                print(f"- 광역자치단체: {len(valid_metro)}개 지역")
                if len(valid_metro) > 0:
                    print(
                        f"  * 정책 종합점수 범위: {valid_metro['종합점수'].min():.3f} ~ {valid_metro['종합점수'].max():.3f}"
                    )
                    print(
                        f"  * 순유입률 범위: {valid_metro['순이동률_인구대비'].min():.2f}% ~ {valid_metro['순이동률_인구대비'].max():.2f}%"
                    )

            if len(municipal_data) > 0:
                print(f"- 기초자치단체: {len(valid_muni)}개 지역")
                if len(valid_muni) > 0:
                    print(
                        f"  * 정책 종합점수 범위: {valid_muni['종합점수'].min():.3f} ~ {valid_muni['종합점수'].max():.3f}"
                    )
                    print(
                        f"  * 순유입률 범위: {valid_muni['순이동률_인구대비'].min():.2f}% ~ {valid_muni['순이동률_인구대비'].max():.2f}%"
                    )

            if len(valid_all) > 0:
                print(f"- 전체: {len(valid_all)}개 지역")
                print(
                    f"  * 정책 종합점수 범위: {valid_all['종합점수'].min():.3f} ~ {valid_all['종합점수'].max():.3f}"
                )
                print(
                    f"  * 순유입률 범위: {valid_all['순이동률_인구대비'].min():.2f}% ~ {valid_all['순이동률_인구대비'].max():.2f}%"
                )

        else:
            print("❌ 필요한 컬럼(종합점수, 순이동률_인구대비)이 없습니다.")

    def print_summary(self, df):
        """결과 요약을 출력합니다."""
        print(f"\n=== Sigmoid Only 평가 결과 요약 ===")
        print(f"총 평가 지역: {len(df)}개")

        # 지역 유형별 통계
        type_stats = df.groupby("지역유형").size()
        print(f"\n📍 지역 유형별 분포:")
        for region_type, count in type_stats.items():
            print(f"   {region_type}: {count}개")

        # 상위 10개 지역 (종합점수 기준)
        print(f"\n🏆 **청년정책 종합평가 순위 결과 (Sigmoid Only)**")
        print("=" * 60)

        print(f"\n📍 **전체 순위 TOP 10**")
        print("-" * 40)
        top10 = df.head(10)[["전체순위", "지역명", "지역유형", "종합점수"]]
        for _, row in top10.iterrows():
            type_icon = "🏛️" if row["지역유형"] == "광역자치단체" else "🏘️"
            print(
                f"{row['전체순위']:2d}위. {type_icon} {row['지역명']} ({row['종합점수']:.4f})"
            )

        # 통계 요약
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
        print("=== Sigmoid Only 청년정책 종합 평가 시스템 시작 ===")

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

        # 6. 시각화
        self.create_visualizations(df)

        # 7. 요약 출력
        self.print_summary(df)

        print(f"\n✅ Sigmoid Only 평가 완료!")
        print(f"📁 결과 저장 위치: {self.result_dir}")

        return df


if __name__ == "__main__":
    evaluator = YouthPolicyEvaluationSigmoidOnly()
    results_df = evaluator.run_evaluation()
