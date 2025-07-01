#전국 청년정책 종합 평가 시스템 v2
#전략적 강도: 엔트로피 지수 (정책 분야별 균형성과 다양성)
#행정적 강도: ln(집중도지수/재정자립도+1) (집중도지수와 재정자립도 고려)
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
class YouthPolicyEvaluationSystemV2:
def init(self):
self.base_path = Path(file).parent.parent.parent.parent
Generated code
# 광역자치단체 목록 정의
    self.metropolitan_areas = {
        "강원특별자치도",
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
    file_path = self.base_path / "data/policy/청년인구/지자체별_청년인구비_통합.csv"
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
    """행정적 강도를 계산합니다."""
    # 데이터 수집
    total_budget = self.get_total_budget(region_name)  # 백만원
    youth_policy_budget = self.calculate_youth_policy_budget(region_name)  # 백만원
    youth_population_ratio = self.get_youth_population_ratio(region_name)
    finance_autonomy = self.get_finance_autonomy(region_name)

    # A: 청년정책 예산 비율
    if total_budget > 0:
        youth_budget_ratio = youth_policy_budget / total_budget
    else:
        youth_budget_ratio = 0

    # B: 청년인구 비율 (이미 비율로 계산됨)
    youth_population_ratio = youth_population_ratio

    # 집중도 지수 = A / B
    if youth_population_ratio > 0:
        concentration_index = youth_budget_ratio / youth_population_ratio
    else:
        concentration_index = 0

    # 행정적 강도 = ln(집중도지수/재정자립도 + 1)
    if finance_autonomy > 0:
        administrative_intensity = math.log(
            concentration_index / finance_autonomy + 1
        )
    else:
        administrative_intensity = math.log(concentration_index + 1)

    return {
        "administrative_intensity": administrative_intensity,
        "concentration_index": concentration_index,
        "youth_budget_ratio": youth_budget_ratio,
        "youth_population_ratio": youth_population_ratio,
        "finance_autonomy": finance_autonomy,
        "total_budget": total_budget,
        "youth_policy_budget": youth_policy_budget,
    }

def calculate_strategic_intensity(self, region_name):
    """전략적 강도를 계산합니다 (개선된 버전)."""
    if region_name not in self.policy_data:
        return {
            "strategic_intensity": 0,
            "entropy": 0,
            "normalized_entropy": 0,
            "total_policies": 0,
            "categories": 0,
            "category_counts": {},
            "policy_penalty": 0,
            "category_penalty": 0,
            "diversity_score": 0,
            "policy_score": 0,
            "category_score": 0,
        }

    region_data = self.policy_data[region_name]
    policy_execution = region_data.get("정책수행", {})

    # 분야별 정책 수 계산 ('사업수' 키 우선 활용)
    category_counts = {}
    total_policies = 0

    for category, category_data in policy_execution.items():
        if isinstance(category_data, dict):
            # 1단계: '사업수' 키 활용 (우선 방식)
            if "사업수" in category_data:
                policy_count = category_data["사업수"]
                if isinstance(policy_count, (int, float)) and policy_count > 0:
                    category_counts[category] = int(policy_count)
                    total_policies += int(policy_count)
                    continue

            # 2단계: 세부사업 개수 계산 (폴백)
            detail_projects = category_data.get("세부사업", [])
            if isinstance(detail_projects, list):
                policy_count = len(detail_projects)
                category_counts[category] = policy_count
                total_policies += policy_count
            else:
                # 3단계: 기존 방식으로 폴백 (세부사업이 없는 경우)
                policy_count = len(category_data)
                category_counts[category] = policy_count
                total_policies += policy_count

    if total_policies == 0:
        return {
            "strategic_intensity": 0,
            "entropy": 0,
            "normalized_entropy": 0,
            "total_policies": 0,
            "categories": 0,
            "category_counts": {},
            "policy_penalty": 0,
            "category_penalty": 0,
            "diversity_score": 0,
            "policy_score": 0,
            "category_score": 0,
        }

    # 엔트로피 계산
    entropy = 0
    categories = len(category_counts)

    for count in category_counts.values():
        if count > 0:
            p_i = count / total_policies
            entropy -= p_i * math.log2(p_i)

    # 정규화 (최대 엔트로피로 나누기)
    if categories > 1:
        max_entropy = math.log2(categories)
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0

    # 개선된 다양성 지표 계산
    # 1. 기본 엔트로피 점수
    entropy_score = normalized_entropy

    # 2. 정책 수 기반 점수 (더 세분화)
    if total_policies >= 50:
        policy_score = 1.0
    elif total_policies >= 40:
        policy_score = 0.9
    elif total_policies >= 30:
        policy_score = 0.8
    elif total_policies >= 20:
        policy_score = 0.6
    elif total_policies >= 15:
        policy_score = 0.5
    elif total_policies >= 10:
        policy_score = 0.4
    elif total_policies >= 5:
        policy_score = 0.2
    else:
        policy_score = 0.1

    # 3. 분야 수 기반 점수
    if categories >= 5:
        category_score = 1.0
    elif categories >= 4:
        category_score = 0.8
    elif categories >= 3:
        category_score = 0.6
    elif categories >= 2:
        category_score = 0.4
    else:
        category_score = 0.2

    # 4. 분산 기반 다양성 점수 (정책 수의 분산이 클수록 낮은 점수)
    if categories > 1:
        mean_policies = total_policies / categories
        variance = (
            sum((count - mean_policies) ** 2 for count in category_counts.values())
            / categories
        )
        # 분산이 클수록 다양성이 낮음 (0에 가까울수록 균등 분포)
        diversity_score = max(0.1, 1.0 - (variance / (mean_policies**2)))
    else:
        diversity_score = 0.5

    # 최종 전략적 강도 계산 (가중 평균)
    strategic_intensity = (
        entropy_score * 0.3  # 엔트로피 30%
        + policy_score * 0.3  # 정책 수 30%
        + category_score * 0.2  # 분야 수 20%
        + diversity_score * 0.2  # 분산 기반 다양성 20%
    )

    return {
        "strategic_intensity": strategic_intensity,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "total_policies": total_policies,
        "categories": categories,
        "category_counts": category_counts,
        "policy_penalty": policy_score,  # 기존 호환성을 위해 유지
        "category_penalty": category_score,  # 기존 호환성을 위해 유지
        "diversity_score": diversity_score,
        "policy_score": policy_score,
        "category_score": category_score,
    }

def evaluate_all_regions(self):
    """모든 지역을 평가합니다."""
    print("\\n=== 전국 청년정책 평가 시작 ===")

    results = []

    for region_name in self.policy_data.keys():
        print(f"평가 중: {region_name}")

        # 행정적 강도 계산
        admin_result = self.calculate_administrative_intensity(region_name)

        # 전략적 강도 계산
        strategic_result = self.calculate_strategic_intensity(region_name)

        # 지역 유형 판별
        region_type = (
            "광역자치단체"
            if self.is_metropolitan_area(region_name)
            else "기초자치단체"
        )

        result = {
            "지역명": region_name,
            "지역유형": region_type,
            "행정적_강도": admin_result["administrative_intensity"],
            "집중도_지수": admin_result["concentration_index"],
            "청년예산_비율": admin_result["youth_budget_ratio"],
            "청년인구_비율": admin_result["youth_population_ratio"],
            "재정자립도": admin_result["finance_autonomy"],
            "총예산_백만원": admin_result["total_budget"],
            "청년정책예산_백만원": admin_result["youth_policy_budget"],
            "전략적_강도": strategic_result["strategic_intensity"],
            "엔트로피": strategic_result["entropy"],
            "정규화_엔트로피": strategic_result["normalized_entropy"],
            "총정책수": strategic_result["total_policies"],
            "정책분야수": strategic_result["categories"],
            "정책페널티": strategic_result["policy_penalty"],
            "분야페널티": strategic_result["category_penalty"],
        }

        results.append(result)

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
    """결과를 저장합니다."""
    # CSV 저장 (순위 포함)
    csv_file = (
        self.base_path
        / "policy_evaluation/evaluation_results_index/evaluation-3/전국_청년정책_종합평가결과_v6.csv"
    )
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    # JSON 저장 (상세 분석용)
    json_file = (
        self.base_path
        / "policy_evaluation/evaluation_results_index/evaluation-3/전국_청년정책_분석결과_v6.json"
    )
    detailed_results = df.to_dict("records")

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 결과 저장 완료:")
    print(f"   CSV (기본): {csv_file}")
    print(f"   JSON: {json_file}")

    return csv_file, json_file

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

if name == "main":
evaluator = YouthPolicyEvaluationSystemV2()
results_df = evaluator.run_evaluation()
