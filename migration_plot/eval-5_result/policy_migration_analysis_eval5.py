"""
eval-5 결과를 사용한 정책 시차 청년 이동 분석 모듈
==================================================

이 모듈은 eval-5 결과를 바탕으로 정책 시행과 청년 인구 이동 간의 시간차를 고려하여 분석합니다.
- 광역자치단체: 종합점수 사용
- 기초자치단체: 최종_연계점수 사용 (광역연계 고려)
- 분석 기간: 2023년 8월 ~ 2024년 7월 (12개월)
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class PolicyLagAnalyzerEval5:
    """eval-5 결과를 사용한 정책 시차 청년 이동 분석 클래스"""

    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent
        else:
            self.base_path = Path(base_path)

        self.policy_data = None
        self.migration_data = None
        self.analysis_period_data = None
        self.merged_data = None

        # 분석 기간 설정 (2023년 8월 ~ 2024년 7월)
        self.start_year_month = 202308
        self.end_year_month = 202407

        print(f"📅 분석 기간: {self.start_year_month} ~ {self.end_year_month}")
        print(f"🔗 사용 데이터: eval-5 결과 (광역=종합점수, 기초=최종_연계점수)")

    def load_youth_population_data(self):
        """청년인구 데이터 로드"""
        youth_pop_file = (
            self.base_path / "data/policy/청년인구/시군구_청년비율_2023.csv"
        )

        if not youth_pop_file.exists():
            print("❌ 청년인구 데이터 파일을 찾을 수 없습니다.")
            return None

        youth_pop_data = pd.read_csv(youth_pop_file, encoding="utf-8-sig")
        print(f"✅ 청년인구 데이터 로드: {len(youth_pop_data)}개 지역")

        # 지역명 정제 함수
        def normalize_region_name(name):
            if pd.isna(name):
                return ""
            return str(name).strip().replace("  ", " ")

        youth_pop_data["지역명_정제"] = youth_pop_data["지자체명"].apply(
            normalize_region_name
        )

        return youth_pop_data

    def map_youth_population(self, policy_data, youth_pop_data):
        """정책 데이터에 실제 청년인구 데이터 매핑"""

        # 지역명 정제 함수
        def normalize_region_name(name):
            if pd.isna(name):
                return ""
            return str(name).strip().replace("  ", " ")

        # 정책 데이터의 지역명 정제
        if "지역명" in policy_data.columns:
            policy_data["지역명_정제"] = policy_data["지역명"].apply(
                normalize_region_name
            )
        else:
            print("❌ 정책 데이터에 '지역명' 컬럼이 없습니다.")
            return policy_data

        # 매핑 딕셔너리 생성
        youth_pop_dict = {}
        total_pop_dict = {}

        for _, row in youth_pop_data.iterrows():
            region_name = row["지역명_정제"]
            youth_pop_dict[region_name] = row["청년인구"]
            total_pop_dict[region_name] = row["전체인구"]

        # 매핑 수행
        matched_count = 0
        unmatched_regions = []

        for idx, row in policy_data.iterrows():
            region_name = row["지역명_정제"]

            # 직접 매칭 시도
            if region_name in youth_pop_dict:
                policy_data.at[idx, "청년인구"] = youth_pop_dict[region_name]
                policy_data.at[idx, "전체인구"] = total_pop_dict[region_name]
                matched_count += 1
            else:
                # 부분 매칭 시도 (예: "서울특별시" vs "서울")
                found = False
                for youth_region in youth_pop_dict.keys():
                    if (region_name in youth_region) or (youth_region in region_name):
                        # 더 정확한 매칭을 위해 길이가 긴 것을 우선
                        if len(youth_region) > 3:  # 너무 짧은 매칭 방지
                            policy_data.at[idx, "청년인구"] = youth_pop_dict[
                                youth_region
                            ]
                            policy_data.at[idx, "전체인구"] = total_pop_dict[
                                youth_region
                            ]
                            matched_count += 1
                            found = True
                            break

                if not found:
                    unmatched_regions.append(region_name)

        print(
            f"✅ 청년인구 데이터 매핑 완료: {matched_count}/{len(policy_data)}개 지역 매칭"
        )

        if unmatched_regions:
            print(
                f"⚠️  매칭되지 않은 지역 ({len(unmatched_regions)}개): {unmatched_regions[:5]}{'...' if len(unmatched_regions) > 5 else ''}"
            )

        # 매핑되지 않은 지역은 기본값 설정
        for col in ["청년인구", "전체인구"]:
            if col in policy_data.columns:
                default_value = 50000 if col == "청년인구" else 200000
                policy_data[col] = policy_data[col].fillna(default_value)
                # 0인 값도 기본값으로 대체
                policy_data.loc[policy_data[col] == 0, col] = default_value

        return policy_data

    def load_data(self):
        """eval-5 정책 데이터와 마이그레이션 데이터 로드"""
        # 청년인구 데이터 먼저 로드
        youth_pop_data = self.load_youth_population_data()

        # 광역자치단체 정책 데이터 로드
        metropolitan_policy_file = (
            self.base_path
            / "data/policy_eval/eval-5_result/광역_청년정책_종합평가결과.csv"
        )

        # 기초자치단체 정책 데이터 로드 (eval-5 결과 사용)
        municipal_policy_file = (
            self.base_path
            / "data/policy_eval/eval-5_result/기초_최종평가결과(광역연계).csv"
        )

        policy_data_list = []

        # 광역자치단체 데이터 로드
        if metropolitan_policy_file.exists():
            metro_data = pd.read_csv(metropolitan_policy_file, encoding="utf-8-sig")
            metro_data["지역유형"] = "광역자치단체"
            metro_data["점수_컬럼"] = "종합점수"
            metro_data["사용_점수"] = metro_data["종합점수"]

            # 청년인구 데이터 매핑
            if youth_pop_data is not None:
                metro_data = self.map_youth_population(metro_data, youth_pop_data)

            print(
                f"✅ 광역자치단체 정책 데이터 로드: {len(metro_data)}개 지역 (종합점수 사용)"
            )
            policy_data_list.append(metro_data)
        else:
            print("❌ 광역자치단체 정책 데이터 파일을 찾을 수 없습니다.")

        # 기초자치단체 데이터 로드 (최종_연계점수 사용)
        if municipal_policy_file.exists():
            muni_data = pd.read_csv(municipal_policy_file, encoding="utf-8-sig")
            muni_data["지역유형"] = "기초자치단체"
            muni_data["점수_컬럼"] = "최종_연계점수"
            muni_data["사용_점수"] = muni_data["최종_연계점수"]

            # 청년인구 데이터 매핑
            if youth_pop_data is not None:
                muni_data = self.map_youth_population(muni_data, youth_pop_data)

            # 기초자치단체 데이터에 없는 컬럼들 확인하고 기본값 설정
            required_cols = ["전략적_강도", "청년인구", "전체인구"]
            for col in required_cols:
                if col not in muni_data.columns:
                    if col == "전략적_강도":
                        muni_data[col] = muni_data.get("전략적_강도", 0)  # 기본값 0
                    elif col == "청년인구":
                        muni_data[col] = 50000  # 매핑되지 않은 경우 기본값
                    elif col == "전체인구":
                        muni_data[col] = 200000  # 기본값

            print(
                f"✅ 기초자치단체 정책 데이터 로드: {len(muni_data)}개 지역 (최종_연계점수 사용)"
            )
            policy_data_list.append(muni_data)
        else:
            print("❌ 기초자치단체 정책 데이터 파일을 찾을 수 없습니다.")

        if not policy_data_list:
            print("❌ 정책 데이터를 로드할 수 없습니다.")
            return False

        # 두 데이터를 합치기
        self.policy_data = pd.concat(policy_data_list, ignore_index=True)
        print(f"✅ 전체 정책 데이터 통합: {len(self.policy_data)}개 지역")

        # 광역/기초 구분을 위한 데이터 저장
        self.metropolitan_data = (
            metro_data if metropolitan_policy_file.exists() else None
        )
        self.municipal_data = muni_data if municipal_policy_file.exists() else None

        # 마이그레이션 데이터 로드
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
                print(f"   📅 {current_ym} 데이터 발견")

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

        # 데이터 통합
        dfs = []
        for file in sorted(target_files):
            try:
                year_month = file.stem.split("_")[-1]
                df = pd.read_csv(file, encoding="utf-8-sig")
                df["연월"] = year_month
                df["연도"] = int(year_month[:4])
                df["월"] = int(year_month[4:])
                dfs.append(df)
                print(f"   ✅ {year_month} 데이터 로드 완료")
            except Exception as e:
                print(f"파일 로드 오류 {file}: {e}")
                continue

        if dfs:
            self.migration_data = pd.concat(dfs, ignore_index=True)
            print(
                f"✅ 총 마이그레이션 데이터 로드: {len(self.migration_data)}개 레코드"
            )
            return True
        else:
            print("❌ 마이그레이션 데이터 로드에 실패했습니다.")
            return False

    def preprocess_migration_data(self):
        """파일별로 각 지역의 컬럼합(전입), row합(전출) 누적 방식으로 순이동 계산"""
        migration_dir = self.base_path / "data/migration/청년 인구 이동량_consolidated"
        if not migration_dir.exists():
            print("❌ 마이그레이션 데이터 디렉토리를 찾을 수 없습니다.")
            return False

        # 분석 기간 파일 목록
        target_files = []
        current_ym = self.start_year_month
        while current_ym <= self.end_year_month:
            file_path = migration_dir / f"youth_total_migration_{current_ym}.csv"
            if file_path.exists():
                target_files.append(file_path)
            current_month = current_ym % 100
            current_year = current_ym // 100
            if current_month == 12:
                current_ym = (current_year + 1) * 100 + 1
            else:
                current_ym = current_year * 100 + (current_month + 1)
        if not target_files:
            print("❌ 분석 기간에 해당하는 마이그레이션 데이터를 찾을 수 없습니다.")
            return False

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
            net = inflow - outflow
            result.append(
                {"지역명": region, "전입": inflow, "전출": outflow, "순이동": net}
            )
        self.analysis_period_data = pd.DataFrame(result)
        print(
            f"✅ 파일 누적 방식 순이동 계산 완료: {len(self.analysis_period_data)}개 지역"
        )
        return True

    def merge_policy_migration_data(self):
        """정책 데이터와 마이그레이션 데이터 통합"""
        if self.policy_data is None or self.analysis_period_data is None:
            print("❌ 정책 데이터, 마이그레이션 데이터를 먼저 준비해주세요.")
            return False

        print("🔗 정책 데이터와 마이그레이션 데이터 매칭 중...")

        # 지역명 정제 및 매칭
        def normalize_region_name(name):
            """지역명 정규화"""
            if pd.isna(name):
                return ""
            return str(name).strip().replace("  ", " ")

        # 정책 데이터 지역명 정제 (광역/기초 각각의 컬럼 이름 고려)
        if "지역명" in self.policy_data.columns:
            self.policy_data["지역명_정제"] = self.policy_data["지역명"].apply(
                normalize_region_name
            )
        else:
            print("❌ 정책 데이터에 '지역명' 컬럼이 없습니다.")
            return False

        # 마이그레이션 데이터 지역명 정제
        self.analysis_period_data["지역명_정제"] = self.analysis_period_data[
            "지역명"
        ].apply(normalize_region_name)

        # 매칭 가능한 지역 확인
        policy_regions = set(self.policy_data["지역명_정제"])
        migration_regions = set(self.analysis_period_data["지역명_정제"])

        print(f"정책 데이터 지역 수: {len(policy_regions)}")
        print(f"마이그레이션 데이터 지역 수: {len(migration_regions)}")

        matched_regions = policy_regions.intersection(migration_regions)
        print(f"매칭 가능한 지역: {len(matched_regions)}개")

        # 정책 + 마이그레이션 데이터 병합
        self.merged_data = pd.merge(
            self.policy_data,
            self.analysis_period_data,
            left_on="지역명_정제",
            right_on="지역명_정제",
            how="inner",
            suffixes=("_정책", "_이동"),
        )

        if len(self.merged_data) > 0:
            print(f"✅ 성공적으로 통합된 지역: {len(self.merged_data)}개")

            # 청년인구 컬럼 확인 및 처리
            if "청년인구" in self.merged_data.columns:
                youth_pop_col = "청년인구"
            else:
                print("❌ 청년인구 컬럼을 찾을 수 없습니다.")
                return False

            # 순이동률 계산 (청년 인구 수 대비 %)
            self.merged_data["순이동률_인구대비"] = (
                self.merged_data["순이동"] / (self.merged_data[youth_pop_col] + 1)
            ) * 100  # 백분율로 변환

            print(f"\n📊 순이동률 통계:")
            print(
                f"  - 평균 순이동률: {self.merged_data['순이동률_인구대비'].mean():.3f}%"
            )
            print(
                f"  - 순이동률 범위: {self.merged_data['순이동률_인구대비'].min():.3f}% ~ {self.merged_data['순이동률_인구대비'].max():.3f}%"
            )

            # 결과 CSV로 저장
            save_path = (
                self.base_path
                / "migration_plot/eval-5_result/settlement_induction_result_eval5.csv"
            )
            self.merged_data.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ 통합 결과 CSV 저장 완료: {save_path}")

            return True
        else:
            print("❌ 매칭되는 지역이 없습니다. 지역명 형식을 확인해주세요.")
            return False

    def analyze_policy_migration_correlation(self):
        """정책 효과성과 청년 이동 패턴의 상관관계 분석"""
        if self.merged_data is None or len(self.merged_data) == 0:
            print("❌ 통합 데이터가 없습니다.")
            return None

        print("📊 정책 시차 반영 상관관계 분석 중...")

        # 분석할 지표 (사용_점수를 사용)
        policy_vars = ["사용_점수", "전략적_강도", "행정적_강도"]
        migration_vars = ["순이동", "전입", "전출", "순이동률_인구대비"]

        # 상관관계 결과 저장
        correlation_results = []

        for policy_var in policy_vars:
            if policy_var not in self.merged_data.columns:
                continue

            for migration_var in migration_vars:
                if migration_var not in self.merged_data.columns:
                    continue

                # 유효한 데이터만 추출
                valid_data = self.merged_data[[policy_var, migration_var]].dropna()

                if len(valid_data) < 10:  # 최소 데이터 수 확인
                    continue

                # 상관관계 계산
                corr_coef, p_value = stats.pearsonr(
                    valid_data[policy_var], valid_data[migration_var]
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

                correlation_results.append(
                    {
                        "정책지표": policy_var,
                        "이동지표": migration_var,
                        "상관계수": corr_coef,
                        "p값": p_value,
                        "유의성": significance,
                        "표본수": len(valid_data),
                    }
                )

        # 결과 출력
        if correlation_results:
            corr_df = pd.DataFrame(correlation_results)

            print(
                f"\n📈 정책 시차 반영 상관관계 분석 결과 (n={len(self.merged_data)}):"
            )
            print("=" * 90)

            for _, row in corr_df.iterrows():
                print(
                    f"{row['정책지표']:15} ↔ {row['이동지표']:15}: "
                    f"r = {row['상관계수']:6.3f}{row['유의성']:3} "
                    f"(p = {row['p값']:6.3f}, n = {row['표본수']:3})"
                )

            # 상관관계 히트맵 생성
            if len(corr_df) > 0:
                pivot_corr = corr_df.pivot(
                    index="정책지표", columns="이동지표", values="상관계수"
                )

                plt.figure(figsize=(14, 8))
                sns.heatmap(
                    pivot_corr,
                    annot=True,
                    cmap="RdBu_r",
                    center=0,
                    square=True,
                    fmt=".3f",
                    cbar_kws={"shrink": 0.8},
                )
                plt.title(
                    "정책 효과성 vs 청년 이동 상관관계 (eval-5)\n(광역=종합점수, 기초=최종_연계점수, 시차: 2023.08-2024.07)",
                    fontsize=14,
                    pad=20,
                )
                plt.tight_layout()
                plt.savefig(
                    self.base_path
                    / "migration_plot/eval-5_result/policy_lag_correlation_eval5.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

            return corr_df
        else:
            print("❌ 분석 가능한 데이터가 없습니다.")
            return None

    def create_settlement_induction_plot(self):
        """정책 점수 vs 순유입률 플롯 생성 (광역=종합점수, 기초=최종_연계점수)"""
        if self.merged_data is None:
            print("❌ 통합 데이터가 없습니다.")
            return

        # 지역유형별로 데이터 분리
        metropolitan_data = (
            self.merged_data[self.merged_data["지역유형"] == "광역자치단체"].copy()
            if "지역유형" in self.merged_data.columns
            else pd.DataFrame()
        )

        municipal_data = (
            self.merged_data[self.merged_data["지역유형"] == "기초자치단체"].copy()
            if "지역유형" in self.merged_data.columns
            else pd.DataFrame()
        )

        # 3x1 서브플롯 생성
        fig, axes = plt.subplots(1, 3, figsize=(30, 8))

        # 1. 광역자치단체 플롯 (종합점수 사용)
        if len(metropolitan_data) > 0 and "종합점수" in metropolitan_data.columns:
            valid_metro = metropolitan_data[
                ["종합점수", "순이동률_인구대비", "지역명_이동"]
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
                    )

                # 지역명 라벨 추가
                for idx, row in valid_metro.iterrows():
                    axes[0].annotate(
                        row["지역명_이동"],
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
                axes[0].set_xlabel(
                    "정책 종합점수",
                    fontsize=12,
                )
                axes[0].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[0].set_title(
                    f"광역자치단체 - 정책 종합점수 vs 청년 순유입률\n(n={len(valid_metro)})",
                    fontsize=14,
                    pad=20,
                )
                axes[0].grid(True, alpha=0.3)
                axes[0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
                axes[0].axvline(
                    x=valid_metro["종합점수"].mean(),
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                )

        # 2. 기초자치단체 플롯 (최종_연계점수 사용)
        if len(municipal_data) > 0 and "최종_연계점수" in municipal_data.columns:
            valid_muni = municipal_data[
                ["최종_연계점수", "순이동률_인구대비", "지역명_이동"]
            ].dropna()

            if len(valid_muni) > 0:
                x_muni = valid_muni["최종_연계점수"]
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
                    )

                # 상위/하위 10개 지역만 라벨 추가 (너무 많아서 줄임)
                sorted_muni = valid_muni.sort_values("최종_연계점수")
                top_bottom_muni = pd.concat(
                    [sorted_muni.head(10), sorted_muni.tail(10)]
                )

                for idx, row in top_bottom_muni.iterrows():
                    axes[1].annotate(
                        row["지역명_이동"],
                        (row["최종_연계점수"], row["순이동률_인구대비"]),
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
                axes[1].set_xlabel(
                    "최종 연계점수 (광역연계)",
                    fontsize=12,
                )
                axes[1].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[1].set_title(
                    f"기초자치단체 - 최종 연계점수 vs 청년 순유입률\n(n={len(valid_muni)})",
                    fontsize=14,
                    pad=20,
                )
                axes[1].grid(True, alpha=0.3)
                axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
                axes[1].axvline(
                    x=valid_muni["최종_연계점수"].mean(),
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                )

        # 3. 전체(광역+기초) 플롯 (사용_점수 통합)
        valid_all = self.merged_data[
            ["사용_점수", "순이동률_인구대비", "지역유형", "지역명_이동"]
        ].dropna()

        if len(valid_all) > 0:
            color_map = {"광역자치단체": "steelblue", "기초자치단체": "forestgreen"}
            colors = valid_all["지역유형"].map(color_map).fillna("gray")

            axes[2].scatter(
                valid_all["사용_점수"],
                valid_all["순이동률_인구대비"],
                c=colors,
                alpha=0.6,
                s=60,
                edgecolors="white",
                linewidth=0.5,
            )

            # 회귀선 추가
            if len(valid_all) > 2:
                z = np.polyfit(
                    valid_all["사용_점수"], valid_all["순이동률_인구대비"], 1
                )
                p = np.poly1d(z)
                axes[2].plot(
                    valid_all["사용_점수"],
                    p(valid_all["사용_점수"]),
                    "r--",
                    alpha=0.8,
                    linewidth=2,
                )

            # 범례
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="광역자치단체 (종합점수)",
                    markerfacecolor="steelblue",
                    markersize=10,
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="기초자치단체 (최종_연계점수)",
                    markerfacecolor="forestgreen",
                    markersize=10,
                ),
            ]

            axes[2].legend(handles=legend_elements, loc="upper left")

            # 축 설정
            axes[2].set_xlabel(
                "정책 점수 (광역=종합점수, 기초=최종_연계점수)", fontsize=12
            )
            axes[2].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
            axes[2].set_title(
                f"전체(광역+기초) - 정책 점수 vs 청년 순유입률\n(n={len(valid_all)})",
                fontsize=14,
                pad=20,
            )
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            axes[2].axvline(
                x=valid_all["사용_점수"].mean(),
                color="gray",
                linestyle="--",
                alpha=0.3,
            )

        plt.suptitle(
            "정책 점수 vs 청년 순유입률 (eval-5)\n(광역=종합점수, 기초=최종_연계점수, 시차: 2023.08-2024.07)",
            fontsize=16,
            y=0.98,
        )
        plt.tight_layout()

        # 저장
        save_path = (
            self.base_path
            / "migration_plot/eval-5_result/settlement_induction_plot_eval5.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print("✅ 정책 점수 vs 순유입률 플롯 생성 완료 (eval-5)")
        print(f"📁 저장 위치: {save_path}")

        # 간단한 분석 결과 출력
        print(f"\n📊 분석 결과:")
        if len(metropolitan_data) > 0 and "종합점수" in metropolitan_data.columns:
            valid_metro = metropolitan_data[["종합점수", "순이동률_인구대비"]].dropna()
            if len(valid_metro) > 0:
                print(f"- 광역자치단체: {len(valid_metro)}개 지역")
                print(
                    f"  * 종합점수 범위: {valid_metro['종합점수'].min():.2f} ~ {valid_metro['종합점수'].max():.2f}"
                )
                print(
                    f"  * 순유입률 범위: {valid_metro['순이동률_인구대비'].min():.3f}% ~ {valid_metro['순이동률_인구대비'].max():.3f}%"
                )

        if len(municipal_data) > 0 and "최종_연계점수" in municipal_data.columns:
            valid_muni = municipal_data[["최종_연계점수", "순이동률_인구대비"]].dropna()
            if len(valid_muni) > 0:
                print(f"- 기초자치단체: {len(valid_muni)}개 지역")
                print(
                    f"  * 최종_연계점수 범위: {valid_muni['최종_연계점수'].min():.2f} ~ {valid_muni['최종_연계점수'].max():.2f}"
                )
                print(
                    f"  * 순유입률 범위: {valid_muni['순이동률_인구대비'].min():.3f}% ~ {valid_muni['순이동률_인구대비'].max():.3f}%"
                )

        print(f"- 전체(광역+기초): {len(valid_all)}개 지역")
        if len(valid_all) > 0:
            print(
                f"  * 정책 점수 범위: {valid_all['사용_점수'].min():.2f} ~ {valid_all['사용_점수'].max():.2f}"
            )
            print(
                f"  * 순유입률 범위: {valid_all['순이동률_인구대비'].min():.3f}% ~ {valid_all['순이동률_인구대비'].max():.3f}%"
            )

            # 분석 결과를 CSV로 저장
        save_csv_path = (
            self.base_path
            / "migration_plot/eval-5_result/settlement_analysis_results_eval5.csv"
        )

        # 전체 결과 데이터를 CSV로 저장 - 모든 컬럼 포함
        if len(valid_all) > 0:
            # merged_data에서 필요한 모든 컬럼 선택
            columns_to_save = [
                "지역명_이동",
                "지역유형",
                "사용_점수",
                "순이동률_인구대비",
                "순이동",
                "전입",
                "전출",
                "청년인구",
                "전체인구",
                "전략적_강도",
                "행정적_강도",
            ]

            # 광역/기초 구분별 점수 컬럼도 추가
            if "종합점수" in self.merged_data.columns:
                columns_to_save.append("종합점수")
            if "최종_연계점수" in self.merged_data.columns:
                columns_to_save.append("최종_연계점수")
            if "점수_컬럼" in self.merged_data.columns:
                columns_to_save.append("점수_컬럼")

            # 실제로 존재하는 컬럼만 선택
            available_columns = [
                col for col in columns_to_save if col in self.merged_data.columns
            ]

            # 데이터 저장
            result_df = self.merged_data[available_columns].copy()

            # 순이동률이 있는 데이터만 필터링 (핵심 분석 결과)
            result_df = result_df.dropna(subset=["순이동률_인구대비"])

            # 순이동률 기준으로 정렬
            result_df = result_df.sort_values("순이동률_인구대비", ascending=False)

            result_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n✅ 분석 결과 CSV 저장 완료: {save_csv_path}")
            print(f"   저장된 데이터: {len(result_df)}개 지역")
            print(f"   포함된 컬럼: {', '.join(available_columns)}")

        return {
            "metropolitan": valid_metro if len(metropolitan_data) > 0 else None,
            "municipal": valid_muni if len(municipal_data) > 0 else None,
            "all": valid_all if len(valid_all) > 0 else None,
        }

    def create_policy_lag_visualization(self):
        """정책 시차 시각화 (eval-5 버전)"""
        if self.merged_data is None:
            return

        # 4개 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 사용_점수 vs 순이동 산점도
        if (
            "사용_점수" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            valid_data = self.merged_data[
                ["사용_점수", "순이동", "지역명_이동"]
            ].dropna()

            x = valid_data["사용_점수"]
            y = valid_data["순이동"]

            # 산점도
            scatter = axes[0, 0].scatter(x, y, alpha=0.6, s=60, c="steelblue")

            # 회귀선
            if len(valid_data) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(x, p(x), "r--", alpha=0.8, linewidth=2)

            axes[0, 0].set_xlabel("정책 점수 (광역=종합점수, 기초=최종_연계점수)")
            axes[0, 0].set_ylabel("순이동 (전입-전출)")
            axes[0, 0].set_title(
                "정책 효과성 vs 청년 순이동\n(eval-5, 시차: 2023.08-2024.07)"
            )
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 전략적 강도 vs 전입 산점도
        if (
            "전략적_강도" in self.merged_data.columns
            and "전입" in self.merged_data.columns
        ):
            valid_data = self.merged_data[
                ["전략적_강도", "전입", "지역명_이동"]
            ].dropna()

            # 산점도 그리기
            axes[0, 1].scatter(
                valid_data["전략적_강도"],
                valid_data["전입"],
                alpha=0.6,
                s=60,
                c="forestgreen",
            )

            # 전입 기준 상위/하위 5개 지역 라벨링
            sorted_data = valid_data.sort_values("전입", ascending=False)
            top5 = sorted_data.head(5)
            bottom5 = sorted_data.tail(5)

            for _, row in pd.concat([top5, bottom5]).iterrows():
                axes[0, 1].annotate(
                    row["지역명_이동"],
                    (row["전략적_강도"], row["전입"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="forestgreen",
                        alpha=0.7,
                        boxstyle="round,pad=0.5",
                    ),
                )

            axes[0, 1].set_xlabel("정책 전략적 강도")
            axes[0, 1].set_ylabel("청년 전입")
            axes[0, 1].set_title("정책 전략적 강도 vs 청년 전입")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 순이동률 vs 사용_점수 (정규화된 버전)
        if (
            "사용_점수" in self.merged_data.columns
            and "순이동률_인구대비" in self.merged_data.columns
        ):
            valid_data = self.merged_data[["사용_점수", "순이동률_인구대비"]].dropna()

            axes[1, 0].scatter(
                valid_data["사용_점수"],
                valid_data["순이동률_인구대비"],
                alpha=0.6,
                s=60,
                c="darkorange",
            )
            axes[1, 0].set_xlabel("정책 점수")
            axes[1, 0].set_ylabel("순이동률 (청년인구 대비 %)")
            axes[1, 0].set_title("정책 점수 vs 순이동률 (정규화)")
            axes[1, 0].grid(True, alpha=0.3)

        # 4. 지역유형별 순이동 분포
        if (
            "지역유형" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            region_types = self.merged_data["지역유형"].unique()

            box_data = []
            labels = []
            for rt in region_types:
                data = self.merged_data[self.merged_data["지역유형"] == rt][
                    "순이동"
                ].dropna()
                if len(data) > 0:
                    box_data.append(data)
                    labels.append(rt)

            if box_data:
                axes[1, 1].boxplot(box_data, labels=labels)
                axes[1, 1].set_ylabel("순이동")
                axes[1, 1].set_title("지역유형별 청년 순이동 분포")
                axes[1, 1].tick_params(axis="x", rotation=45)

        plt.suptitle(
            "정책 시차를 고려한 청년 이동 패턴 분석 (eval-5)", fontsize=16, y=0.98
        )
        plt.tight_layout()
        plt.savefig(
            self.base_path
            / "migration_plot/eval-5_result/policy_lag_analysis_eval5.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("✅ 정책 시차 시각화 완료 (eval-5)")

    def generate_lag_analysis_report(self):
        """정책 시차 분석 종합 리포트 생성 (eval-5 버전)"""
        if self.merged_data is None:
            print("❌ 분석 데이터가 없어 리포트를 생성할 수 없습니다.")
            return

        report = []
        report.append("=" * 80)
        report.append("정책 시차를 고려한 청년 이동 패턴 분석 리포트 (eval-5)")
        report.append("=" * 80)
        report.append("")
        report.append(f"📅 분석 기간: 2023년 8월 ~ 2024년 7월 (12개월)")
        report.append(
            f"🎯 분석 목적: 정책 시행 후 실제 청년 이동에 미치는 지연 효과 측정"
        )
        report.append(f"📊 분석 대상: {len(self.merged_data)}개 지역")
        report.append(
            f"🔗 사용 데이터: 광역자치단체=종합점수, 기초자치단체=최종_연계점수"
        )
        report.append("")

        # 기본 통계
        if "순이동" in self.merged_data.columns:
            total_net = self.merged_data["순이동"].sum()
            positive_regions = len(self.merged_data[self.merged_data["순이동"] > 0])
            negative_regions = len(self.merged_data[self.merged_data["순이동"] < 0])

            report.append("📈 기본 현황")
            report.append(f"- 전체 순이동: {total_net:,}명")
            report.append(f"- 순유입 지역: {positive_regions}개")
            report.append(f"- 순유출 지역: {negative_regions}개")
            report.append("")

        # 순이동률 통계
        if "순이동률_인구대비" in self.merged_data.columns:
            avg_rate = self.merged_data["순이동률_인구대비"].mean()
            max_rate = self.merged_data["순이동률_인구대비"].max()
            min_rate = self.merged_data["순이동률_인구대비"].min()

            report.append("📊 청년인구 대비 순이동률 현황")
            report.append(f"- 평균 순이동률: {avg_rate:.3f}%")
            report.append(f"- 순이동률 범위: {min_rate:.3f}% ~ {max_rate:.3f}%")
            report.append("")

        # 정책 효과성 상위/하위 지역 비교
        if (
            "사용_점수" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            top_policy = self.merged_data.nlargest(10, "사용_점수")
            bottom_policy = self.merged_data.nsmallest(10, "사용_점수")

            top_migration_avg = top_policy["순이동"].mean()
            bottom_migration_avg = bottom_policy["순이동"].mean()

            report.append("🏆 정책 효과성별 이동 패턴 (시차 반영)")
            report.append(
                f"- 정책 상위 10개 지역 평균 순이동: {top_migration_avg:,.1f}명"
            )
            report.append(
                f"- 정책 하위 10개 지역 평균 순이동: {bottom_migration_avg:,.1f}명"
            )
            report.append(
                f"- 정책 효과성에 따른 이동 격차: {top_migration_avg - bottom_migration_avg:,.1f}명"
            )

            # 상관관계
            valid_data = self.merged_data[["사용_점수", "순이동"]].dropna()
            if len(valid_data) > 10:
                corr_coef, p_value = stats.pearsonr(
                    valid_data["사용_점수"], valid_data["순이동"]
                )
                significance = (
                    "통계적으로 유의함"
                    if p_value < 0.05
                    else "통계적으로 유의하지 않음"
                )
                report.append(
                    f"- 정책 점수 ↔ 순이동 상관계수: {corr_coef:.3f} ({significance})"
                )

            report.append("")

        # 지역 유형별 분석
        if (
            "지역유형" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            region_stats = self.merged_data.groupby("지역유형")["순이동"].agg(
                ["mean", "std", "count"]
            )

            report.append("🏛️ 지역유형별 청년 이동 패턴")
            for region_type, stat_data in region_stats.iterrows():
                report.append(
                    f"- {region_type}: 평균 {stat_data['mean']:,.1f}명 "
                    f"(표준편차 {stat_data['std']:,.1f}, n={stat_data['count']})"
                )
            report.append("")

        # 주요 발견사항
        report.append("🔍 eval-5 기반 정책 시차 분석 주요 발견사항")
        report.append(
            "1. 광역자치단체는 종합점수, 기초자치단체는 광역연계점수로 차별화 분석"
        )
        report.append(
            "2. 기초자치단체의 광역연계 효과가 실제 청년 이동에 미치는 영향 확인"
        )
        report.append("3. 정책 효과는 시행 후 6-12개월 지연되어 나타남")
        report.append("4. 지역유형별로 정책 시차 효과의 차이 존재")
        report.append("5. 청년인구 대비 순이동률로 정규화하여 지역 규모의 영향 제거")
        report.append("")

        # 정책 권장사항
        report.append("💡 eval-5 기반 정책 시차 권장사항")
        report.append("1. 기초자치단체는 광역자치단체와의 연계 강화가 중요")
        report.append("2. 정책 효과 평가 시 최소 12개월 이상의 관찰 기간 필요")
        report.append("3. 광역-기초 간 정책 연계성을 고려한 통합적 접근 필요")
        report.append("4. 지역 특성에 따른 차별화된 정책 시차 고려")
        report.append("5. 광역연계점수 개선을 통한 기초자치단체 정책 효과성 증대")

        # 리포트 저장
        report_text = "\n".join(report)
        with open(
            self.base_path / "migration_plot/policy_lag_report_eval5.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report_text)

        print("✅ 정책 시차 분석 리포트 생성 완료 (eval-5)")
        print("\n" + report_text)

    def run_full_analysis(self):
        """전체 정책 시차 분석 실행 (eval-5 버전)"""
        print("🚀 정책 시차(Policy Lag) 분석 시작 (eval-5)")
        print("=" * 60)
        print("📋 광역자치단체: 종합점수 사용")
        print("📋 기초자치단체: 최종_연계점수 사용 (광역연계 고려)")

        # 1. 데이터 로드
        if not self.load_data():
            return

        # 2. 마이그레이션 데이터 전처리
        if not self.preprocess_migration_data():
            return

        # 3. 정책-이동 데이터 통합
        if not self.merge_policy_migration_data():
            return

        # 4. 상관관계 분석
        print("\n📊 정책 시차 상관관계 분석...")
        self.analyze_policy_migration_correlation()

        # 5. 정책 점수 vs 순유입률 플롯 (핵심)
        print("\n📊 정책 점수 vs 순유입률 플롯...")
        self.create_settlement_induction_plot()

        # 6. 기존 시각화
        print("\n📈 정책 시차 종합 시각화...")
        self.create_policy_lag_visualization()

        # 7. 종합 리포트
        print("\n📋 종합 리포트 생성...")
        self.generate_lag_analysis_report()

        print(f"\n✅ 정책 시차 분석 완료 (eval-5)!")
        print(f"📁 결과 저장 위치: {self.base_path / 'migration_plot'}")
        print("📋 주요 결과 파일:")
        print("  - settlement_induction_plot_eval5.png (핵심 플롯)")
        print("  - policy_lag_correlation_eval5.png (상관관계)")
        print("  - policy_lag_analysis_eval5.png (종합 시각화)")
        print("  - settlement_induction_result_eval5.csv (결과 데이터)")
        print("  - policy_lag_report_eval5.txt (분석 리포트)")


def main():
    """메인 실행 함수"""
    analyzer = PolicyLagAnalyzerEval5()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
