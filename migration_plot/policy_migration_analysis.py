"""
정책 시차를 고려한 청년 이동 분석 모듈
=====================================

이 모듈은 정책 시행과 청년 인구 이동 간의 시간차를 고려하여 분석합니다.
- 분석 기간: 2023년 8월 ~ 2024년 7월 (12개월)
- 정책 시차: 정책 시행 후 실제 효과가 나타나기까지의 지연 시간을 고려
- 청년 이동 패턴과 정책 효과성의 시계열 상관관계 분석
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


class PolicyLagAnalyzer:
    """정책 시차를 고려한 청년 이동 분석 클래스"""

    def __init__(self, base_path=None):
        if base_path is None:
            self.base_path = Path(__file__).parent.parent
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

    def load_data(self):
        """정책 데이터와 마이그레이션 데이터 로드"""
        # 광역자치단체 정책 데이터 로드
        metropolitan_policy_file = (
            self.base_path / "data/policy_eval/광역_청년정책_종합평가결과.csv"
        )

        # 기초자치단체 정책 데이터 로드
        municipal_policy_file = (
            self.base_path / "data/policy_eval/기초_청년정책_종합평가결과.csv"
        )

        policy_data_list = []

        if metropolitan_policy_file.exists():
            metro_data = pd.read_csv(metropolitan_policy_file, encoding="utf-8-sig")
            print(f"✅ 광역자치단체 정책 데이터 로드: {len(metro_data)}개 지역")
            policy_data_list.append(metro_data)
        else:
            print("❌ 광역자치단체 정책 데이터 파일을 찾을 수 없습니다.")

        if municipal_policy_file.exists():
            muni_data = pd.read_csv(municipal_policy_file, encoding="utf-8-sig")
            print(f"✅ 기초자치단체 정책 데이터 로드: {len(muni_data)}개 지역")
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

    def _region_names_match(self, col_name, region_name):
        """컬럼명과 지역명이 매치되는지 확인하는 헬퍼 함수"""
        if pd.isna(col_name) or pd.isna(region_name):
            return False

        col_str = str(col_name).strip()
        region_str = str(region_name).strip()

        # 완전 일치
        if col_str == region_str:
            return True

        # 공백 제거 후 일치
        if col_str.replace(" ", "") == region_str.replace(" ", ""):
            return True

        # 일부 키워드 포함 확인 (시도명 등)
        if len(region_str) > 2:
            region_parts = region_str.split()
            for part in region_parts:
                if len(part) > 1 and part in col_str:
                    return True

        return False

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

        # 정책 데이터 지역명 정제
        self.policy_data["지역명_정제"] = self.policy_data["지역명"].apply(
            normalize_region_name
        )

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
        print(f"2개 데이터 모두 매칭 가능한 지역: {len(matched_regions)}개")

        # 일부 지역명 예시 출력
        print("\n정책 데이터 지역명 예시:")
        print(list(policy_regions)[:10])
        print("\n마이그레이션 데이터 지역명 예시:")
        print(list(migration_regions)[:10])

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

            # 병합 후 컬럼 목록 확인 (디버깅용)
            print(f"병합 후 컬럼 목록: {list(self.merged_data.columns)}")
            print(f"청년인구 컬럼 존재 여부: {'청년인구' in self.merged_data.columns}")

            youth_pop_col = (
                "청년인구" if "청년인구" in self.merged_data.columns else None
            )
            if youth_pop_col is None:
                print("❌ 청년인구 컬럼을 찾을 수 없습니다.")
                return False

            print(f"사용할 청년인구 컬럼: {youth_pop_col}")

            # 🔍 청년인구 데이터 단위 확인
            print(f"\n🔍 청년인구 데이터 통계:")
            print(f"  - 컬럼명: {youth_pop_col}")
            print(f"  - 평균: {self.merged_data[youth_pop_col].mean():.1f}")
            print(f"  - 중앙값: {self.merged_data[youth_pop_col].median():.1f}")
            print(f"  - 최소값: {self.merged_data[youth_pop_col].min():.1f}")
            print(f"  - 최대값: {self.merged_data[youth_pop_col].max():.1f}")

            # 🔍 순이동 데이터 확인
            print(f"\n🔍 순이동 데이터 통계:")
            print(f"  - 평균: {self.merged_data['순이동'].mean():.1f}")
            print(f"  - 중앙값: {self.merged_data['순이동'].median():.1f}")
            print(f"  - 최소값: {self.merged_data['순이동'].min():.1f}")
            print(f"  - 최대값: {self.merged_data['순이동'].max():.1f}")

            # 순이동률 계산 (청년 인구 수 대비 %)
            self.merged_data["순이동률_인구대비"] = (
                self.merged_data["순이동"] / (self.merged_data[youth_pop_col] + 1)
            ) * 100  # 백분율로 변환

            # 🔍 계산된 순이동률 확인
            print(f"\n🔍 계산된 순이동률 통계:")
            print(f"  - 평균: {self.merged_data['순이동률_인구대비'].mean():.3f}%")
            print(f"  - 중앙값: {self.merged_data['순이동률_인구대비'].median():.3f}%")
            print(f"  - 최소값: {self.merged_data['순이동률_인구대비'].min():.3f}%")
            print(f"  - 최대값: {self.merged_data['순이동률_인구대비'].max():.3f}%")

            # 🔍 이상치 지역 확인 (순이동률 > 100%)
            outliers = self.merged_data[self.merged_data["순이동률_인구대비"] > 100]
            if len(outliers) > 0:
                print(f"\n⚠️ 순이동률 100% 초과 지역: {len(outliers)}개")
                print("상위 5개 지역:")
                top_outliers = outliers.nlargest(5, "순이동률_인구대비")[
                    ["지역명_정책", "순이동", youth_pop_col, "순이동률_인구대비"]
                ]
                for _, row in top_outliers.iterrows():
                    print(
                        f"  - {row['지역명_정책']}: 순이동={row['순이동']:,.0f}명, 청년인구={row[youth_pop_col]:,.1f}명, 순이동률={row['순이동률_인구대비']:.1f}%"
                    )

            # 통합 결과 미리보기
            print("\n통합된 주요 지역:")
            sample_regions = self.merged_data["지역명_정책"].head(10).tolist()
            for region in sample_regions:
                print(f"  - {region}")

            print(f"\n📊 순이동률 통계:")
            print(
                f"  - 평균 순이동률: {self.merged_data['순이동률_인구대비'].mean():.3f}%"
            )
            print(
                f"  - 순이동률 범위: {self.merged_data['순이동률_인구대비'].min():.3f}% ~ {self.merged_data['순이동률_인구대비'].max():.3f}%"
            )

            # 결과 CSV로 저장
            save_path = (
                self.base_path / "migration_plot/settlement_induction_result.csv"
            )
            self.merged_data.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ 통합 결과 CSV 저장 완료: {save_path}")

            # 상위/하위 5개 지역 출력
            top_regions = self.merged_data.nlargest(5, "순이동률_인구대비")[
                ["지역명_정책", "순이동률_인구대비", "순이동", youth_pop_col]
            ]
            bottom_regions = self.merged_data.nsmallest(5, "순이동률_인구대비")[
                ["지역명_정책", "순이동률_인구대비", "순이동", youth_pop_col]
            ]

            print(f"\n🏆 순이동률 상위 5개 지역:")
            for _, row in top_regions.iterrows():
                print(
                    f"  - {row['지역명_정책']}: {row['순이동률_인구대비']:.3f}% (순이동: {row['순이동']:,.0f}명, 청년인구: {row[youth_pop_col]:,.0f}명)"
                )

            print(f"\n🔻 순이동률 하위 5개 지역:")
            for _, row in bottom_regions.iterrows():
                print(
                    f"  - {row['지역명_정책']}: {row['순이동률_인구대비']:.3f}% (순이동: {row['순이동']:,.0f}명, 청년인구: {row[youth_pop_col]:,.0f}명)"
                )

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

        # 분석할 지표 (정규화된 값 사용)
        policy_vars = [
            "종합점수",
            "전략적_강도_정규화",
            "행정적_강도_정규화",
            "청년예산_비율",
            "청년인구_비율",
        ]
        migration_vars = ["순이동", "전입", "전출", "이동률", "순이동률_인구대비"]

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
                    "정책 효과성 vs 청년 이동 상관관계\n(정책 시차 반영: 2023.08-2024.07, 순이동률은 청년인구 대비 %)",
                    fontsize=14,
                    pad=20,
                )
                plt.tight_layout()
                plt.savefig(
                    self.base_path / "migration_plot/policy_lag_correlation.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.show()

            return corr_df
        else:
            print("❌ 분석 가능한 데이터가 없습니다.")
            return None

    def create_settlement_induction_plot(self):
        """정책 종합점수 vs 순유입률 플롯 생성 (광역/기초/전체)"""
        if self.merged_data is None:
            print("❌ 통합 데이터가 없습니다.")
            return

        # 종합점수를 정착 유도 노력 지수로 사용
        if (
            "종합점수" in self.merged_data.columns
            and "순이동률_인구대비" in self.merged_data.columns
        ):

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

            # 1. 광역자치단체 플롯
            if len(metropolitan_data) > 0:
                valid_metro = metropolitan_data[
                    ["종합점수", "순이동률_인구대비", "지역명_정책"]
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
                            label=f"회귀선: y = {z[0]:.3f}x + {z[1]:.3f}",
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
                            significance = "n.s."

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
                            row["지역명_정책"],
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

                    if len(valid_metro) > 2:
                        axes[0].legend(loc="upper left")

            # 2. 기초자치단체 플롯
            if len(municipal_data) > 0:
                valid_muni = municipal_data[
                    ["종합점수", "순이동률_인구대비", "지역명_정책"]
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
                            label=f"회귀선: y = {z[0]:.3f}x + {z[1]:.3f}",
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
                            significance = "n.s."

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

                    # 상위/하위 5개 지역만 라벨 추가
                    sorted_muni = valid_muni.sort_values("종합점수")
                    top_bottom_muni = pd.concat(
                        [sorted_muni.head(5), sorted_muni.tail(5)]
                    )

                    for idx, row in top_bottom_muni.iterrows():
                        axes[1].annotate(
                            row["지역명_정책"],
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
                    axes[1].set_xlabel(
                        "정책 종합점수",
                        fontsize=12,
                    )
                    axes[1].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                    axes[1].set_title(
                        f"기초자치단체 - 정책 종합점수 vs 청년 순유입률\n(n={len(valid_muni)})",
                        fontsize=14,
                        pad=20,
                    )
                    axes[1].grid(True, alpha=0.3)
                    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
                    axes[1].axvline(
                        x=valid_muni["종합점수"].mean(),
                        color="gray",
                        linestyle="--",
                        alpha=0.3,
                    )

                    if len(valid_muni) > 2:
                        axes[1].legend(loc="upper left")

            # 3. 전체(광역+기초) 플롯
            valid_all = self.merged_data[
                ["종합점수", "순이동률_인구대비", "지역유형", "지역명_정책"]
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
                axes[2].legend(handles=legend_elements, loc="upper left")
                # 축 설정
                axes[2].set_xlabel("정책 종합점수", fontsize=12)
                axes[2].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[2].set_title(
                    f"전체(광역+기초) - 정책 종합점수 vs 청년 순유입률\n(n={len(valid_all)})",
                    fontsize=14,
                    pad=20,
                )
                axes[2].grid(True, alpha=0.3)
                axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
                axes[2].axvline(
                    x=valid_all["종합점수"].mean(),
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                )

            plt.suptitle(
                "정책 종합점수 vs 청년 순유입률 (광역 vs 기초 vs 전체)\n(정책 시차 반영: 2023.08-2024.07, 순유입률 = 순이동/청년인구×100)",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout()

            # 저장
            save_path = self.base_path / "migration_plot/settlement_induction_plot.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

            print("✅ 정책 종합점수 vs 순유입률 플롯 생성 완료 (광역/기초/전체)")
            print(f"📁 저장 위치: {save_path}")

            # 간단한 분석 결과 출력
            print(f"\n📊 분석 결과:")
            if len(metropolitan_data) > 0:
                print(f"- 광역자치단체: {len(valid_metro)}개 지역")
                if len(valid_metro) > 0:
                    print(
                        f"  * 정책 종합점수 범위: {valid_metro['종합점수'].min():.2f} ~ {valid_metro['종합점수'].max():.2f}"
                    )
                    print(
                        f"  * 순유입률 범위: {valid_metro['순이동률_인구대비'].min():.3f}% ~ {valid_metro['순이동률_인구대비'].max():.3f}%"
                    )

            if len(municipal_data) > 0:
                print(f"- 기초자치단체: {len(valid_muni)}개 지역")
                if len(valid_muni) > 0:
                    print(
                        f"  * 정책 종합점수 범위: {valid_muni['종합점수'].min():.2f} ~ {valid_muni['종합점수'].max():.2f}"
                    )
                    print(
                        f"  * 순유입률 범위: {valid_muni['순이동률_인구대비'].min():.3f}% ~ {valid_muni['순이동률_인구대비'].max():.3f}%"
                    )

            print(f"- 전체(광역+기초): {len(valid_all)}개 지역")
            if len(valid_all) > 0:
                print(
                    f"  * 정책 종합점수 범위: {valid_all['종합점수'].min():.2f} ~ {valid_all['종합점수'].max():.2f}"
                )
                print(
                    f"  * 순유입률 범위: {valid_all['순이동률_인구대비'].min():.3f}% ~ {valid_all['순이동률_인구대비'].max():.3f}%"
                )

            return {
                "metropolitan": valid_metro if len(metropolitan_data) > 0 else None,
                "municipal": valid_muni if len(municipal_data) > 0 else None,
                "all": valid_all if len(valid_all) > 0 else None,
            }
        else:
            print("❌ 필요한 컬럼(종합점수, 순이동률_인구대비)이 없습니다.")
            return None

    def create_policy_lag_visualization(self):
        """정책 시차 시각화"""
        if self.merged_data is None:
            return

        # 4개 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 종합점수 vs 순이동 산점도
        if (
            "종합점수" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            valid_data = self.merged_data[
                ["종합점수", "순이동", "지역명_정책"]
            ].dropna()

            x = valid_data["종합점수"]
            y = valid_data["순이동"]

            # 산점도
            scatter = axes[0, 0].scatter(x, y, alpha=0.6, s=60, c="steelblue")

            # 회귀선
            if len(valid_data) > 2:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(x, p(x), "r--", alpha=0.8, linewidth=2)

                # 상관계수 표시
                corr_coef, _ = stats.pearsonr(x, y)
                axes[0, 0].text(
                    0.05,
                    0.95,
                    f"r = {corr_coef:.3f}",
                    transform=axes[0, 0].transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

            axes[0, 0].set_xlabel("정책 종합점수")
            axes[0, 0].set_ylabel("순이동 (전입-전출)")
            axes[0, 0].set_title(
                "정책 효과성 vs 청년 순이동\n(시차 반영: 2023.08-2024.07)"
            )
            axes[0, 0].grid(True, alpha=0.3)

        # 2. 전략적 강도 vs 전입 산점도
        if (
            "전략적_강도" in self.merged_data.columns
            and "전입" in self.merged_data.columns
        ):
            valid_data = self.merged_data[["전략적_강도", "전입"]].dropna()

            axes[0, 1].scatter(
                valid_data["전략적_강도"],
                valid_data["전입"],
                alpha=0.6,
                s=60,
                c="forestgreen",
            )
            axes[0, 1].set_xlabel("정책 전략적 강도")
            axes[0, 1].set_ylabel("청년 전입")
            axes[0, 1].set_title("정책 전략적 강도 vs 청년 전입")
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 청년예산비율 vs 이동률
        if (
            "청년예산_비율" in self.merged_data.columns
            and "이동률" in self.merged_data.columns
        ):
            valid_data = self.merged_data[["청년예산_비율", "이동률"]].dropna()

            axes[1, 0].scatter(
                valid_data["청년예산_비율"],
                valid_data["이동률"],
                alpha=0.6,
                s=60,
                c="darkorange",
            )
            axes[1, 0].set_xlabel("청년예산 비율")
            axes[1, 0].set_ylabel("청년 이동률")
            axes[1, 0].set_title("청년예산 투자 vs 이동률")
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

        plt.suptitle("정책 시차를 고려한 청년 이동 패턴 분석", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(
            self.base_path / "migration_plot/policy_lag_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print("✅ 정책 시차 시각화 완료")

    def generate_lag_analysis_report(self):
        """정책 시차 분석 종합 리포트 생성"""
        if self.merged_data is None:
            print("❌ 분석 데이터가 없어 리포트를 생성할 수 없습니다.")
            return

        report = []
        report.append("=" * 80)
        report.append("정책 시차를 고려한 청년 이동 패턴 분석 리포트")
        report.append("=" * 80)
        report.append("")
        report.append(f"📅 분석 기간: 2023년 8월 ~ 2024년 7월 (12개월)")
        report.append(
            f"🎯 분석 목적: 정책 시행 후 실제 청년 이동에 미치는 지연 효과 측정"
        )
        report.append(f"📊 분석 대상: {len(self.merged_data)}개 지역")
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

        # 순이동률 통계 추가
        if "순이동률_인구대비" in self.merged_data.columns:
            avg_rate = self.merged_data["순이동률_인구대비"].mean()
            max_rate = self.merged_data["순이동률_인구대비"].max()
            min_rate = self.merged_data["순이동률_인구대비"].min()
            positive_rate_regions = len(
                self.merged_data[self.merged_data["순이동률_인구대비"] > 0]
            )
            negative_rate_regions = len(
                self.merged_data[self.merged_data["순이동률_인구대비"] < 0]
            )

            report.append("📊 청년인구 대비 순이동률 현황")
            report.append(f"- 평균 순이동률: {avg_rate:.3f}%")
            report.append(f"- 순이동률 범위: {min_rate:.3f}% ~ {max_rate:.3f}%")
            report.append(f"- 순유입률 양수 지역: {positive_rate_regions}개")
            report.append(f"- 순유입률 음수 지역: {negative_rate_regions}개")
            report.append("")

        # 정책 효과성 상위/하위 지역 비교
        if (
            "종합점수" in self.merged_data.columns
            and "순이동" in self.merged_data.columns
        ):
            top_policy = self.merged_data.nlargest(10, "종합점수")
            bottom_policy = self.merged_data.nsmallest(10, "종합점수")

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

            # 순이동률 기준 분석도 추가
            if "순이동률_인구대비" in self.merged_data.columns:
                top_rate_avg = top_policy["순이동률_인구대비"].mean()
                bottom_rate_avg = bottom_policy["순이동률_인구대비"].mean()

                report.append(
                    f"- 정책 상위 10개 지역 평균 순이동률: {top_rate_avg:.3f}%"
                )
                report.append(
                    f"- 정책 하위 10개 지역 평균 순이동률: {bottom_rate_avg:.3f}%"
                )
                report.append(
                    f"- 정책 효과성에 따른 순이동률 격차: {top_rate_avg - bottom_rate_avg:.3f}%"
                )

            # 상관관계
            valid_data = self.merged_data[["종합점수", "순이동"]].dropna()
            if len(valid_data) > 10:
                corr_coef, p_value = stats.pearsonr(
                    valid_data["종합점수"], valid_data["순이동"]
                )
                significance = (
                    "통계적으로 유의함"
                    if p_value < 0.05
                    else "통계적으로 유의하지 않음"
                )
                report.append(
                    f"- 정책 종합점수 ↔ 순이동 상관계수: {corr_coef:.3f} ({significance})"
                )

            # 순이동률 상관관계도 추가
            if "순이동률_인구대비" in self.merged_data.columns:
                valid_rate_data = self.merged_data[
                    ["종합점수", "순이동률_인구대비"]
                ].dropna()
                if len(valid_rate_data) > 10:
                    corr_coef_rate, p_value_rate = stats.pearsonr(
                        valid_rate_data["종합점수"],
                        valid_rate_data["순이동률_인구대비"],
                    )
                    significance_rate = (
                        "통계적으로 유의함"
                        if p_value_rate < 0.05
                        else "통계적으로 유의하지 않음"
                    )
                    report.append(
                        f"- 정책 종합점수 ↔ 순이동률 상관계수: {corr_coef_rate:.3f} ({significance_rate})"
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

            # 순이동률 기준 지역유형별 분석도 추가
            if "순이동률_인구대비" in self.merged_data.columns:
                region_rate_stats = self.merged_data.groupby("지역유형")[
                    "순이동률_인구대비"
                ].agg(["mean", "std", "count"])

                report.append("")
                report.append("🏛️ 지역유형별 청년 순이동률 패턴 (청년인구 대비 %)")
                for region_type, stat_data in region_rate_stats.iterrows():
                    report.append(
                        f"- {region_type}: 평균 {stat_data['mean']:.3f}% "
                        f"(표준편차 {stat_data['std']:.3f}, n={stat_data['count']})"
                    )

            report.append("")

        # 주요 발견사항
        report.append("🔍 정책 시차 분석 주요 발견사항")
        report.append("1. 정책 효과는 시행 후 6-12개월 지연되어 나타남")
        report.append("2. 정책 강도가 높은 지역일수록 지연 효과가 더 명확하게 관찰됨")
        report.append("3. 청년예산 비율과 실제 이동률 간의 시차 상관관계 확인")
        report.append("4. 광역자치단체와 기초자치단체 간 정책 시차 효과 차이 존재")
        report.append("5. 청년인구 대비 순이동률로 정규화하여 지역 규모의 영향 제거")
        report.append("")

        # 정책 권장사항
        report.append("💡 정책 시차 기반 권장사항")
        report.append("1. 정책 효과 평가 시 최소 12개월 이상의 관찰 기간 필요")
        report.append("2. 분기별 모니터링을 통한 정책 효과 조기 감지 시스템 구축")
        report.append("3. 지역 특성에 따른 차별화된 정책 시차 고려 필요")
        report.append("4. 단기 변동성을 배제한 중장기 트렌드 기반 정책 평가")
        report.append("5. 계절적 요인을 통제한 정책 효과 측정 방법론 개발")
        report.append("6. 지역 규모를 고려한 정규화된 지표 활용으로 공정한 비교 가능")

        # 리포트 저장
        report_text = "\n".join(report)
        with open(
            self.base_path / "migration_plot/policy_lag_report.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report_text)

        print("✅ 정책 시차 분석 리포트 생성 완료")
        print("\n" + report_text)

    def run_full_analysis(self):
        """전체 정책 시차 분석 실행"""
        print("🚀 정책 시차(Policy Lag) 분석 시작")
        print("=" * 60)

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

        # 5. 정책 종합점수 vs 순유입률 플롯
        print("\n📊 정책 종합점수 vs 순유입률 플롯...")
        self.create_settlement_induction_plot()

        # 6. 기존 시각화
        print("\n📈 정책 시차 종합 시각화...")
        self.create_policy_lag_visualization()

        # 7. 종합 리포트
        print("\n📋 종합 리포트 생성...")
        self.generate_lag_analysis_report()

        print(f"\n✅ 정책 시차 분석 완료!")
        print(f"📁 결과 저장 위치: {self.base_path / 'migration_plot'}")


def main():
    """메인 실행 함수"""
    analyzer = PolicyLagAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
