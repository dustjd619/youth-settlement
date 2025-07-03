"""
eval-4 결과를 사용한 정책 시차 청년 이동 분석 모듈
==================================================

이 모듈은 eval-4 결과를 바탕으로 정책 시행과 청년 인구 이동 간의 시간차를 고려하여 분석합니다.
- 광역자치단체: 종합점수 사용
- 기초자치단체: 최종_연계점수 사용 (광역연계 고려)
- 분석 기간: 2023년 8월 ~ 2024년 7월 (12개월)
"""

import os
import warnings
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


class PolicyLagAnalyzerEval4:
    """eval-4 결과를 사용한 정책 시차 청년 이동 분석 클래스"""

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

        # 결과 저장 디렉토리 생성
        self.result_dir = self.base_path / "migration_plot/eval-4_result"
        os.makedirs(self.result_dir, exist_ok=True)

        print(f"📅 분석 기간: {self.start_year_month} ~ {self.end_year_month}")
        print(f"🔗 사용 데이터: eval-4 결과 (광역=종합점수, 기초=최종_연계점수)")

    def load_data(self):
        """eval-4 정책 데이터와 마이그레이션 데이터 로드"""
        try:
            # 광역자치단체 정책 데이터 로드
            metropolitan_policy_file = (
                self.base_path
                / "data/policy_eval/eval-4_result/광역_청년정책_종합평가결과.csv"
            )

            # 기초자치단체 정책 데이터 로드 (eval-4 결과 사용)
            municipal_policy_file = (
                self.base_path
                / "data/policy_eval/eval-4_result/기초_최종평가결과(광역연계).csv"
            )

            policy_data_list = []

            print(f"광역 파일 경로: {metropolitan_policy_file}")
            print(f"기초 파일 경로: {municipal_policy_file}")
            print(f"광역 파일 존재: {metropolitan_policy_file.exists()}")
            print(f"기초 파일 존재: {municipal_policy_file.exists()}")

            # 광역자치단체 데이터 로드
            if metropolitan_policy_file.exists():
                metro_data = pd.read_csv(metropolitan_policy_file, encoding="utf-8-sig")
                metro_data["지역유형"] = "광역자치단체"
                metro_data["점수_컬럼"] = "종합점수"
                metro_data["사용_점수"] = metro_data["종합점수"]
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

                # 필요한 컬럼들이 있는지 확인하고 없으면 기본값 설정
                required_cols = ["전략적_강도", "청년인구", "전체인구"]
                for col in required_cols:
                    if col not in muni_data.columns:
                        if col == "전략적_강도":
                            muni_data[col] = muni_data.get("전략적_강도", 0)
                        elif col == "청년인구":
                            muni_data[col] = 50000  # 기본값
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

            return self.load_migration_data()

        except Exception as e:
            print(f"❌ 데이터 로드 중 오류 발생: {e}")
            return False

    def load_migration_data(self):
        """마이그레이션 데이터 로드"""
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

        print(f"✅ 총 {len(target_files)}개 마이그레이션 파일 발견")
        return True

    def preprocess_migration_data(self):
        """파일별로 각 지역의 컬럼합(전입), row합(전출) 누적 방식으로 순이동 계산"""
        migration_dir = self.base_path / "data/migration/청년 인구 이동량_consolidated"

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
            if pd.isna(name):
                return ""
            return str(name).strip().replace("  ", " ")

        self.policy_data["지역명_정제"] = self.policy_data["지역명"].apply(
            normalize_region_name
        )
        self.analysis_period_data["지역명_정제"] = self.analysis_period_data[
            "지역명"
        ].apply(normalize_region_name)

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
            ) * 100

            # 결과 CSV로 저장
            save_path = self.result_dir / "settlement_induction_result_eval4.csv"
            self.merged_data.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"✅ 통합 결과 CSV 저장 완료: {save_path}")
            return True
        else:
            print("❌ 매칭되는 지역이 없습니다.")
            return False

    def create_settlement_induction_plot(self):
        """정책 점수 vs 순유입률 플롯 생성"""
        if self.merged_data is None:
            print("❌ 통합 데이터가 없습니다.")
            return

        # 지역유형별로 데이터 분리
        metropolitan_data = self.merged_data[
            self.merged_data["지역유형"] == "광역자치단체"
        ].copy()
        municipal_data = self.merged_data[
            self.merged_data["지역유형"] == "기초자치단체"
        ].copy()

        # 3x1 서브플롯 생성
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # 1. 광역자치단체 플롯
        if len(metropolitan_data) > 0 and "종합점수" in metropolitan_data.columns:
            valid_metro = metropolitan_data[
                ["종합점수", "순이동률_인구대비", "지역명_이동"]
            ].dropna()

            if len(valid_metro) > 0:
                axes[0].scatter(
                    valid_metro["종합점수"],
                    valid_metro["순이동률_인구대비"],
                    alpha=0.7,
                    s=120,
                    c="steelblue",
                    edgecolors="white",
                    linewidth=1,
                )

                # 회귀선 추가
                if len(valid_metro) > 2:
                    z = np.polyfit(
                        valid_metro["종합점수"], valid_metro["순이동률_인구대비"], 1
                    )
                    p = np.poly1d(z)
                    axes[0].plot(
                        valid_metro["종합점수"],
                        p(valid_metro["종합점수"]),
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                    )

                    # 상관계수 계산
                    corr_coef, p_value = stats.pearsonr(
                        valid_metro["종합점수"], valid_metro["순이동률_인구대비"]
                    )
                    axes[0].text(
                        0.05,
                        0.95,
                        f"r = {corr_coef:.3f}\np = {p_value:.3f}\nn = {len(valid_metro)}",
                        transform=axes[0].transAxes,
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9
                        ),
                    )

                axes[0].set_xlabel("정책 종합점수", fontsize=12)
                axes[0].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[0].set_title(
                    f"광역자치단체 - 정책 종합점수 vs 청년 순유입률", fontsize=14
                )
                axes[0].grid(True, alpha=0.3)

        # 2. 기초자치단체 플롯
        if len(municipal_data) > 0 and "최종_연계점수" in municipal_data.columns:
            valid_muni = municipal_data[
                ["최종_연계점수", "순이동률_인구대비", "지역명_이동"]
            ].dropna()

            if len(valid_muni) > 0:
                axes[1].scatter(
                    valid_muni["최종_연계점수"],
                    valid_muni["순이동률_인구대비"],
                    alpha=0.6,
                    s=60,
                    c="forestgreen",
                    edgecolors="white",
                    linewidth=0.5,
                )

                # 회귀선 추가
                if len(valid_muni) > 2:
                    z = np.polyfit(
                        valid_muni["최종_연계점수"], valid_muni["순이동률_인구대비"], 1
                    )
                    p = np.poly1d(z)
                    axes[1].plot(
                        valid_muni["최종_연계점수"],
                        p(valid_muni["최종_연계점수"]),
                        "r--",
                        alpha=0.8,
                        linewidth=2,
                    )

                    # 상관계수 계산
                    corr_coef, p_value = stats.pearsonr(
                        valid_muni["최종_연계점수"], valid_muni["순이동률_인구대비"]
                    )
                    axes[1].text(
                        0.05,
                        0.95,
                        f"r = {corr_coef:.3f}\np = {p_value:.3f}\nn = {len(valid_muni)}",
                        transform=axes[1].transAxes,
                        fontsize=10,
                        bbox=dict(
                            boxstyle="round,pad=0.5", facecolor="white", alpha=0.9
                        ),
                    )

                axes[1].set_xlabel("최종 연계점수 (광역연계)", fontsize=12)
                axes[1].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
                axes[1].set_title(
                    f"기초자치단체 - 최종 연계점수 vs 청년 순유입률", fontsize=14
                )
                axes[1].grid(True, alpha=0.3)

        # 3. 전체(광역+기초) 플롯
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

                # 상관계수 계산
                corr_coef, p_value = stats.pearsonr(
                    valid_all["사용_점수"], valid_all["순이동률_인구대비"]
                )
                axes[2].text(
                    0.05,
                    0.95,
                    f"전체 r = {corr_coef:.3f}\np = {p_value:.3f}\nn = {len(valid_all)}",
                    transform=axes[2].transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
                )

            axes[2].set_xlabel(
                "정책 점수 (광역=종합점수, 기초=최종_연계점수)", fontsize=12
            )
            axes[2].set_ylabel("순유입률 (청년인구 대비 %)", fontsize=12)
            axes[2].set_title(
                f"전체(광역+기초) - 정책 점수 vs 청년 순유입률", fontsize=14
            )
            axes[2].grid(True, alpha=0.3)

        plt.suptitle("정책 점수 vs 청년 순유입률 (eval-4)", fontsize=16, y=0.98)
        plt.tight_layout()

        # 저장
        save_path = self.result_dir / "settlement_induction_plot_eval4.png"
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 플롯 저장 성공: {save_path}")
        except Exception as e:
            print(f"❌ 플롯 저장 실패: {e}")

        plt.close()

    def run_full_analysis(self):
        """전체 정책 시차 분석 실행 (eval-4 버전)"""
        print("🚀 정책 시차(Policy Lag) 분석 시작 (eval-4)")
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

        # 4. 정책 점수 vs 순유입률 플롯 (핵심)
        print("\n📊 정책 점수 vs 순유입률 플롯...")
        self.create_settlement_induction_plot()

        print(f"\n✅ 정책 시차 분석 완료 (eval-4)!")
        print(f"📁 결과 저장 위치: {self.result_dir}")


def main():
    """메인 실행 함수"""
    analyzer = PolicyLagAnalyzerEval4()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
