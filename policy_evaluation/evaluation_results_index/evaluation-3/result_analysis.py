import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = [
    "Arial Unicode MS",
    "Malgun Gothic",
    "AppleGothic",
    "Noto Sans CJK KR",
]
plt.rcParams["axes.unicode_minus"] = False


class YouthPolicyResultAnalysis:
    def __init__(self):
        self.df = None
        self.numerical_cols = []

    def load_data(self, file_path="전국_청년정책_종합평가결과_v6.csv"):
        """데이터 로드"""
        self.df = pd.read_csv(file_path, encoding="utf-8-sig")

        # 수치형 컬럼 식별
        self.numerical_cols = [
            "행정적_강도",
            "집중도_지수",
            "청년예산_비율",
            "청년인구_비율",
            "재정자립도",
            "총예산_백만원",
            "청년정책예산_백만원",
            "전략적_강도",
            "엔트로피",
            "정규화_엔트로피",
            "총정책수",
            "정책분야수",
            "정책페널티",
            "분야페널티",
            "행정적_강도_정규화",
            "전략적_강도_정규화",
            "종합점수",
        ]

        print(f"📊 데이터 로드 완료: {len(self.df)}개 지역")
        print(
            f"📊 광역자치단체: {len(self.df[self.df['지역유형'] == '광역자치단체'])}개"
        )
        print(
            f"📊 기초자치단체: {len(self.df[self.df['지역유형'] == '기초자치단체'])}개"
        )

    def basic_statistics(self):
        """기본 통계량 분석"""
        print("\n" + "=" * 80)
        print("📈 기본 통계량 분석")
        print("=" * 80)

        # 전체 통계
        desc = self.df[self.numerical_cols].describe()
        print("\n📊 전체 지역 기본 통계량:")
        print(desc.round(4))

        # 지역유형별 통계
        print("\n📊 지역유형별 주요 지표 평균:")
        regional_stats = (
            self.df.groupby("지역유형")[
                [
                    "종합점수",
                    "행정적_강도",
                    "전략적_강도",
                    "청년예산_비율",
                    "청년인구_비율",
                    "재정자립도",
                ]
            ]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        print(regional_stats)

        # 상위/하위 지역 분석
        print(f"\n🏆 종합점수 상위 10개 지역:")
        top_10 = self.df.nlargest(10, "종합점수")[
            ["전체순위", "지역명", "지역유형", "종합점수"]
        ]
        print(top_10.to_string(index=False))

        print(f"\n📉 종합점수 하위 10개 지역:")
        bottom_10 = self.df.nsmallest(10, "종합점수")[
            ["전체순위", "지역명", "지역유형", "종합점수"]
        ]
        print(bottom_10.to_string(index=False))

    def correlation_analysis(self):
        """상관관계 분석"""
        print("\n" + "=" * 80)
        print("🔍 상관관계 분석")
        print("=" * 80)

        # 주요 지표간 상관관계
        key_indicators = [
            "종합점수",
            "행정적_강도",
            "전략적_강도",
            "청년예산_비율",
            "청년인구_비율",
            "재정자립도",
            "총정책수",
            "정책분야수",
        ]

        corr_matrix = self.df[key_indicators].corr()

        print("\n📊 주요 지표간 상관관계 (종합점수 기준):")
        corr_with_total = corr_matrix["종합점수"].sort_values(ascending=False)
        for idx, corr in corr_with_total.items():
            if idx != "종합점수":
                print(f"{idx:15s}: {corr:6.3f}")

        # 상관관계 히트맵 생성
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            fmt=".3f",
        )
        plt.title("청년정책 주요 지표간 상관관계", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()

        # 강한 상관관계 (|r| > 0.5) 찾기
        print(f"\n🔥 강한 상관관계 (|r| > 0.5):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                    )

        for var1, var2, corr in sorted(
            strong_corr, key=lambda x: abs(x[2]), reverse=True
        ):
            print(f"{var1} ↔ {var2}: {corr:.3f}")

    def regional_analysis(self):
        """지역별 심층 분석"""
        print("\n" + "=" * 80)
        print("🗺️ 지역별 심층 분석")
        print("=" * 80)

        # 광역자치단체 순위
        metro = self.df[self.df["지역유형"] == "광역자치단체"].sort_values(
            "종합점수", ascending=False
        )
        print(f"\n🏛️ 광역자치단체 종합 순위:")
        metro_display = metro[
            [
                "광역순위",
                "지역명",
                "종합점수",
                "행정적_강도",
                "전략적_강도",
                "청년예산_비율",
                "재정자립도",
            ]
        ].head(10)
        print(metro_display.to_string(index=False))

        # 기초자치단체 상위 순위
        basic = self.df[self.df["지역유형"] == "기초자치단체"].sort_values(
            "종합점수", ascending=False
        )
        print(f"\n🏘️ 기초자치단체 상위 10위:")
        basic_display = basic[
            [
                "기초순위",
                "지역명",
                "종합점수",
                "행정적_강도",
                "전략적_강도",
                "청년예산_비율",
                "재정자립도",
            ]
        ].head(10)
        print(basic_display.to_string(index=False))

        # 지역유형별 특성 분석
        print(f"\n📊 지역유형별 특성 비교:")
        comparison = (
            self.df.groupby("지역유형")
            .agg(
                {
                    "종합점수": ["mean", "median", "std"],
                    "행정적_강도": ["mean", "median"],
                    "전략적_강도": ["mean", "median"],
                    "청년예산_비율": ["mean", "median"],
                    "청년인구_비율": ["mean", "median"],
                    "재정자립도": ["mean", "median"],
                    "총정책수": ["mean", "median"],
                }
            )
            .round(4)
        )
        print(comparison)

    def budget_analysis(self):
        """예산 관련 분석"""
        print("\n" + "=" * 80)
        print("💰 예산 관련 분석")
        print("=" * 80)

        # 청년정책예산 상위 지역
        budget_top = self.df.nlargest(10, "청년정책예산_백만원")[
            ["지역명", "지역유형", "청년정책예산_백만원", "청년예산_비율", "종합점수"]
        ]
        print(f"\n💰 청년정책예산 상위 10개 지역:")
        print(budget_top.to_string(index=False))

        # 예산 대비 효율성 분석 (청년예산 비율 기준)
        efficiency = self.df[self.df["청년예산_비율"] > 0].copy()
        efficiency["예산효율성"] = efficiency["종합점수"] / efficiency["청년예산_비율"]
        efficiency_top = efficiency.nlargest(10, "예산효율성")[
            ["지역명", "지역유형", "청년예산_비율", "종합점수", "예산효율성"]
        ]
        print(f"\n⚡ 예산 효율성 상위 10개 지역:")
        print(efficiency_top.to_string(index=False))

        # 재정자립도와 성과의 관계
        print(f"\n📊 재정자립도별 평균 종합점수:")
        self.df["재정자립도_구간"] = pd.cut(
            self.df["재정자립도"],
            bins=[0, 0.2, 0.3, 0.4, 1.0],
            labels=["낮음(~20%)", "보통(20-30%)", "높음(30-40%)", "매우높음(40%~)"],
        )
        autonomy_analysis = (
            self.df.groupby("재정자립도_구간")
            .agg(
                {
                    "종합점수": ["mean", "count"],
                    "청년예산_비율": "mean",
                    "행정적_강도": "mean",
                }
            )
            .round(4)
        )
        print(autonomy_analysis)

    def visualization(self):
        """시각화"""
        print("\n" + "=" * 80)
        print("📊 시각화 생성")
        print("=" * 80)

        # 1. 종합점수 분포
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 종합점수 히스토그램
        axes[0, 0].hist(
            self.df["종합점수"], bins=30, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("종합점수 분포", fontsize=14)
        axes[0, 0].set_xlabel("종합점수")
        axes[0, 0].set_ylabel("빈도")

        # 지역유형별 종합점수 박스플롯
        self.df.boxplot(column="종합점수", by="지역유형", ax=axes[0, 1])
        axes[0, 1].set_title("지역유형별 종합점수 분포")
        axes[0, 1].set_xlabel("지역유형")

        # 청년예산비율 vs 종합점수 산점도
        colors = {"광역자치단체": "red", "기초자치단체": "blue"}
        for region_type in self.df["지역유형"].unique():
            data = self.df[self.df["지역유형"] == region_type]
            axes[1, 0].scatter(
                data["청년예산_비율"],
                data["종합점수"],
                c=colors[region_type],
                label=region_type,
                alpha=0.6,
            )
        axes[1, 0].set_xlabel("청년예산 비율")
        axes[1, 0].set_ylabel("종합점수")
        axes[1, 0].set_title("청년예산 비율 vs 종합점수")
        axes[1, 0].legend()

        # 재정자립도 vs 종합점수 산점도
        for region_type in self.df["지역유형"].unique():
            data = self.df[self.df["지역유형"] == region_type]
            axes[1, 1].scatter(
                data["재정자립도"],
                data["종합점수"],
                c=colors[region_type],
                label=region_type,
                alpha=0.6,
            )
        axes[1, 1].set_xlabel("재정자립도")
        axes[1, 1].set_ylabel("종합점수")
        axes[1, 1].set_title("재정자립도 vs 종합점수")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig("comprehensive_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

    def policy_effectiveness_analysis(self):
        """정책 효과성 분석"""
        print("\n" + "=" * 80)
        print("🎯 정책 효과성 분석")
        print("=" * 80)

        # 정책수와 성과의 관계
        print(
            f"\n📊 정책수와 종합점수의 상관관계: {self.df['총정책수'].corr(self.df['종합점수']):.3f}"
        )
        print(
            f"📊 정책분야수와 종합점수의 상관관계: {self.df['정책분야수'].corr(self.df['종합점수']):.3f}"
        )

        # 정책 다양성(분야수)별 성과
        self.df["정책다양성"] = pd.cut(
            self.df["정책분야수"],
            bins=[0, 2, 3, 4, 5, 10],
            labels=["매우낮음(~2)", "낮음(3)", "보통(4)", "높음(5)", "매우높음(5+)"],
        )
        diversity_analysis = (
            self.df.groupby("정책다양성")
            .agg(
                {
                    "종합점수": ["mean", "count"],
                    "전략적_강도": "mean",
                    "총정책수": "mean",
                }
            )
            .round(4)
        )
        print(f"\n📊 정책 다양성별 평균 성과:")
        print(diversity_analysis)

        # 행정적 강도와 전략적 강도의 균형 분석
        print(f"\n⚖️ 행정적 강도와 전략적 강도의 균형 분석:")
        balance = abs(self.df["행정적_강도_정규화"] - self.df["전략적_강도_정규화"])
        self.df["강도균형"] = balance
        balanced_regions = self.df[self.df["강도균형"] < 0.1].sort_values(
            "종합점수", ascending=False
        )
        print(f"균형잡힌 지역(강도 차이 < 0.1) 수: {len(balanced_regions)}개")
        if len(balanced_regions) > 0:
            print("상위 균형잡힌 지역:")
            print(
                balanced_regions[
                    [
                        "지역명",
                        "지역유형",
                        "종합점수",
                        "행정적_강도_정규화",
                        "전략적_강도_정규화",
                        "강도균형",
                    ]
                ]
                .head()
                .to_string(index=False)
            )

    def generate_report(self):
        """종합 리포트 생성"""
        print("\n" + "=" * 80)
        print("📋 종합 분석 리포트")
        print("=" * 80)

        report = f"""
📊 청년정책 종합평가 결과 분석 리포트
============================================

1️⃣ 기본 현황
- 총 분석 지역: {len(self.df)}개
- 광역자치단체: {len(self.df[self.df['지역유형'] == '광역자치단체'])}개
- 기초자치단체: {len(self.df[self.df['지역유형'] == '기초자치단체'])}개

2️⃣ 종합점수 분석
- 전체 평균: {self.df['종합점수'].mean():.3f}
- 표준편차: {self.df['종합점수'].std():.3f}
- 최고점: {self.df['종합점수'].max():.3f} ({self.df.loc[self.df['종합점수'].idxmax(), '지역명']})
- 최저점: {self.df['종합점수'].min():.3f} ({self.df.loc[self.df['종합점수'].idxmin(), '지역명']})

3️⃣ 지역유형별 성과
광역자치단체 평균: {self.df[self.df['지역유형'] == '광역자치단체']['종합점수'].mean():.3f}
기초자치단체 평균: {self.df[self.df['지역유형'] == '기초자치단체']['종합점수'].mean():.3f}

4️⃣ 주요 발견사항
- 종합점수와 가장 높은 상관관계: {self.df[self.numerical_cols].corr()['종합점수'].abs().nlargest(2).index[1]} ({self.df[self.numerical_cols].corr()['종합점수'].abs().nlargest(2).iloc[1]:.3f})
- 청년예산비율 평균: {self.df['청년예산_비율'].mean():.4f} ({self.df['청년예산_비율'].mean()*100:.2f}%)
- 재정자립도 평균: {self.df['재정자립도'].mean():.3f} ({self.df['재정자립도'].mean()*100:.1f}%)

5️⃣ 권장사항
- 상위 성과 지역의 우수사례 벤치마킹 필요
- 예산 효율성이 높은 지역의 정책 모델 분석 권장
- 지역 특성에 맞는 차별화된 청년정책 개발 필요
        """
        print(report)

        # 리포트를 파일로 저장
        with open("youth_policy_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n💾 분석 리포트가 'youth_policy_analysis_report.txt'로 저장되었습니다.")


def main():
    """메인 실행 함수"""
    analyzer = YouthPolicyResultAnalysis()

    # 데이터 로드
    analyzer.load_data()

    # 분석 실행
    analyzer.basic_statistics()
    analyzer.correlation_analysis()
    analyzer.regional_analysis()
    analyzer.budget_analysis()
    analyzer.policy_effectiveness_analysis()
    analyzer.visualization()
    analyzer.generate_report()

    print("\n✅ 모든 분석이 완료되었습니다!")
    print("📊 생성된 파일:")
    print("- correlation_heatmap.png: 상관관계 히트맵")
    print("- comprehensive_analysis.png: 종합 분석 차트")
    print("- youth_policy_analysis_report.txt: 분석 리포트")


if __name__ == "__main__":
    main()
