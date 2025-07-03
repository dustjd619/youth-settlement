"""
첨부된 이미지의 정책 점수 vs 청년 순유입률 플롯만 생성하는 간단한 스크립트
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def create_simple_settlement_plot():
    """정책 점수 vs 청년 순유입률 단일 플롯 생성"""
    base_path = Path(__file__).parent.parent.parent

    # CSV 결과 파일 읽기
    result_file = (
        base_path / "migration_plot/eval-5_result/settlement_analysis_results_eval5.csv"
    )

    if not result_file.exists():
        print("❌ 분석 결과 파일을 찾을 수 없습니다.")
        print(f"   경로: {result_file}")
        print("   먼저 policy_migration_analysis_eval5.py를 실행해주세요.")
        return

    # 데이터 로드
    data = pd.read_csv(result_file, encoding="utf-8-sig")
    print(f"✅ 데이터 로드 완료: {len(data)}개 지역")

    # 필요한 컬럼 확인
    required_cols = ["사용_점수", "순이동률_인구대비", "지역유형", "지역명_이동"]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        print(f"❌ 필요한 컬럼이 없습니다: {missing_cols}")
        return

        # 기초자치단체만 필터링
    municipal_data = data[data["지역유형"] == "기초자치단체"]
    valid_data = municipal_data[required_cols].dropna()
    print(f"✅ 기초자치단체 데이터: {len(valid_data)}개 지역")

    if len(valid_data) == 0:
        print("❌ 유효한 데이터가 없습니다.")
        return

    # 플롯 생성
    plt.figure(figsize=(12, 8))

    # 특정 지역들을 빨간색으로 하이라이트
    highlight_regions = [
        "충청남도 아산시",
        "충청북도 청주시",
        "경기도 화성시",
        "대구광역시 중구",
        "인천광역시 서구",
        "강원도 영월군",
        "강원도 철원군",
        "경상남도 거창군",
        "전라북도 진안군",
    ]

    # 색상 배열 생성
    colors = []
    for region in valid_data["지역명_이동"]:
        if region in highlight_regions:
            colors.append("red")
        else:
            colors.append("forestgreen")

    # 산점도 (특정 지역은 빨간색, 나머지는 녹색)
    plt.scatter(
        valid_data["사용_점수"],
        valid_data["순이동률_인구대비"],
        c=colors,
        alpha=0.7,
        s=80,
        edgecolors="white",
        linewidth=1,
    )

    # 빨간색 지역들에 라벨 추가 (위치 조정으로 겹침 방지)
    label_offsets = {
        "충청남도 아산시": (10, 15),
        "충청북도 청주시": (-80, -20),
        "경기도 화성시": (10, -20),
        "대구광역시 중구": (-80, 15),
        "인천광역시 서구": (10, 10),
        "강원도 영월군": (10, -25),
        "강원도 철원군": (10, 15),
        "경상남도 거창군": (-90, 10),
        "전라북도 진안군": (10, 10),
    }

    for idx, row in valid_data.iterrows():
        if row["지역명_이동"] in highlight_regions:
            # 각 지역별로 다른 위치에 라벨 배치
            offset = label_offsets.get(row["지역명_이동"], (8, 8))

            plt.annotate(
                row["지역명_이동"],
                (row["사용_점수"], row["순이동률_인구대비"]),
                xytext=offset,  # 각 지역별로 다른 offset 적용
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="darkred",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="red",
                    alpha=0.8,
                ),
                arrowprops=dict(arrowstyle="->", color="red", alpha=0.7, lw=1),
            )

    # 축 설정
    plt.xlabel("정책 점수", fontsize=14)
    plt.ylabel("순유입률 (청년인구 대비 %)", fontsize=14)
    plt.title(
        f"기초자치단체 정책 점수 vs 청년 순유입률\n(2023.08-2024.07)",
        fontsize=16,
        pad=20,
    )

    # 격자 및 참조선
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.5)  # y=0 선
    plt.axvline(
        x=valid_data["사용_점수"].mean(),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"평균 정책점수 ({valid_data['사용_점수'].mean():.2f})",
    )

    plt.tight_layout()

    # 저장
    save_path = base_path / "migration_plot/eval-5_result/simple_settlement_plot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 플롯 저장 완료: {save_path}")

    # 플롯 표시
    plt.show()

    # 하이라이트된 지역 확인
    highlighted_found = valid_data[valid_data["지역명_이동"].isin(highlight_regions)]

    # 간단한 통계 출력
    print(f"\n📊 플롯 통계:")
    print(f"- 기초자치단체: {len(valid_data)}개")
    print(f"- 하이라이트된 지역: {len(highlighted_found)}개 (빨간색)")
    print(
        f"- 정책점수 범위: {valid_data['사용_점수'].min():.2f} ~ {valid_data['사용_점수'].max():.2f}"
    )
    print(
        f"- 순유입률 범위: {valid_data['순이동률_인구대비'].min():.3f}% ~ {valid_data['순이동률_인구대비'].max():.3f}%"
    )

    if len(highlighted_found) > 0:
        print(f"\n🔍 하이라이트된 지역 목록:")
        for _, row in highlighted_found.iterrows():
            print(
                f"  - {row['지역명_이동']}: 정책점수 {row['사용_점수']:.3f}, 순유입률 {row['순이동률_인구대비']:.3f}%"
            )


if __name__ == "__main__":
    create_simple_settlement_plot()
