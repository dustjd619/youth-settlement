from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# 한글 폰트 설정
plt.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 데이터 경로
base_path = Path(__file__).parent.parent  # migration_plot
csv_path = base_path / "settlement_induction_result.csv"

# 데이터 로드
if not csv_path.exists():
    print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 필요한 컬럼 확인
if not (
    "종합점수" in df.columns and "순이동" in df.columns and "지역유형" in df.columns
):
    print("❌ 필요한 컬럼(종합점수, 순이동, 지역유형)이 없습니다.")
    exit(1)

# 데이터 분리
df_valid = df[["종합점수", "순이동", "지역유형", "지역명_정책"]].dropna()
metro = df_valid[df_valid["지역유형"] == "광역자치단체"]
muni = df_valid[df_valid["지역유형"] == "기초자치단체"]

# 3x1 플롯
fig, axes = plt.subplots(1, 3, figsize=(30, 8))

# 1. 광역
if len(metro) > 0:
    x = metro["종합점수"]
    y = metro["순이동"]
    axes[0].scatter(
        x, y, c="steelblue", s=120, alpha=0.7, edgecolors="white", linewidth=1
    )
    # 순이동량 상위/하위 5개만 라벨링
    # sorted_metro = metro.sort_values("순이동")
    # top_bottom = pd.concat([sorted_metro.head(5), sorted_metro.tail(5)])
    for idx, row in metro.iterrows():
        axes[0].annotate(
            row["지역명_정책"],
            (row["종합점수"], row["순이동"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        )
    if len(metro) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[0].plot(
            x,
            p(x),
            "r--",
            linewidth=2,
            alpha=0.8,
            label=f"회귀선: y={z[0]:.2f}x+{z[1]:.2f}",
        )
        corr, pval = stats.pearsonr(x, y)
        axes[0].text(
            0.05,
            0.95,
            f"r={corr:.3f}\np={pval:.4f}\nn={len(x)}",
            transform=axes[0].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
            verticalalignment="top",
        )
        axes[0].legend(loc="upper left")
    axes[0].set_xlabel("정책 종합점수", fontsize=12)
    axes[0].set_ylabel("순유입량", fontsize=12)
    axes[0].set_title(f"광역자치단체 (n={len(x)})", fontsize=14, pad=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[0].axvline(x=x.mean(), color="gray", linestyle="--", alpha=0.3)

# 2. 기초
if len(muni) > 0:
    x = muni["종합점수"]
    y = muni["순이동"]
    axes[1].scatter(
        x, y, c="forestgreen", s=60, alpha=0.6, edgecolors="white", linewidth=0.5
    )
    # 순이동량 상위/하위 5개만 라벨링
    sorted_muni = muni.sort_values("순이동")
    top_bottom = pd.concat([sorted_muni.head(5), sorted_muni.tail(5)])
    for idx, row in top_bottom.iterrows():
        axes[1].annotate(
            row["지역명_정책"],
            (row["종합점수"], row["순이동"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
        )
    if len(muni) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        axes[1].plot(
            x,
            p(x),
            "r--",
            linewidth=2,
            alpha=0.8,
            label=f"회귀선: y={z[0]:.2f}x+{z[1]:.2f}",
        )
        corr, pval = stats.pearsonr(x, y)
        axes[1].text(
            0.05,
            0.95,
            f"r={corr:.3f}\np={pval:.4f}\nn={len(x)}",
            transform=axes[1].transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
            verticalalignment="top",
        )
        axes[1].legend(loc="upper left")
    axes[1].set_xlabel("정책 종합점수", fontsize=12)
    axes[1].set_ylabel("순유입량", fontsize=12)
    axes[1].set_title(f"기초자치단체 (n={len(x)})", fontsize=14, pad=20)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes[1].axvline(x=x.mean(), color="gray", linestyle="--", alpha=0.3)

# 3. 전체
color_map = {"광역자치단체": "steelblue", "기초자치단체": "forestgreen"}
colors = df_valid["지역유형"].map(color_map).fillna("gray")
x = df_valid["종합점수"]
y = df_valid["순이동"]
axes[2].scatter(x, y, c=colors, s=60, alpha=0.6, edgecolors="white", linewidth=0.5)
# 순이동량 상위/하위 5개만 라벨링
sorted_all = df_valid.sort_values("순이동")
top_bottom = pd.concat([sorted_all.head(5), sorted_all.tail(5)])
for idx, row in top_bottom.iterrows():
    axes[2].annotate(
        row["지역명_정책"],
        (row["종합점수"], row["순이동"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )
if len(df_valid) > 2:
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    axes[2].plot(
        x,
        p(x),
        "r--",
        linewidth=2,
        alpha=0.8,
        label=f"회귀선: y={z[0]:.2f}x+{z[1]:.2f}",
    )
    corr, pval = stats.pearsonr(x, y)
    axes[2].text(
        0.05,
        0.95,
        f"r={corr:.3f}\np={pval:.4f}\nn={len(x)}",
        transform=axes[2].transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        verticalalignment="top",
    )
    axes[2].legend(loc="upper left")
axes[2].set_xlabel("정책 종합점수", fontsize=12)
axes[2].set_ylabel("순유입량", fontsize=12)
axes[2].set_title(f"전체(광역+기초) (n={len(x)})", fontsize=14, pad=20)
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
axes[2].axvline(x=x.mean(), color="gray", linestyle="--", alpha=0.3)

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

plt.suptitle(
    "정책 종합점수 vs 청년 순유입량 (광역 vs 기초 vs 전체)\n(정책 시차 반영: 2023.08-2024.07, 순유입량 = 전입-전출)",
    fontsize=16,
    y=0.98,
)
plt.tight_layout()

save_path = base_path / "순유입량_분석/settlement_induction_net_plot.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ 정책 종합점수 vs 순유입량 플롯 생성 완료 (광역/기초/전체)")
print(f"📁 저장 위치: {save_path}")
