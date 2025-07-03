import os
import warnings

import matplotlib

matplotlib.use("Agg")  # GUI 없는 백엔드 설정
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")

# 한글 폰트 설정
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"  # macOS 기본 폰트
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False  # 음수 표시 문제 해결


def load_data():
    """데이터 로드"""
    df = pd.read_csv("migration_plot/settlement_induction_result.csv")
    return df


def prepare_data(df):
    """개선된 데이터 전처리"""
    # 필요한 컬럼만 선택
    columns = [
        "지역명_정책",
        "지역유형",
        "행정적_강도",
        "전략적_강도",
        "행정적_강도_정규화",
        "전략적_강도_정규화",
        "순이동률_인구대비",
        "재정자립도",
        "청년1인당_정책예산_원",
        "전체1인당_총예산_원",
        "일자리_점수",
        "주거_점수",
        "교육_점수",
        "복지·문화_점수",
        "참여·권리_점수",
    ]

    data = df[columns].copy()

    # 결측치 확인 및 처리
    print("결측치 확인:")
    print(data.isnull().sum())

    # 결측치가 있는 행 제거
    data = data.dropna()

    # 이상치 탐지 및 제거 (IQR 방법)
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # 순이동률_인구대비 이상치 제거
    data_clean = remove_outliers(data, "순이동률_인구대비")
    print(f"\n이상치 제거 전: {len(data)}개")
    print(f"이상치 제거 후: {len(data_clean)}개")

    # 추가 변수 생성
    data_clean["정책_강도_합계"] = data_clean["행정적_강도"] + data_clean["전략적_강도"]
    data_clean["정책_강도_비율"] = data_clean["전략적_강도"] / (
        data_clean["행정적_강도"] + 1e-8
    )
    data_clean["종합_점수"] = (
        data_clean["일자리_점수"]
        + data_clean["주거_점수"]
        + data_clean["교육_점수"]
        + data_clean["복지·문화_점수"]
        + data_clean["참여·권리_점수"]
    ) / 5

    print(f"\n전체 데이터 수: {len(data_clean)}")
    print(
        f"광역자치단체 수: {len(data_clean[data_clean['지역유형'] == '광역자치단체'])}"
    )
    print(
        f"기초자치단체 수: {len(data_clean[data_clean['지역유형'] == '기초자치단체'])}"
    )

    # 기술통계 출력
    print("\n주요 변수 기술통계:")
    print(
        data_clean[
            [
                "순이동률_인구대비",
                "행정적_강도",
                "전략적_강도",
                "재정자립도",
                "종합_점수",
            ]
        ].describe()
    )

    return data_clean


def run_multiple_regression(data, version="original"):
    """개선된 다중 회귀 분석 실행"""
    print(f"\n=== {version.upper()} 버전 다중 회귀 분석 ===")

    if version == "original":
        X = data[["행정적_강도", "전략적_강도"]]
        feature_names = ["행정적_강도", "전략적_강도"]
    elif version == "normalized":
        X = data[["행정적_강도_정규화", "전략적_강도_정규화"]]
        feature_names = ["행정적_강도_정규화", "전략적_강도_정규화"]
    elif version == "extended":
        X = data[
            ["행정적_강도", "전략적_강도", "재정자립도", "종합_점수", "정책_강도_합계"]
        ]
        feature_names = [
            "행정적_강도",
            "전략적_강도",
            "재정자립도",
            "종합_점수",
            "정책_강도_합계",
        ]

    y = data["순이동률_인구대비"]

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )

    # 스케일링 (RobustScaler 사용)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 다양한 모델 시도
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }

    best_model = None
    best_score = -np.inf
    results = {}

    for name, model in models.items():
        print(f"\n--- {name} ---")

        # 모델 학습
        model.fit(X_train_scaled, y_train)

        # 예측
        y_pred = model.predict(X_test_scaled)

        # 교차 검증
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")

        # 모델 성능 평가
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"R² Score: {r2:.4f}")
        print(
            f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")

        # 계수 출력 (선형 모델인 경우)
        if hasattr(model, "coef_"):
            print(f"\n회귀 계수:")
            for i, feature in enumerate(feature_names):
                print(f"{feature}: {model.coef_[i]:.6f}")
            print(f"절편: {model.intercept_:.6f}")

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "r2": r2,
            "cv_r2": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
        }

        if r2 > best_score:
            best_score = r2
            best_model = name

    print(f"\n최고 성능 모델: {best_model} (R² = {best_score:.4f})")

    # Statsmodels를 사용한 상세 분석 (최고 성능 모델)
    best_model_obj = results[best_model]["model"]
    if hasattr(best_model_obj, "coef_"):
        X_with_const = sm.add_constant(X_train_scaled)
        model_sm = sm.OLS(y_train, X_with_const).fit()

        print(f"\nStatsmodels 결과 ({best_model}):")
        print(model_sm.summary())

        results[best_model]["model_sm"] = model_sm

    return results, scaler, X_train, X_test, y_train, y_test, feature_names


def create_enhanced_visualizations(data, results_dict, feature_names, y_test):
    """개선된 시각화 생성"""

    # 저장 디렉토리 생성
    plot_dir = "migration_plot/multiple_regression/plot"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"그래프 저장 디렉토리: {plot_dir}")

    # 1. 상관관계 히트맵 (더 많은 변수 포함)
    plt.figure(figsize=(12, 10))
    correlation_vars = [
        "순이동률_인구대비",
        "행정적_강도",
        "전략적_강도",
        "재정자립도",
        "종합_점수",
        "정책_강도_합계",
        "정책_강도_비율",
    ]
    corr_matrix = data[correlation_vars].corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("변수 간 상관관계 히트맵", fontsize=16, pad=20)
    plt.tight_layout()
    try:
        save_path = os.path.join(plot_dir, "enhanced_correlation_heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 상관관계 히트맵 저장 완료: {save_path}")
    except Exception as e:
        print(f"✗ 상관관계 히트맵 저장 실패: {e}")
    plt.show()

    # 2. 모델 성능 비교
    model_names = list(results_dict.keys())
    r2_scores = [results_dict[name]["r2"] for name in model_names]
    cv_scores = [results_dict[name]["cv_r2"] for name in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # R² 점수 비교
    bars1 = ax1.bar(model_names, r2_scores, color="skyblue", alpha=0.7)
    ax1.set_title("모델별 R² 점수 비교", fontsize=14)
    ax1.set_ylabel("R² Score")
    ax1.tick_params(axis="x", rotation=45)

    # 값 표시
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # 교차검증 점수 비교
    bars2 = ax2.bar(model_names, cv_scores, color="lightcoral", alpha=0.7)
    ax2.set_title("모델별 교차검증 R² 점수 비교", fontsize=14)
    ax2.set_ylabel("CV R² Score")
    ax2.tick_params(axis="x", rotation=45)

    # 값 표시
    for bar, score in zip(bars2, cv_scores):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    try:
        save_path = os.path.join(plot_dir, "model_performance_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 모델 성능 비교 그래프 저장 완료: {save_path}")
    except Exception as e:
        print(f"✗ 모델 성능 비교 그래프 저장 실패: {e}")
    plt.show()

    # 3. 최고 성능 모델의 예측 vs 실제 값
    best_model_name = max(results_dict.keys(), key=lambda x: results_dict[x]["r2"])
    best_results = results_dict[best_model_name]

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, best_results["y_pred"], alpha=0.6, s=50)

    # 회귀선 추가
    min_val = min(y_test.min(), best_results["y_pred"].min())
    max_val = max(y_test.max(), best_results["y_pred"].max())
    plt.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    plt.xlabel("실제 순이동률 (%)", fontsize=12)
    plt.ylabel("예측 순이동률 (%)", fontsize=12)
    plt.title(
        f'{best_model_name} - 예측 vs 실제 값\nR² = {best_results["r2"]:.4f}',
        fontsize=14,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        save_path = os.path.join(plot_dir, "best_model_prediction.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 최고 모델 예측 그래프 저장 완료: {save_path}")
    except Exception as e:
        print(f"✗ 최고 모델 예측 그래프 저장 실패: {e}")
    plt.show()

    # 4. 잔차 분석
    residuals = y_test - best_results["y_pred"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 잔차 vs 예측값
    axes[0, 0].scatter(best_results["y_pred"], residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color="r", linestyle="--")
    axes[0, 0].set_xlabel("예측값")
    axes[0, 0].set_ylabel("잔차")
    axes[0, 0].set_title("잔차 vs 예측값")
    axes[0, 0].grid(True, alpha=0.3)

    # 잔차 히스토그램
    axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("잔차")
    axes[0, 1].set_ylabel("빈도")
    axes[0, 1].set_title("잔차 분포")
    axes[0, 1].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (정규성 검정)")
    axes[1, 0].grid(True, alpha=0.3)

    # 잔차 vs 실제값
    axes[1, 1].scatter(y_test, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color="r", linestyle="--")
    axes[1, 1].set_xlabel("실제값")
    axes[1, 1].set_ylabel("잔차")
    axes[1, 1].set_title("잔차 vs 실제값")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    try:
        save_path = os.path.join(plot_dir, "residual_analysis_enhanced.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 잔차 분석 그래프 저장 완료: {save_path}")
    except Exception as e:
        print(f"✗ 잔차 분석 그래프 저장 실패: {e}")
    plt.show()

    # 5. 지역별 분석
    plt.figure(figsize=(12, 8))

    # 지역 유형별로 색상 구분
    colors = {"광역자치단체": "red", "기초자치단체": "blue"}

    for region_type in ["광역자치단체", "기초자치단체"]:
        mask = data["지역유형"] == region_type
        plt.scatter(
            data[mask]["행정적_강도"],
            data[mask]["순이동률_인구대비"],
            c=colors[region_type],
            label=region_type,
            alpha=0.6,
            s=50,
        )

    plt.xlabel("행정적 강도", fontsize=12)
    plt.ylabel("순이동률 (%)", fontsize=12)
    plt.title("지역 유형별 행정적 강도와 순이동률 관계", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    try:
        save_path = os.path.join(plot_dir, "regional_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 지역별 분석 그래프 저장 완료: {save_path}")
    except Exception as e:
        print(f"✗ 지역별 분석 그래프 저장 실패: {e}")
    plt.show()


def save_enhanced_results(results_dict, data, feature_names, X_test, y_test):
    """개선된 결과 저장"""

    # 모델 성능 비교 결과
    performance_df = pd.DataFrame(
        {
            "Model": list(results_dict.keys()),
            "R²": [results_dict[name]["r2"] for name in results_dict.keys()],
            "CV_R²": [results_dict[name]["cv_r2"] for name in results_dict.keys()],
            "CV_Std": [results_dict[name]["cv_std"] for name in results_dict.keys()],
            "MSE": [results_dict[name]["mse"] for name in results_dict.keys()],
            "MAE": [results_dict[name]["mae"] for name in results_dict.keys()],
            "RMSE": [results_dict[name]["rmse"] for name in results_dict.keys()],
        }
    )

    performance_df = performance_df.sort_values("R²", ascending=False)
    performance_df.to_csv(
        "enhanced_model_performance.csv", index=False, encoding="utf-8-sig"
    )

    # 최고 성능 모델의 상세 결과
    best_model_name = performance_df.iloc[0]["Model"]
    best_results = results_dict[best_model_name]

    # 계수 정보 (선형 모델인 경우)
    if hasattr(best_results["model"], "coef_"):
        coefficients_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Coefficient": best_results["model"].coef_,
                "Abs_Coefficient": np.abs(best_results["model"].coef_),
            }
        )
        coefficients_df = coefficients_df.sort_values(
            "Abs_Coefficient", ascending=False
        )
        coefficients_df.to_csv(
            "feature_importance.csv", index=False, encoding="utf-8-sig"
        )

    # 예측 결과
    test_indices = X_test.index
    predictions_df = pd.DataFrame(
        {
            "지역명": data.loc[test_indices, "지역명_정책"].values,
            "지역유형": data.loc[test_indices, "지역유형"].values,
            "실제_순이동률": y_test.values,
            "예측_순이동률": best_results["y_pred"],
            "잔차": y_test.values - best_results["y_pred"],
            "절대_잔차": np.abs(y_test.values - best_results["y_pred"]),
        }
    )

    predictions_df = predictions_df.sort_values("절대_잔차", ascending=False)
    predictions_df.to_csv("detailed_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"\n결과가 다음 파일들로 저장되었습니다:")
    print(f"- enhanced_model_performance.csv: 모델 성능 비교")
    print(f"- feature_importance.csv: 변수 중요도 (최고 성능 모델)")
    print(f"- detailed_predictions.csv: 상세 예측 결과")
    print(f"- 최고 성능 모델: {best_model_name} (R² = {best_results['r2']:.4f})")


def main():
    """메인 함수"""
    print("개선된 다중 회귀 분석 시작...")

    # 데이터 로드
    df = load_data()

    # 데이터 전처리
    data = prepare_data(df)

    # 다양한 버전의 다중 회귀 분석
    versions = ["original", "normalized", "extended"]
    all_results = {}

    for version in versions:
        results, scaler, X_train, X_test, y_train, y_test, feature_names = (
            run_multiple_regression(data, version)
        )
        all_results[version] = results

    # 시각화 생성 (최고 성능 버전 사용)
    best_version = max(
        all_results.keys(),
        key=lambda v: max(all_results[v].values(), key=lambda x: x["r2"])["r2"],
    )
    best_results = all_results[best_version]

    # 최고 성능 모델의 y_test 가져오기
    best_model_name = max(best_results.keys(), key=lambda x: best_results[x]["r2"])
    _, _, _, X_test, _, y_test, _ = run_multiple_regression(data, best_version)

    print(f"\n최고 성능 버전: {best_version}")
    create_enhanced_visualizations(data, best_results, feature_names, y_test)

    # 결과 저장
    save_enhanced_results(best_results, data, feature_names, X_test, y_test)

    print("\n분석 완료!")


if __name__ == "__main__":
    main()
