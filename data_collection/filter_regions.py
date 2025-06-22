import os

import pandas as pd


def filter_regions_data():
    # 파일 경로
    input_file = "/Users/shinyeonseong/source/repos/youth-settlement/data/policy/crawling/타지역_정책.csv"
    output_file = "경기_인천_정책.csv"

    # 파일 존재 확인
    if not os.path.exists(input_file):
        print(f"파일을 찾을 수 없습니다: {input_file}")
        return

    try:
        # CSV 파일 읽기
        print("데이터 로딩 중...")
        df = pd.read_csv(input_file, encoding="utf-8")

        print(f"전체 데이터 수: {len(df)}개")
        print(f"컬럼: {list(df.columns)}")

        # 지역명 확인
        if "지역명" not in df.columns:
            print("'지역명' 컬럼을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return

        # 고유 지역명 확인 (NaN 값 제거)
        unique_regions = df["지역명"].dropna().unique()
        unique_regions_str = [
            str(region) for region in unique_regions if pd.notna(region)
        ]
        print(f"\n전체 지역명: {sorted(unique_regions_str)}")

        # '경기'와 '인천' 데이터만 필터링
        filtered_df = df[df["지역명"].isin(["경기", "인천"])]

        print(f"\n필터링 결과:")
        print(f"- 경기 데이터: {len(df[df['지역명'] == '경기'])}개")
        print(f"- 인천 데이터: {len(df[df['지역명'] == '인천'])}개")
        print(f"- 총 필터링된 데이터: {len(filtered_df)}개")

        # 새로운 파일로 저장
        filtered_df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"\n'{output_file}' 파일로 저장 완료!")

        # 결과 미리보기
        print("\n저장된 데이터 미리보기:")
        print(filtered_df[["지역명", "정책명", "정책유형"]].head(10))

        return filtered_df

    except Exception as e:
        print(f"오류 발생: {e}")
        return None


if __name__ == "__main__":
    result = filter_regions_data()
