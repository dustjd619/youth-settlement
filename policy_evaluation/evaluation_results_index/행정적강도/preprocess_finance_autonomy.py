# 재정자립도 전처리 코드

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
재정자립도 데이터 전처리 스크립트
행정구역별(1), 행정구역별(2)를 통합하여 지자체명으로 만들기
"""

import numpy as np
import pandas as pd


def preprocess_finance_autonomy():
    """
    재정자립도 데이터를 전처리하여 청년비율 데이터와 같은 형식으로 변환
    """
    # 원본 데이터 읽기
    print("재정자립도 데이터 읽는 중...")
    df = pd.read_csv("data/policy/재정자립도/finance_autonomy.csv", encoding="utf-8")

    print(f"원본 데이터 행 수: {len(df)}")
    print(f"컬럼: {list(df.columns)}")
    print("\n원본 데이터 미리보기:")
    print(df.head(10))

    # 전처리된 데이터를 저장할 리스트
    processed_data = []
    current_province = None

    for idx, row in df.iterrows():
        province = row["행정구역별(1)"]
        district = row["행정구역별(2)"]
        autonomy_rate = row["재정자립도(세입과목개편후)"]

        # 전국 데이터는 건너뛰기
        if province == "전국":
            continue

        # 행정구역별(1)이 비어있지 않으면 상위 행정구역 업데이트
        if pd.notna(province) and province.strip() != "":
            current_province = province.strip()

            # "소계" 행 처리 - 상위 행정구역 전체 데이터
            if district == "소계":
                # 세종특별자치시는 특별한 경우
                if current_province == "세종특별자치시":
                    combined_name = current_province
                else:
                    combined_name = current_province

                processed_data.append(
                    {"지자체명": combined_name, "재정자립도": autonomy_rate}
                )
                continue

        # 행정구역별(1)이 비어있고 행정구역별(2)가 "소계"가 아닌 경우 (개별 구/군/시)
        elif pd.notna(district) and district != "소계":
            if current_province is None:
                print(f"경고: {idx}행에서 상위 행정구역을 찾을 수 없습니다.")
                continue

            # 통합된 지자체명 생성
            combined_name = f"{current_province} {district}"

            processed_data.append(
                {"지자체명": combined_name, "재정자립도": autonomy_rate}
            )

    # 새로운 DataFrame 생성
    processed_df = pd.DataFrame(processed_data)

    print(f"\n전처리된 데이터 행 수: {len(processed_df)}")
    print("\n전처리된 데이터 미리보기:")
    print(processed_df.head(15))

    # 특별한 케이스들 확인
    print("\n특별 케이스 확인:")

    # 소계 데이터 (시/도 전체)
    province_totals = processed_df[~processed_df["지자체명"].str.contains(" ")].copy()
    print(f"시/도 전체 데이터 ({len(province_totals)}개):")
    print(province_totals.head(10))

    # 개별 구/군/시 데이터
    district_data = processed_df[processed_df["지자체명"].str.contains(" ")].copy()
    print(f"\n개별 구/군/시 데이터 ({len(district_data)}개):")
    print(district_data.head(10))

    # 청년비율 데이터와 매칭 확인을 위해 비교
    print("\n청년비율 데이터와 매칭 확인...")
    youth_df = pd.read_csv("data/policy/청년비율_시군구_기준.csv")

    # 지자체명 매칭 확인
    finance_names = set(processed_df["지자체명"])
    youth_names = set(youth_df["지자체명"])

    # 매칭되는 지자체
    matched = finance_names.intersection(youth_names)
    print(f"매칭되는 지자체 수: {len(matched)}")

    # 재정자립도에만 있는 지자체
    finance_only = finance_names - youth_names
    if finance_only:
        print(f"재정자립도에만 있는 지자체 ({len(finance_only)}개):")
        for name in sorted(finance_only)[:15]:  # 처음 15개 출력
            print(f"  - {name}")

    # 청년비율에만 있는 지자체
    youth_only = youth_names - finance_names
    if youth_only:
        print(f"청년비율에만 있는 지자체 ({len(youth_only)}개):")
        for name in sorted(youth_only)[:15]:  # 처음 15개 출력
            print(f"  - {name}")

    # 전처리된 데이터 저장
    output_file = "data/policy/재정자립도/finance_autonomy_processed.csv"
    processed_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n전처리된 데이터 저장 완료: {output_file}")

    return processed_df


if __name__ == "__main__":
    processed_df = preprocess_finance_autonomy()
