import pandas as pd


def preprocess_budget_data():
    """
    세출예산.csv 파일을 전처리합니다.
    1. '자치단체명', '세출총계' 컬럼만 남김 (지역명 제거)
    2. 따옴표 제거
    3. 자치단체명 형식 변경:
       - 서울: '서울XXX' → '서울특별시 XXX'
       - 부산: '부산XXX' → '부산광역시 XXX'
       - 대구: '대구XXX' → '대구광역시 XXX'
       - 인천: '인천XXX' → '인천광역시 XXX'
       - 광주: '광주XXX' → '광주광역시 XXX'
       - 울산: '울산XXX' → '울산광역시 XXX'
       - 세종: '세종XXX' → '세종특별자치시 XXX'
       - 경기: '경기XXX' → '경기도 XXX'
       - 강원: '강원XXX' → '강원도 XXX'
       - 충북: '충북XXX' → '충청북도 XXX'
       - 충남: '충남XXX' → '충청남도 XXX'
       - 전북: '전북XXX' → '전라북도 XXX'
       - 전남: '전남XXX' → '전라남도 XXX'
       - 경북: '경북XXX' → '경상북도 XXX'
       - 경남: '경남XXX' → '경상남도 XXX'
       - 제주: '제주XXX' → '제주특별자치도 XXX'
       - 기타: 'XXYYY' → 'XX YYY'
    4. 세출총계를 백만원 단위로 변환
    5. 본청 데이터와 일반 데이터 분리하여 각각 저장
    """

    # CSV 파일 읽기 (따옴표 처리를 위해 quotechar 지정)
    df = pd.read_csv("세출예산.csv", quotechar='"')

    # 필요한 컬럼만 선택
    columns_to_keep = ["자치단체명", "세출총계"]
    df_filtered = df[columns_to_keep].copy()

    # 문자열 컬럼의 경우 따옴표가 있다면 제거 (pandas가 자동으로 처리하지만 확실히 하기 위해)
    for col in df_filtered.columns:
        if df_filtered[col].dtype == "object":
            df_filtered[col] = df_filtered[col].astype(str).str.strip('"')

    # 자치단체명 형식 변경 (앞 두글자에 따라 다른 형식 적용)
    def format_municipality_name(name):
        if len(name) <= 2:
            return name

        prefix = name[:2]
        suffix = name[2:]

        if prefix == "서울":
            return f"서울특별시 {suffix}"
        elif prefix == "부산":
            return f"부산광역시 {suffix}"
        elif prefix == "대구":
            return f"대구광역시 {suffix}"
        elif prefix == "인천":
            return f"인천광역시 {suffix}"
        elif prefix == "광주":
            return f"광주광역시 {suffix}"
        elif prefix == "울산":
            return f"울산광역시 {suffix}"
        elif prefix == "세종":
            return f"세종특별자치시 {suffix}"
        elif prefix == "경기":
            return f"경기도 {suffix}"
        elif prefix == "강원":
            return f"강원도 {suffix}"
        elif prefix == "충북":
            return f"충청북도 {suffix}"
        elif prefix == "충남":
            return f"충청남도 {suffix}"
        elif prefix == "전북":
            return f"전라북도 {suffix}"
        elif prefix == "전남":
            return f"전라남도 {suffix}"
        elif prefix == "경북":
            return f"경상북도 {suffix}"
        elif prefix == "경남":
            return f"경상남도 {suffix}"
        elif prefix == "제주":
            return f"제주특별자치도 {suffix}"
        else:
            return f"{prefix} {suffix}"

    df_filtered["자치단체명"] = df_filtered["자치단체명"].apply(
        format_municipality_name
    )

    # 세출총계를 백만원 단위로 변환 (원 단위를 백만원 단위로)
    df_filtered["세출총계"] = pd.to_numeric(df_filtered["세출총계"]) / 1000000
    df_filtered["세출총계"] = df_filtered["세출총계"].round(1)  # 소수점 첫째자리까지

    # 본청 데이터와 일반 데이터 분리
    df_central = df_filtered[df_filtered["자치단체명"].str.contains("본청")].copy()
    df_general = df_filtered[~df_filtered["자치단체명"].str.contains("본청")].copy()

    # 처리된 데이터를 새 파일로 저장
    df_filtered.to_csv("세출예산_processed.csv", index=False, quoting=0)
    df_central.to_csv("세출예산_본청.csv", index=False, quoting=0)
    df_general.to_csv("세출예산_일반.csv", index=False, quoting=0)

    print("전처리 완료!")
    print(f"원본 데이터 크기: {df.shape}")
    print(f"전체 처리된 데이터 크기: {df_filtered.shape}")
    print(f"본청 데이터 크기: {df_central.shape}")
    print(f"일반 데이터 크기: {df_general.shape}")
    print("\n본청 데이터 미리보기:")
    print(df_central.head())
    print("\n일반 데이터 미리보기:")
    print(df_general.head())

    return df_filtered, df_central, df_general


if __name__ == "__main__":
    processed_data = preprocess_budget_data()
