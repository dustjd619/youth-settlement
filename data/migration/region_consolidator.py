"""
지역명 통합 및 표준화 모듈
=========================

이 모듈은 구로 나뉘어진 시 단위 지역들을 하나로 통합하는 기능을 제공합니다.
예: 경기도 용인시 기흥구, 경기도 용인시 수지구, 경기도 용인시 처인구 → 경기도 용인시

사용자 요청 통합 대상:
- 경기도 고양시, 성남시, 수원시, 안산시, 안양시, 용인시
- 경상남도 창원시
- 경상북도 포항시
- 전라북도 전주시
- 제주특별자치도
- 충청남도 천안시
- 충청북도 청주시
"""

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


class RegionConsolidator:
    """지역명 통합 및 표준화 클래스"""

    def __init__(self):
        # 시도명 표준화 규칙 (특별자치도 → 일반 명칭)
        self.province_standardization = {
            "강원특별자치도": "강원도",
            "전북특별자치도": "전라북도",
        }

        # 통합이 필요한 시 목록 정의 (사용자 요청 기준)
        self.consolidation_rules = {
            # 경기도
            "경기도 고양시": [
                "경기도 고양시 덕양구",
                "경기도 고양시 일산동구",
                "경기도 고양시 일산서구",
            ],
            "경기도 성남시": [
                "경기도 성남시 수정구",
                "경기도 성남시 중원구",
                "경기도 성남시 분당구",
            ],
            "경기도 수원시": [
                "경기도 수원시 영통구",
                "경기도 수원시 장안구",
                "경기도 수원시 팔달구",
                "경기도 수원시 권선구",
            ],
            "경기도 안산시": ["경기도 안산시 단원구", "경기도 안산시 상록구"],
            "경기도 안양시": ["경기도 안양시 만안구", "경기도 안양시 동안구"],
            "경기도 용인시": [
                "경기도 용인시 기흥구",
                "경기도 용인시 수지구",
                "경기도 용인시 처인구",
            ],
            # 경상남도
            "경상남도 창원시": [
                "경상남도 창원시 의창구",
                "경상남도 창원시 성산구",
                "경상남도 창원시 마산합포구",
                "경상남도 창원시 마산회원구",
                "경상남도 창원시 진해구",
            ],
            # 경상북도
            "경상북도 포항시": ["경상북도 포항시 남구", "경상북도 포항시 북구"],
            # 전라북도
            "전라북도 전주시": ["전라북도 전주시 완산구", "전라북도 전주시 덕진구"],
            # 제주특별자치도
            "제주특별자치도": ["제주특별자치도 제주시", "제주특별자치도 서귀포시"],
            # 충청남도
            "충청남도 천안시": ["충청남도 천안시 동남구", "충청남도 천안시 서북구"],
            # 충청북도
            "충청북도 청주시": [
                "충청북도 청주시 상당구",
                "충청북도 청주시 서원구",
                "충청북도 청주시 흥덕구",
                "충청북도 청주시 청원구",
            ],
        }

        # 역방향 매핑 생성 (구 이름 → 통합된 시 이름)
        self.district_to_city = {}
        for city, districts in self.consolidation_rules.items():
            for district in districts:
                self.district_to_city[district] = city

        print(
            f"🏛️ 통합 규칙 로드 완료: {len(self.consolidation_rules)}개 시, {sum(len(v) for v in self.consolidation_rules.values())}개 구"
        )
        print(f"📍 시도명 표준화 규칙: {len(self.province_standardization)}개")

    def standardize_region_name(self, region_name):
        """지역명을 표준화하여 통합 대상인지 확인하고 변환"""
        if pd.isna(region_name) or region_name == "":
            return region_name

        region_name = str(region_name).strip()

        # 1단계: 시도명 표준화 (특별자치도 → 일반 명칭)
        for old_province, new_province in self.province_standardization.items():
            if region_name.startswith(old_province):
                region_name = region_name.replace(old_province, new_province, 1)
                break

        # 2단계: 구 통합 (구별 지역 → 시 단위 통합)
        if region_name in self.district_to_city:
            return self.district_to_city[region_name]

        return region_name

    def consolidate_dataframe(self, df, region_column="전출행정기관명_현재"):
        """데이터프레임의 지역명을 통합 처리"""
        if region_column not in df.columns:
            print(f"⚠️ 지역명 컬럼 '{region_column}'을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {list(df.columns)}")
            return df

        print(f"📍 지역명 통합 처리 중... (컬럼: {region_column})")

        # 원본 지역 수
        original_regions = df[region_column].nunique()

        # 지역명 표준화 적용
        df["통합지역명"] = df[region_column].apply(self.standardize_region_name)

        # 데이터 집계 (숫자형 컬럼들을 합계)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = [
            col
            for col in df.columns
            if col not in numeric_columns and col != region_column
        ]

        # 그룹화 및 집계
        if numeric_columns:
            # 숫자형 컬럼은 합계
            agg_dict = {col: "sum" for col in numeric_columns}

            # 비숫자형 컬럼 중 첫 번째 값 유지
            for col in non_numeric_columns:
                if col != "통합지역명":
                    agg_dict[col] = "first"

            consolidated_df = df.groupby("통합지역명").agg(agg_dict).reset_index()
            consolidated_df.rename(columns={"통합지역명": region_column}, inplace=True)
        else:
            consolidated_df = df.copy()
            consolidated_df[region_column] = consolidated_df["통합지역명"]
            consolidated_df.drop("통합지역명", axis=1, inplace=True)

        # 통합 결과
        consolidated_regions = consolidated_df[region_column].nunique()

        print(
            f"  ✅ 지역 수: {original_regions} → {consolidated_regions} ({original_regions - consolidated_regions}개 통합)"
        )

        return consolidated_df

    def consolidate_csv_columns(self, csv_file_path):
        """CSV 파일의 컬럼을 통합 (같은 시로 통합되는 컬럼들의 값을 합침)"""
        try:
            # CSV 파일 읽기
            df = pd.read_csv(csv_file_path, encoding="utf-8-sig")
            original_columns = df.columns.tolist()

            print(f"    📊 원본 컬럼 수: {len(original_columns)}")

            # 헤더 표준화 적용
            standardized_headers = {}
            for col in original_columns:
                standardized_col = col.strip()

                # 1단계: 시도명 표준화 (특별자치도 → 일반 명칭)
                for old_province, new_province in self.province_standardization.items():
                    if standardized_col.startswith(old_province):
                        standardized_col = standardized_col.replace(
                            old_province, new_province, 1
                        )
                        break

                # 2단계: 구 통합 (구별 지역 → 시 단위 통합)
                if standardized_col in self.district_to_city:
                    standardized_col = self.district_to_city[standardized_col]

                standardized_headers[col] = standardized_col

            # 같은 이름으로 통합되는 컬럼들 찾기
            consolidated_groups = {}
            for original_col, standardized_col in standardized_headers.items():
                if standardized_col not in consolidated_groups:
                    consolidated_groups[standardized_col] = []
                consolidated_groups[standardized_col].append(original_col)

            # 새로운 DataFrame 생성
            new_df = pd.DataFrame()

            for standardized_col, original_cols in consolidated_groups.items():
                if len(original_cols) == 1:
                    # 통합이 필요없는 컬럼은 그대로 복사
                    new_df[standardized_col] = df[original_cols[0]]
                else:
                    # 여러 컬럼을 합쳐야 하는 경우
                    print(
                        f"    🔗 컬럼 통합: {len(original_cols)}개 → {standardized_col}"
                    )
                    print(f"      └─ {', '.join(original_cols)}")

                    # 숫자형 컬럼인지 확인
                    numeric_data = []
                    for col in original_cols:
                        try:
                            # 숫자로 변환 시도
                            numeric_col = pd.to_numeric(df[col], errors="coerce")
                            numeric_data.append(numeric_col)
                        except:
                            # 숫자가 아닌 경우 0으로 처리
                            numeric_data.append(pd.Series([0] * len(df)))

                    # 컬럼들의 값을 합계
                    if numeric_data:
                        new_df[standardized_col] = pd.concat(numeric_data, axis=1).sum(
                            axis=1
                        )
                    else:
                        # 숫자가 아닌 경우 첫 번째 컬럼 값 사용
                        new_df[standardized_col] = df[original_cols[0]]

            # 파일 저장
            new_df.to_csv(csv_file_path, index=False, encoding="utf-8-sig")

            print(
                f"    ✅ 컬럼 통합 완료: {len(original_columns)} → {len(new_df.columns)} 컬럼"
            )

        except Exception as e:
            print(f"    ❌ 컬럼 통합 오류: {e}")

    def standardize_csv_headers(self, csv_file_path):
        """CSV 파일의 헤더를 표준화 (파일 직접 수정) - 구버전, consolidate_csv_columns 사용 권장"""
        try:
            # CSV 파일 읽기
            with open(csv_file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            # 첫 번째 줄(헤더) 분리
            lines = content.split("\n")
            if lines:
                header_line = lines[0]
                header_columns = header_line.split(",")

                # 각 헤더 컬럼에 대해 표준화 적용
                standardized_columns = []
                for col in header_columns:
                    col = col.strip()

                    # 1단계: 시도명 표준화 (특별자치도 → 일반 명칭)
                    for (
                        old_province,
                        new_province,
                    ) in self.province_standardization.items():
                        if col.startswith(old_province):
                            col = col.replace(old_province, new_province, 1)
                            break

                    # 2단계: 구 통합 (구별 지역 → 시 단위 통합)
                    if col in self.district_to_city:
                        col = self.district_to_city[col]

                    standardized_columns.append(col)

                # 중복 컬럼 처리 (같은 시 단위로 통합된 경우)
                unique_columns = []
                seen_columns = set()
                for col in standardized_columns:
                    if col not in seen_columns:
                        unique_columns.append(col)
                        seen_columns.add(col)
                    else:
                        # 중복된 컬럼은 번호를 붙여서 구분
                        counter = 2
                        new_col = f"{col}_{counter}"
                        while new_col in seen_columns:
                            counter += 1
                            new_col = f"{col}_{counter}"
                        unique_columns.append(new_col)
                        seen_columns.add(new_col)

                # 수정된 헤더로 교체
                lines[0] = ",".join(unique_columns)

                # 파일 다시 저장
                with open(csv_file_path, "w", encoding="utf-8-sig") as f:
                    f.write("\n".join(lines))

                print(
                    f"    📝 헤더 표준화 완료: {len(header_columns)} → {len(unique_columns)} 컬럼"
                )

        except Exception as e:
            print(f"  ⚠️ 헤더 표준화 오류: {e}")

    def process_migration_files(self, source_dir, target_dir=None):
        """마이그레이션 파일들을 일괄 처리하여 지역명 통합"""
        source_path = Path(source_dir)

        if target_dir is None:
            target_dir = source_path.parent / f"{source_path.name}_consolidated"

        target_path = Path(target_dir)
        target_path.mkdir(exist_ok=True)

        print(f"📂 소스 디렉토리: {source_path}")
        print(f"📂 대상 디렉토리: {target_path}")

        # CSV 파일 찾기
        csv_files = list(source_path.glob("*.csv"))
        if not csv_files:
            print("❌ CSV 파일을 찾을 수 없습니다.")
            return False

        print(f"📄 처리할 파일 수: {len(csv_files)}")

        processed_count = 0
        for csv_file in csv_files:
            try:
                print(f"\n🔄 처리 중: {csv_file.name}")

                # 파일 로드
                df = pd.read_csv(csv_file, encoding="utf-8-sig")
                print(f"  📊 원본 데이터: {df.shape}")

                # 지역명 컬럼 자동 감지
                possible_region_cols = [
                    "전출행정기관명_현재",
                    "전출행정기관명",
                    "지역명",
                    "시도",
                    "시군구",
                    "행정구역",
                    "지역",
                ]

                region_col = None
                for col in possible_region_cols:
                    if col in df.columns:
                        region_col = col
                        break

                if region_col is None:
                    print(
                        f"  ⚠️ 지역명 컬럼을 찾을 수 없습니다. 첫 번째 컬럼 사용: {df.columns[0]}"
                    )
                    region_col = df.columns[0]

                # 지역명 통합 처리
                consolidated_df = self.consolidate_dataframe(df, region_col)

                # 파일 저장
                output_file = target_path / csv_file.name
                consolidated_df.to_csv(output_file, index=False, encoding="utf-8-sig")

                # 컬럼 통합 후처리 (같은 시로 통합되는 컬럼들의 값을 합침)
                self.consolidate_csv_columns(output_file)

                print(f"  ✅ 저장 완료: {output_file.name}")
                processed_count += 1

            except Exception as e:
                print(f"  ❌ 오류 발생: {e}")
                continue

        print(f"\n🎉 처리 완료: {processed_count}/{len(csv_files)} 파일")
        return True

    def get_consolidation_summary(self):
        """통합 규칙 요약 정보 반환"""
        summary = {
            "총_통합_시_수": len(self.consolidation_rules),
            "총_통합_구_수": sum(
                len(districts) for districts in self.consolidation_rules.values()
            ),
            "시도명_표준화_규칙수": len(self.province_standardization),
            "통합_규칙": self.consolidation_rules,
            "시도명_표준화": self.province_standardization,
        }
        return summary


def main():
    """메인 실행 함수"""
    print("🚀 지역명 통합 처리 시작")
    print("=" * 50)

    # 통합기 생성
    consolidator = RegionConsolidator()

    # 통합 규칙 요약 출력
    summary = consolidator.get_consolidation_summary()
    print(
        f"📋 통합 대상: {summary['총_통합_시_수']}개 시, {summary['총_통합_구_수']}개 구"
    )
    print(f"📍 시도명 표준화: {summary['시도명_표준화_규칙수']}개 규칙")

    # 시도명 표준화 규칙 출력
    for old_name, new_name in summary["시도명_표준화"].items():
        print(f"   • {old_name} → {new_name}")

    # 현재 스크립트 위치 기준으로 경로 설정
    base_path = Path(__file__).parent

    # 청년 인구 이동량 데이터 처리
    youth_migration_dir = base_path / "청년 인구 이동량"

    if youth_migration_dir.exists():
        print(f"\n📁 청년 인구 이동량 데이터 처리...")
        consolidator.process_migration_files(youth_migration_dir)
    else:
        print(f"⚠️ 청년 인구 이동량 디렉토리를 찾을 수 없습니다: {youth_migration_dir}")

    print("\n✨ 지역명 통합 처리 완료!")


if __name__ == "__main__":
    main()
