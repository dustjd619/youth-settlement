from pathlib import Path

import pandas as pd


class MetropolitanYouthRatioCalculator:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.data_file = self.base_path / "data/policy/지자체별_청년인구비.csv"

        # 광역자치단체 목록
        self.metropolitan_areas = [
            "강원특별자치도",
            "경기도",
            "경상남도",
            "경상북도",
            "광주광역시",
            "대구광역시",
            "대전광역시",
            "부산광역시",
            "서울특별시",
            "세종특별자치시",
            "울산광역시",
            "인천광역시",
            "전라남도",
            "전북특별자치도",  # 데이터에서는 "전라북도"로 되어있음
            "제주특별자치도",
            "충청남도",
            "충청북도",
        ]

    def load_data(self):
        """기존 CSV 데이터를 로드합니다."""
        self.df = pd.read_csv(self.data_file, encoding="utf-8-sig")
        print(f"✓ 기존 데이터 로드: {len(self.df)}개 지역")

    def extract_metropolitan_area(self, region_name):
        """지역명에서 광역자치단체명을 추출합니다."""
        region_name = region_name.strip()

        # 매핑 규칙
        if region_name.startswith("강원"):
            return "강원특별자치도"
        elif region_name.startswith("경기도"):
            return "경기도"
        elif region_name.startswith("경상남도"):
            return "경상남도"
        elif region_name.startswith("경상북도"):
            return "경상북도"
        elif region_name.startswith("광주광역시"):
            return "광주광역시"
        elif region_name.startswith("대구광역시"):
            return "대구광역시"
        elif region_name.startswith("대전광역시"):
            return "대전광역시"
        elif region_name.startswith("부산광역시"):
            return "부산광역시"
        elif region_name.startswith("서울특별시"):
            return "서울특별시"
        elif region_name.startswith("세종특별자치시"):
            return "세종특별자치시"
        elif region_name.startswith("울산광역시"):
            return "울산광역시"
        elif region_name.startswith("인천광역시"):
            return "인천광역시"
        elif region_name.startswith("전라남도"):
            return "전라남도"
        elif region_name.startswith("전라북도"):
            return "전북특별자치도"
        elif region_name.startswith("제주특별자치도"):
            return "제주특별자치도"
        elif region_name.startswith("충청남도"):
            return "충청남도"
        elif region_name.startswith("충청북도"):
            return "충청북도"
        else:
            return None

    def calculate_metropolitan_ratios(self):
        """광역자치단체별 청년인구 비율을 계산합니다."""
        print("\\n=== 광역자치단체별 청년인구 비율 계산 ===")

        # 광역자치단체별 데이터 집계
        metropolitan_data = {}

        for _, row in self.df.iterrows():
            metro_area = self.extract_metropolitan_area(row["지자체명"])

            if (
                metro_area and metro_area != "세종특별자치시"
            ):  # 세종시는 이미 있으므로 제외
                if metro_area not in metropolitan_data:
                    metropolitan_data[metro_area] = {
                        "youth_population": 0,
                        "total_population": 0,
                        "regions": [],
                    }

                metropolitan_data[metro_area]["youth_population"] += row["청년인구"]
                metropolitan_data[metro_area]["total_population"] += row["전체인구"]
                metropolitan_data[metro_area]["regions"].append(row["지자체명"])

        # 결과 출력 및 새로운 행 생성
        new_rows = []

        for metro_area, data in metropolitan_data.items():
            youth_ratio = data["youth_population"] / data["total_population"]

            print(f"\\n📍 {metro_area}")
            print(f"   └ 하위 지역: {len(data['regions'])}개")
            print(f"   └ 청년인구: {data['youth_population']:,}명")
            print(f"   └ 전체인구: {data['total_population']:,}명")
            print(f"   └ 청년비율: {youth_ratio:.4f} ({youth_ratio*100:.2f}%)")

            # 행정코드는 임시로 99XXX 형태로 지정 (광역자치단체 코드)
            admin_code_mapping = {
                "서울특별시": 11000,
                "부산광역시": 21000,
                "대구광역시": 22000,
                "인천광역시": 23000,
                "광주광역시": 24000,
                "대전광역시": 25000,
                "울산광역시": 26000,
                "세종특별자치시": 29000,
                "경기도": 31000,
                "강원특별자치도": 32000,
                "충청북도": 33000,
                "충청남도": 34000,
                "전북특별자치도": 35000,
                "전라남도": 36000,
                "경상북도": 37000,
                "경상남도": 38000,
                "제주특별자치도": 39000,
            }

            new_row = {
                "지자체명": metro_area,
                "행정코드": admin_code_mapping.get(metro_area, 99000),
                "청년인구": data["youth_population"],
                "전체인구": data["total_population"],
                "청년비율": youth_ratio,
            }
            new_rows.append(new_row)

        return new_rows

    def save_updated_data(self, new_rows):
        """업데이트된 데이터를 저장합니다."""
        # 새로운 행들을 DataFrame으로 변환
        new_df = pd.DataFrame(new_rows)

        # 기존 데이터와 합치기
        updated_df = pd.concat([self.df, new_df], ignore_index=True)

        # 행정코드 순으로 정렬
        updated_df = updated_df.sort_values("행정코드").reset_index(drop=True)

        # 파일 저장
        output_file = (
            self.base_path / "data/policy/지자체별_청년인구비_with_metropolitan.csv"
        )
        updated_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\\n✅ 업데이트된 데이터 저장 완료: {output_file}")
        print(
            f"   └ 총 {len(updated_df)}개 지역 (기존 {len(self.df)}개 + 추가 {len(new_rows)}개)"
        )

        # 추가된 광역자치단체 목록 출력
        print(f"\\n📋 추가된 광역자치단체:")
        for i, row in enumerate(new_rows, 1):
            print(
                f"   {i:2d}. {row['지자체명']} (청년비율: {row['청년비율']*100:.2f}%)"
            )

        return output_file

    def run(self):
        """전체 프로세스를 실행합니다."""
        print("=== 광역자치단체 청년인구 비율 계산 시작 ===")

        # 1. 데이터 로드
        self.load_data()

        # 2. 광역자치단체별 비율 계산
        new_rows = self.calculate_metropolitan_ratios()

        # 3. 업데이트된 데이터 저장
        output_file = self.save_updated_data(new_rows)

        return output_file


if __name__ == "__main__":
    calculator = MetropolitanYouthRatioCalculator()
    calculator.run()
