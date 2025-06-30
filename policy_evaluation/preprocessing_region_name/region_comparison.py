from pathlib import Path

import pandas as pd


class RegionComparison:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent

    def load_policy_regions(self):
        """정책_지역_목록.txt에서 지역명을 로드합니다."""
        policy_file = (
            self.base_path
            / "policy_evaluation/preprocessing_region_name/정책_지역_목록.txt"
        )

        with open(policy_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 줄별로 분리하고 번호 제거
        lines = content.strip().split("\n")
        regions = []

        for line in lines:
            # 번호와 점 제거 (예: "1. 강원특별자치도" -> "강원특별자치도")
            if ". " in line:
                region = line.split(". ", 1)[1].strip()
                regions.append(region)

        return set(regions)

    def load_youth_population_regions(self):
        """지자체별_청년인구비.csv에서 지역명을 로드합니다."""
        youth_file = self.base_path / "data/policy/지자체별_청년인구비.csv"
        youth_df = pd.read_csv(youth_file, encoding="utf-8-sig")

        return set(youth_df["지자체명"].str.strip())

    def normalize_region_name(self, region_name):
        """지역명을 정규화합니다."""
        # 공통 변환
        normalized = region_name.strip()

        # 특별자치도 변환
        if "강원특별자치도" in normalized:
            normalized = normalized.replace("강원특별자치도", "강원도")
        if "전북특별자치도" in normalized:
            normalized = normalized.replace("전북특별자치도", "전라북도")
        if "전라북도" in normalized:
            normalized = normalized.replace("전라북도", "전북특별자치도")

        # 시/군/구 세분화 처리
        city_mappings = {
            "경기도 수원시": [
                "경기도 수원시 장안구",
                "경기도 수원시 권선구",
                "경기도 수원시 팔달구",
                "경기도 수원시 영통구",
            ],
            "경기도 성남시": [
                "경기도 성남시 수정구",
                "경기도 성남시 중원구",
                "경기도 성남시 분당구",
            ],
            "경기도 안양시": ["경기도 안양시 만안구", "경기도 안양시 동안구"],
            "경기도 안산시": ["경기도 안산시 상록구", "경기도 안산시 단원구"],
            "경기도 고양시": [
                "경기도 고양시 덕양구",
                "경기도 고양시 일산동구",
                "경기도 고양시 일산서구",
            ],
            "경기도 용인시": [
                "경기도 용인시 처인구",
                "경기도 용인시 기흥구",
                "경기도 용인시 수지구",
            ],
            "충청남도 천안시": ["충청남도 천안시 동남구", "충청남도 천안시 서북구"],
            "충청북도 청주시": [
                "충청북도 청주시 상당구",
                "충청북도 청주시 서원구",
                "충청북도 청주시 흥덕구",
                "충청북도 청주시 청원구",
            ],
            "전라북도 전주시": ["전라북도 전주시 완산구", "전라북도 전주시 덕진구"],
            "경상남도 창원시": [
                "경상남도 창원시 의창구",
                "경상남도 창원시 성산구",
                "경상남도 창원시 마산합포구",
                "경상남도 창원시 마산회원구",
                "경상남도 창원시 진해구",
            ],
            "경상북도 포항시": ["경상북도 포항시 남구", "경상북도 포항시 북구"],
        }

        return normalized, city_mappings.get(normalized, [])

    def compare_regions(self):
        """두 데이터셋의 지역을 비교합니다."""
        print("=== 정책 지역 목록 vs 청년인구 데이터 비교 ===\n")

        # 데이터 로드
        policy_regions = self.load_policy_regions()
        youth_regions = self.load_youth_population_regions()

        print(f"정책 지역 목록: {len(policy_regions)}개 지역")
        print(f"청년인구 데이터: {len(youth_regions)}개 지역\n")

        # 정책 지역 목록에 있지만 청년인구 데이터에 없는 지역
        missing_in_youth = []
        found_as_subdivisions = []

        for policy_region in policy_regions:
            normalized_region, subdivisions = self.normalize_region_name(policy_region)

            # 직접 매칭 시도
            direct_match = False
            for youth_region in youth_regions:
                if (
                    policy_region == youth_region
                    or normalized_region == youth_region
                    or policy_region in youth_region
                    or youth_region in policy_region
                ):
                    direct_match = True
                    break

            if not direct_match:
                # 세분화된 구역들이 있는지 확인
                subdivision_matches = []
                for subdivision in subdivisions:
                    if subdivision in youth_regions:
                        subdivision_matches.append(subdivision)

                if subdivision_matches:
                    found_as_subdivisions.append(
                        {
                            "policy_region": policy_region,
                            "subdivisions": subdivision_matches,
                        }
                    )
                else:
                    missing_in_youth.append(policy_region)

        # 결과 출력
        print("🔍 정책 지역 목록에는 있지만 청년인구 데이터에는 없는 지역:")
        print(f"총 {len(missing_in_youth)}개 지역\n")

        if missing_in_youth:
            for i, region in enumerate(sorted(missing_in_youth), 1):
                print(f"{i:2d}. {region}")
        else:
            print("누락된 지역이 없습니다.")

        print(f"\n🏙️ 세분화된 구역으로 존재하는 지역:")
        print(f"총 {len(found_as_subdivisions)}개 지역\n")

        if found_as_subdivisions:
            for i, item in enumerate(found_as_subdivisions, 1):
                print(f"{i:2d}. {item['policy_region']}")
                for subdivision in item["subdivisions"]:
                    print(f"    → {subdivision}")
                print()


if __name__ == "__main__":
    comparator = RegionComparison()
    comparator.compare_regions()
