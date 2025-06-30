import json
import os
from collections import Counter, defaultdict
from pathlib import Path


class PolicyRegionExtractor:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.policy_folder = self.base_path / "data/policy/정책책자"
        self.all_regions = []
        self.regions_by_province = defaultdict(list)

    def extract_all_regions(self):
        """모든 JSON 파일에서 지역 key들을 추출합니다."""
        print("=== 정책책자 JSON 파일 분석 ===\n")

        json_files = list(self.policy_folder.glob("*_정책_최종본.json"))

        for file_path in json_files:
            if file_path.name == "empty":
                continue

            # 광역자치단체명 추출 (파일명에서)
            province_name = file_path.stem.replace("_정책_최종본", "")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # JSON 파일의 최상위 key들 추출
                regions = list(data.keys())

                print(f"📁 {province_name}: {len(regions)}개 지역")
                for region in sorted(regions):
                    print(f"   - {region}")
                    self.all_regions.append(region)
                    self.regions_by_province[province_name].append(region)

                print()

            except Exception as e:
                print(f"❌ {file_path.name} 파일 읽기 오류: {e}")

        return self.all_regions, self.regions_by_province

    def analyze_regions(self):
        """지역 분석 수행"""
        all_regions, regions_by_province = self.extract_all_regions()

        print("=" * 80)
        print("📊 전체 분석 결과")
        print("=" * 80)

        # 전체 통계
        print(f"\n🏛️ 전체 지역 수: {len(all_regions)}개")
        print(f"🏛️ 광역자치단체 수: {len(regions_by_province)}개")

        # 광역자치단체별 지역 수
        print(f"\n📈 광역자치단체별 지역 수:")
        for province, regions in sorted(regions_by_province.items()):
            print(f"   {province}: {len(regions)}개")

        # 중복 지역명 확인
        region_counter = Counter(all_regions)
        duplicates = {
            region: count for region, count in region_counter.items() if count > 1
        }

        if duplicates:
            print(f"\n⚠️ 중복된 지역명:")
            for region, count in sorted(duplicates.items()):
                print(f"   {region}: {count}번 등장")
        else:
            print(f"\n✅ 중복된 지역명 없음")

        # 지역 유형별 분류
        print(f"\n🏷️ 지역 유형별 분류:")

        region_types = {"광역자치단체": [], "시": [], "군": [], "구": [], "기타": []}

        for region in set(all_regions):  # 중복 제거
            if any(
                keyword in region
                for keyword in ["특별시", "광역시", "특별자치도", "특별자치시"]
            ):
                if not any(
                    sub in region for sub in ["시", "군", "구"]
                ):  # 하위 행정구역이 포함되지 않은 경우만
                    region_types["광역자치단체"].append(region)
                else:
                    if "시" in region:
                        region_types["시"].append(region)
                    elif "군" in region:
                        region_types["군"].append(region)
                    elif "구" in region:
                        region_types["구"].append(region)
            elif "도" in region and not any(
                sub in region for sub in ["시", "군", "구"]
            ):
                region_types["광역자치단체"].append(region)
            elif "시" in region:
                region_types["시"].append(region)
            elif "군" in region:
                region_types["군"].append(region)
            elif "구" in region:
                region_types["구"].append(region)
            else:
                region_types["기타"].append(region)

        for region_type, regions in region_types.items():
            print(f"   {region_type}: {len(regions)}개")
            if len(regions) <= 10:  # 10개 이하면 모두 출력
                for region in sorted(regions):
                    print(f"      - {region}")
            else:  # 10개 초과면 일부만 출력
                for region in sorted(regions)[:5]:
                    print(f"      - {region}")
                print(f"      ... 외 {len(regions)-5}개")

        return all_regions, regions_by_province, region_types

    def save_results(self, all_regions, regions_by_province, region_types):
        """분석 결과를 파일로 저장"""
        # 전체 지역 목록 저장
        output_file = (
            self.base_path
            / "policy_evaluation/evaluation_results_index/정책_지역_목록.txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=== 정책책자 JSON 파일 지역 분석 결과 ===\n\n")

            f.write(f"전체 지역 수: {len(all_regions)}개\n")
            f.write(f"고유 지역 수: {len(set(all_regions))}개\n")
            f.write(f"광역자치단체 수: {len(regions_by_province)}개\n\n")

            f.write("=== 광역자치단체별 지역 목록 ===\n\n")
            for province, regions in sorted(regions_by_province.items()):
                f.write(f"{province} ({len(regions)}개 지역):\n")
                for region in sorted(regions):
                    f.write(f"  - {region}\n")
                f.write("\n")

            f.write("=== 지역 유형별 분류 ===\n\n")
            for region_type, regions in region_types.items():
                f.write(f"{region_type} ({len(regions)}개):\n")
                for region in sorted(regions):
                    f.write(f"  - {region}\n")
                f.write("\n")

            f.write("=== 전체 지역 목록 (중복 포함) ===\n\n")
            for i, region in enumerate(sorted(all_regions), 1):
                f.write(f"{i:3d}. {region}\n")

        print(f"\n💾 분석 결과 저장: {output_file}")

        # JSON 형태로도 저장
        json_output = (
            self.base_path
            / "policy_evaluation/evaluation_results_index/정책_지역_분석.json"
        )
        result_data = {
            "전체_지역_수": len(all_regions),
            "고유_지역_수": len(set(all_regions)),
            "광역자치단체_수": len(regions_by_province),
            "광역자치단체별_지역": dict(regions_by_province),
            "지역_유형별_분류": region_types,
            "전체_지역_목록": sorted(all_regions),
        }

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"💾 JSON 결과 저장: {json_output}")


def main():
    extractor = PolicyRegionExtractor()
    all_regions, regions_by_province, region_types = extractor.analyze_regions()
    extractor.save_results(all_regions, regions_by_province, region_types)


if __name__ == "__main__":
    main()
