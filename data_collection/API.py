import json
import os
from datetime import datetime

from data_collection.API import YouthPolicyAPI


def main():
    """2022년 10월부터 2025년 4월까지의 전체 청년정책 데이터를 JSON으로 수집"""

    # API 키 설정
    API_KEY = "610bb6ce-9ba2-4089-a4c7-0c25de78159e"

    # API 클래스 인스턴스 생성
    api = YouthPolicyAPI(API_KEY)

    print("=== 청년정책 전체 데이터 수집 (JSON 형태) ===")
    print("기간: 2022년 10월 1일 ~ 2025년 4월 30일")
    print()

    all_policies = []
    page_num = 1
    start_time = datetime.now()

    print(f"정책 데이터 수집 시작")

    while True:
        print(f"페이지 {page_num} 데이터 수집 중...")

        response = api.get_policy_list(page_num=page_num, page_size=100)

        if "error" in response:
            print(f"페이지 {page_num} 조회 실패: {response['error']}")
            break

        # 원본 JSON 응답 그대로 저장
        if "result" in response and "youthPolicyList" in response["result"]:
            policies = response["result"]["youthPolicyList"]
            total_count = response["result"]["pagging"]["totCount"]
            current_page_size = len(policies)

            print(f"  - 총 {total_count}개 중 {current_page_size}개 수집")

            # 각 정책에 수집 정보 추가
            for policy in policies:
                policy["_collected_at"] = datetime.now().isoformat()
                policy["_page_num"] = page_num

            all_policies.extend(policies)

            # 마지막 페이지인지 확인
            if current_page_size < 100:
                print("마지막 페이지 도달")
                break
        else:
            print("데이터가 없습니다.")
            break

        page_num += 1

        # 안전장치: 너무 많은 페이지 방지
        if page_num > 50:
            print("최대 페이지 수 도달")
            break

    end_time = datetime.now()
    print(f"\n수집 완료! 소요시간: {end_time - start_time}")

    if not all_policies:
        print("수집된 데이터가 없습니다.")
        return

    # 날짜 필터링 (2022-10-01 ~ 2025-04-30)
    filtered_policies = []
    start_date = datetime(2022, 10, 1)
    end_date = datetime(2025, 4, 30)

    for policy in all_policies:
        try:
            reg_date = datetime.strptime(policy["frstRegDt"], "%Y-%m-%d %H:%M:%S")
            if start_date <= reg_date <= end_date:
                filtered_policies.append(policy)
        except:
            # 날짜 파싱 실패시 포함
            filtered_policies.append(policy)

    print(f"날짜 필터링 후 {len(filtered_policies)}개 데이터")

    # JSON으로 저장
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # 메타데이터 포함
    output_data = {
        "metadata": {
            "collected_at": datetime.now().isoformat(),
            "period": "2022-10-01 to 2025-04-30",
            "total_count": len(filtered_policies),
            "collection_time": str(end_time - start_time),
            "api_key_used": "610bb6ce-9ba2-4089-a4c7-0c25de78159e",
        },
        "policies": filtered_policies,
    }

    json_path = os.path.join(output_dir, "youth_policies_2022-10_to_2025-04.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nJSON 저장 완료: {json_path}")

    # 파일 크기 확인
    file_size = os.path.getsize(json_path) / (1024 * 1024)  # MB
    print(f"파일 크기: {file_size:.2f} MB")

    # 데이터 통계
    print(f"\n=== 수집된 데이터 정보 ===")
    print(f"총 데이터 수: {len(filtered_policies)}개")

    # 카테고리별 통계
    categories = {}
    subcategories = {}

    for policy in filtered_policies:
        # 대분류
        if "lclsfNm" in policy:
            cat = policy["lclsfNm"]
            categories[cat] = categories.get(cat, 0) + 1

        # 중분류
        if "mclsfNm" in policy:
            subcat = policy["mclsfNm"]
            subcategories[subcat] = subcategories.get(subcat, 0) + 1

    print(f"\n=== 대분류별 상위 10개 ===")
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    for cat, count in sorted_categories:
        print(f"{cat}: {count}개")

    print(f"\n=== 중분류별 상위 10개 ===")
    sorted_subcategories = sorted(
        subcategories.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for subcat, count in sorted_subcategories:
        print(f"{subcat}: {count}개")

    return output_data


if __name__ == "__main__":
    data = main()
