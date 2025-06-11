import csv
import os
import re
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Chrome 옵션 설정
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ChromeDriver 자동 설치 및 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# 전역 변수로 번호 관리
policy_counter = 0


def save_policies_to_csv(policies, filename="local_youth_policies_new.csv"):
    """정책 데이터를 CSV 파일에 저장 (페이지별로 추가)"""
    global policy_counter

    try:
        # 기존 파일이 있는지 확인
        file_exists = os.path.exists(filename)

        # 파일이 새로 만들어지는 경우 카운터 초기화
        if not file_exists:
            policy_counter = 0

        with open(
            filename, mode="a" if file_exists else "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.writer(file)

            # 파일이 새로 만들어지는 경우에만 헤더 작성
            if not file_exists:
                writer.writerow(
                    [
                        "번호",
                        "지역명",
                        "정책명",
                        "정책유형",
                        "정책소개",
                        "사업운영기간",
                        "사업신청기간",
                        "주관기관",
                    ]
                )

            # 전역 카운터를 사용하여 번호 매기기
            for i, policy in enumerate(policies):
                policy_counter += 1
                writer.writerow(
                    [
                        policy_counter,
                        policy["region"],
                        policy["policy_name"],
                        policy["policy_type"],
                        policy["policy_introduction"],
                        policy["operation_period"],
                        policy["application_period"],
                        policy["agency"],
                    ]
                )

        print(
            f"✅ {len(policies)}개 정책을 '{filename}'에 저장했습니다. (번호 {policy_counter-len(policies)+1}~{policy_counter})"
        )
        return True
    except Exception as e:
        print(f"❌ CSV 저장 중 오류: {e}")
        return False


def extract_policy_details_from_detail_page(policy_id):
    """상세 페이지에서 모든 정책 정보 추출 (지역명, 정책유형, 정책명, 정책요약, 주관기관)"""
    try:
        # 실제 정책 상세 페이지 URL 구성
        detail_url = f"https://youth.seoul.go.kr/infoData/youthPlcyInfo/view.do?plcyBizId={policy_id}&tab=003&key=2309160001&sc_detailAt=&pageIndex=1&orderBy=regYmd+desc&blueWorksYn=N&tab="

        # 새 탭에서 상세 페이지 열기
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(detail_url)

        # 상세 페이지 로딩 대기
        time.sleep(3)

        policy_data = {
            "region": "",
            "policy_name": "",
            "policy_type": "",
            "policy_summary": "",
            "policy_introduction": "",  # 정책 소개 추가
            "operation_period": "",  # 사업 운영 기간 추가
            "application_period": "",  # 사업 신청 기간 추가
            "agency": "",
        }

        try:
            # 정책명 추출 - 스크린샷에서 확인한 구조: <strong class="title">
            policy_name_found = False

            # 1. strong.title에서 정책명 찾기 (스크린샷 구조)
            try:
                title_element = driver.find_element(By.CSS_SELECTOR, "strong.title")
                if title_element and title_element.text.strip():
                    policy_data["policy_name"] = title_element.text.strip()
                    print(f"    정책명: {policy_data['policy_name']}")
                    policy_name_found = True
            except:
                pass

            # 2. 다른 title 관련 셀렉터들 시도
            if not policy_name_found:
                title_selectors = [
                    ".title",  # title 클래스
                    "h1.title",  # h1 태그의 title 클래스
                    "h2.title",  # h2 태그의 title 클래스
                    ".lf strong",  # 스크린샷에서 보이는 구조
                    "strong",  # 일반 strong 태그
                    "h1",  # h1 태그
                    "h2",  # h2 태그
                ]

                for selector in title_selectors:
                    try:
                        title_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        for title_element in title_elements:
                            title_text = title_element.text.strip()
                            if title_text and len(title_text) > 5:
                                policy_data["policy_name"] = title_text
                                print(
                                    f"    정책명({selector}): {policy_data['policy_name']}"
                                )
                                policy_name_found = True
                                break
                        if policy_name_found:
                            break
                    except:
                        continue

            # 테이블에서 정보 추출
            table_rows = driver.find_elements(
                By.CSS_SELECTOR, "table tr, .form-table tr, .form-resp-table tr"
            )

            for row in table_rows:
                row_text = row.text.strip()
                if not row_text:
                    continue

                # 시행지역 찾기
                if "시행지역" in row_text and not policy_data["region"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if "시행지역" in cell.text:
                                if i + 1 < len(cells):
                                    policy_data["region"] = cells[i + 1].text.strip()
                                    print(f"    시행지역: {policy_data['region']}")
                                break
                    except Exception as e:
                        print(f"    시행지역 추출 중 오류: {e}")

                # 정책유형 찾기
                if (
                    "정책유형" in row_text or "정책 유형" in row_text
                ) and not policy_data["policy_type"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if "정책유형" in cell.text or "정책 유형" in cell.text:
                                if i + 1 < len(cells):
                                    policy_data["policy_type"] = cells[
                                        i + 1
                                    ].text.strip()
                                    print(f"    정책유형: {policy_data['policy_type']}")
                                break
                    except Exception as e:
                        print(f"    정책유형 추출 중 오류: {e}")

                # 주관기관 찾기
                if (
                    "주관기관" in row_text or "주관 기관" in row_text
                ) and not policy_data["agency"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if "주관기관" in cell.text or "주관 기관" in cell.text:
                                if i + 1 < len(cells):
                                    policy_data["agency"] = cells[i + 1].text.strip()
                                    print(f"    주관기관: {policy_data['agency']}")
                                break
                    except Exception as e:
                        print(f"    주관기관 추출 중 오류: {e}")

                # 사업 운영 기간 찾기
                if (
                    "사업운영기간" in row_text or "사업 운영 기간" in row_text
                ) and not policy_data["operation_period"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if (
                                "사업운영기간" in cell.text
                                or "사업 운영 기간" in cell.text
                            ):
                                if i + 1 < len(cells):
                                    policy_data["operation_period"] = cells[
                                        i + 1
                                    ].text.strip()
                                    print(
                                        f"    사업운영기간: {policy_data['operation_period']}"
                                    )
                                break
                    except Exception as e:
                        print(f"    사업운영기간 추출 중 오류: {e}")

                # 사업 신청 기간 찾기
                if (
                    "사업신청기간" in row_text or "사업 신청 기간" in row_text
                ) and not policy_data["application_period"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if (
                                "사업신청기간" in cell.text
                                or "사업 신청 기간" in cell.text
                            ):
                                if i + 1 < len(cells):
                                    policy_data["application_period"] = cells[
                                        i + 1
                                    ].text.strip()
                                    print(
                                        f"    사업신청기간: {policy_data['application_period']}"
                                    )
                                break
                    except Exception as e:
                        print(f"    사업신청기간 추출 중 오류: {e}")

                # 정책소개 추출 - 스크린샷에서 확인한 구조
                if "정책 소개" in row_text and not policy_data["policy_introduction"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if "정책 소개" in cell.text:
                                # 다음 td에서 내용 찾기 (일반적인 경우)
                                if i + 1 < len(cells):
                                    policy_data["policy_introduction"] = cells[
                                        i + 1
                                    ].text.strip()
                                    print(
                                        f"    정책소개: {policy_data['policy_introduction'][:50]}..."
                                    )
                                    break
                                # colspan이 있는 경우 같은 행의 다른 td 찾기
                                else:
                                    colspan_cells = row.find_elements(
                                        By.CSS_SELECTOR, "td[colspan]"
                                    )
                                    for colspan_cell in colspan_cells:
                                        content = colspan_cell.text.strip()
                                        if (
                                            content and len(content) > 10
                                        ):  # 충분한 길이의 내용
                                            policy_data["policy_introduction"] = content
                                            print(
                                                f"    정책소개(colspan): {policy_data['policy_introduction'][:50]}..."
                                            )
                                            break
                                break
                    except Exception as e:
                        print(f"    정책소개 추출 중 오류: {e}")

                # 정책개요/정책요약 추출
                if (
                    "정책개요" in row_text
                    or "정책 개요" in row_text
                    or "정책요약" in row_text
                ) and not policy_data["policy_summary"]:
                    try:
                        cells = row.find_elements(By.CSS_SELECTOR, "td, th")
                        for i, cell in enumerate(cells):
                            if (
                                "정책개요" in cell.text
                                or "정책 개요" in cell.text
                                or "정책요약" in cell.text
                            ):
                                if i + 1 < len(cells):
                                    policy_data["policy_summary"] = cells[
                                        i + 1
                                    ].text.strip()
                                    print(
                                        f"    정책요약: {policy_data['policy_summary'][:50]}..."
                                    )
                                break
                    except Exception as e:
                        print(f"    정책요약 추출 중 오류: {e}")

        except Exception as e:
            print(f"상세 페이지에서 정보 추출 중 오류: {e}")

        # 원래 탭으로 돌아가기
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

        return policy_data

    except Exception as e:
        print(f"상세 페이지 처리 중 오류: {e}")
        # 오류 발생 시 원래 탭으로 돌아가기
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        return None


def crawl_policies_from_current_page():
    """현재 페이지의 정책 ID들을 크롤링"""
    policies = []

    try:
        # 페이지 로딩 대기
        time.sleep(3)

        print("정책 리스트 검색 중...")

        # 정책 리스트 항목들 찾기 (스크린샷 구조: ul.policy-list > li)
        policy_list_items = driver.find_elements(By.CSS_SELECTOR, ".policy-list li")
        print(f"정책 리스트 항목: {len(policy_list_items)}개")

        # 각 정책 항목에서 지역명과 정책 ID 추출
        for i, policy_item in enumerate(policy_list_items, 1):
            try:
                # 지역명 추출 (스크린샷 구조: span.bg-purple)
                region = ""
                try:
                    region_element = policy_item.find_element(
                        By.CSS_SELECTOR, ".bg-purple"
                    )
                    region = region_element.text.strip()
                    print(f"\n  정책 {i}: 지역 {region}")
                except:
                    print(f"\n  정책 {i}: 지역명 없음")

                # 정책 ID 추출 (onclick에서)
                policy_id = ""
                try:
                    onclick_elements = policy_item.find_elements(
                        By.CSS_SELECTOR, "[onclick*='goView']"
                    )
                    for onclick_element in onclick_elements:
                        onclick_attr = onclick_element.get_attribute("onclick")
                        if onclick_attr and "goView" in onclick_attr:
                            match = re.search(r"goView\('([^']+)'\)", onclick_attr)
                            if match:
                                policy_id = match.group(1)
                                print(f"    정책 ID: {policy_id}")
                                break
                except:
                    pass

                if policy_id:
                    # 상세 페이지에서 나머지 정보 추출
                    policy_data = extract_policy_details_from_detail_page(policy_id)

                    if policy_data:
                        # 리스트 페이지에서 추출한 지역명으로 업데이트
                        if region:
                            policy_data["region"] = region

                        policies.append(policy_data)
                        print(
                            f"  ✅ 정책 {i} 완료: {policy_data['region']} - {policy_data['policy_name'][:30]}..."
                        )
                    else:
                        print(f"  ❌ 정책 {i} 실패")
                else:
                    print(f"  ❌ 정책 {i}: 정책 ID를 찾을 수 없음")

            except Exception as e:
                print(f"정책 {i} 처리 중 오류: {e}")
                continue

    except Exception as e:
        print(f"페이지 크롤링 중 오류: {e}")

    return policies


try:
    # URL - list2.do 페이지 (지역 정책) - 처음부터 시작
    url = "https://youth.seoul.go.kr/infoData/youthPlcyInfo/list2.do?plcyBizId=&tab=003&key=2309160001&sc_detailAt=&pageIndex=1&orderBy=regYmd+desc&blueWorksYn=N&tabKind=003"

    print("페이지 로딩 중...")
    driver.get(url)
    time.sleep(5)

    print("현재 페이지 제목:", driver.title)
    print("현재 URL:", driver.current_url)

    # 전체 정책 수집
    all_policies = []
    current_page = 1  # 1페이지부터 시작
    max_pages = 1868  # 기본값

    # CSV 파일 준비 - 새로 생성
    csv_filename = "local_youth_policies_new.csv"

    print(f"\n=== 지역 청년정책 크롤링 시작 (처음부터 새로 시작) ===")

    # 첫 페이지에서 총 페이지 수 확인
    try:
        # 총 개수 정보 찾기
        total_elements = driver.find_elements(By.CSS_SELECTOR, "body")
        if total_elements:
            body_text = total_elements[0].text
            # "타지역 정책(9,338건)" 같은 패턴 찾기
            total_match = re.search(r"타지역\s*정책\s*\(([0-9,]+)\s*건\)", body_text)
            if total_match:
                total_count_str = total_match.group(1).replace(",", "")
                total_count = int(total_count_str)
                print(f"총 지역 정책 수: {total_count}")
                # 페이지당 5개씩이라고 가정
                max_pages = (total_count // 5) + 1
                print(f"예상 최대 페이지 수: {max_pages}")

    except Exception as e:
        print(f"페이지 정보 확인 중 오류: {e}")

    while True:
        print(f"\n--- 페이지 {current_page} 크롤링 중 ---")

        # 현재 페이지의 정책들 크롤링
        page_policies = crawl_policies_from_current_page()

        # 현재 페이지에서 정책이 없으면 종료
        if not page_policies:
            print("현재 페이지에서 정책을 찾을 수 없습니다. 크롤링을 종료합니다.")
            break

        # 페이지별로 바로 저장 (중복 방지)
        if page_policies:
            save_policies_to_csv(page_policies, csv_filename)

        all_policies.extend(page_policies)

        print(f"페이지 {current_page}에서 {len(page_policies)}개 정책 수집")
        print(f"총 수집된 정책: {len(all_policies)}개")

        # 다음 페이지로 이동
        try:
            next_page_number = current_page + 1

            # 마지막 페이지 확인
            if current_page >= max_pages:
                print(f"최대 페이지 수({max_pages})에 도달했습니다.")
                break

            # URL 직접 변경으로 다음 페이지 이동
            print(f"페이지 {next_page_number}으로 이동 중...")

            current_url = driver.current_url
            if "pageIndex=" in current_url:
                new_url = re.sub(
                    r"pageIndex=\d+", f"pageIndex={next_page_number}", current_url
                )
            else:
                separator = "&" if "?" in current_url else "?"
                new_url = f"{current_url}{separator}pageIndex={next_page_number}"

            driver.get(new_url)
            current_page = next_page_number
            print(f"페이지 {next_page_number} 이동 완료")
            time.sleep(4)

            # 페이지 이동 성공 확인
            final_url = driver.current_url
            if "pageIndex=" in final_url:
                match = re.search(r"pageIndex=(\d+)", final_url)
                if match:
                    actual_page = int(match.group(1))
                    if actual_page < next_page_number:
                        print("마지막 페이지에 도달한 것 같습니다.")
                        break

            # 안전장치
            if len(all_policies) > 10000:  # 너무 많이 수집되면 중단
                print("예상보다 많은 정책이 수집되었습니다. 크롤링을 중단합니다.")
                break

        except Exception as page_e:
            print(f"페이지 이동 중 오류: {page_e}")
            break

    # ✅ 최종 저장 (페이지별로 이미 저장했으므로 불필요)
    # if all_policies:
    #     print(f"\n=== 최종 저장 중 ===")
    #     save_policies_to_csv(all_policies, csv_filename, is_final=True)

    # 최종 카운트 확인
    final_count = 0
    try:
        with open(csv_filename, "r", encoding="utf-8-sig") as file:
            lines = file.readlines()
            final_count = len(lines) - 1  # 헤더 제외
    except:
        final_count = len(all_policies)

    print(
        f"✅ 총 {final_count}개의 지역 정책 정보를 '{csv_filename}' 파일로 저장했습니다."
    )
    print(f"크롤링한 마지막 페이지: {current_page}")
    print(f"이번 세션에서 추가된 정책: {len(all_policies)}개")

except Exception as e:
    print(f"전체 프로세스 중 오류 발생: {e}")

finally:
    driver.quit()

print("\n지역 정책 크롤링 완료!")
