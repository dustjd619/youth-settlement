import csv
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


def extract_agency_from_detail_page(policy_id):
    """상세 페이지에서 주관기관 정보 추출"""
    try:
        # 실제 정책 상세 페이지 URL 구성
        detail_url = f"https://youth.seoul.go.kr/infoData/youthPlcyInfo/view.do?plcyBizId={policy_id}&tab=003&key=2309160001&sc_detailAt=&pageIndex=1&orderBy=regYmd+desc&blueWorksYn=N&tab="

        # 새 탭에서 상세 페이지 열기
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(detail_url)

        # 상세 페이지 로딩 대기
        time.sleep(3)

        agency = ""
        # 주관기관 정보 찾기
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text

            if "주관 기관" in body_text:
                rows = driver.find_elements(By.CSS_SELECTOR, "table tr, .form-table tr")

                for row in rows:
                    row_text = row.text.strip()
                    if "주관 기관" in row_text:
                        # 행 텍스트에서 직접 추출
                        parts = row_text.split("주관 기관")
                        if len(parts) > 1:
                            agency_part = parts[1].strip()
                            agency_words = agency_part.split()
                            if agency_words:
                                agency = agency_words[0]
                                break
        except Exception as e:
            print(f"상세 페이지에서 주관기관 추출 중 오류: {e}")

        # 원래 탭으로 돌아가기
        driver.close()
        driver.switch_to.window(driver.window_handles[0])

        return agency

    except Exception as e:
        print(f"상세 페이지 처리 중 오류: {e}")
        # 오류 발생 시 원래 탭으로 돌아가기
        if len(driver.window_handles) > 1:
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
        return ""


def crawl_policies_from_current_page():
    """현재 페이지의 정책들을 크롤링"""
    policies = []

    try:
        # 페이지 로딩 대기
        time.sleep(3)

        # 다양한 셀렉터로 정책 리스트 찾기
        print("정책 리스트 검색 중...")

        # 방법 1: 기본 셀렉터 (list2.do 페이지 구조에 맞게)
        policy_list = driver.find_elements(By.CSS_SELECTOR, "ul.policy-list > li")
        print(f"셀렉터 'ul.policy-list > li'로 발견된 정책: {len(policy_list)}개")

        if not policy_list:
            policy_list = driver.find_elements(By.CSS_SELECTOR, ".policy-list li")
            print(f"셀렉터 '.policy-list li'로 발견된 정책: {len(policy_list)}개")

        if not policy_list:
            # list2.do에서 사용되는 다른 가능한 셀렉터들
            policy_list = driver.find_elements(By.CSS_SELECTOR, "li[onclick*='goView']")
            print(
                f"셀렉터 'li[onclick*=\"goView\"]'로 발견된 정책: {len(policy_list)}개"
            )

        if not policy_list:
            # 방법 2: 더 광범위한 검색
            policy_list = driver.find_elements(By.CSS_SELECTOR, "li")
            print(f"모든 li 요소 개수: {len(policy_list)}")

            # 정책 관련 li만 필터링
            filtered_policies = []
            for li in policy_list:
                text = li.text.strip()
                # 정책처럼 보이는 항목들 필터링 (제목이 있고 충분한 텍스트가 있는 것)
                if len(text) > 20 and (
                    "정책" in text
                    or "지원" in text
                    or "사업" in text
                    or "프로그램" in text
                    or "모집" in text
                    or "장학" in text
                ):

                    # 링크가 있는지 확인
                    link_elem = li.find_elements(By.CSS_SELECTOR, "a")
                    if link_elem:
                        onclick_attr = link_elem[0].get_attribute("onclick")
                        if onclick_attr and (
                            "goView" in onclick_attr or "view" in onclick_attr.lower()
                        ):
                            filtered_policies.append(li)

            policy_list = filtered_policies
            print(f"필터링된 정책 항목: {len(policy_list)}개")

        if not policy_list:
            # 방법 3: 다른 가능한 구조 찾기
            print("다른 구조로 정책 찾기 시도...")
            possible_selectors = [
                ".list-item",
                ".policy-item",
                ".item",
                "tr",  # 테이블 형태일 가능성
                ".row",
                "[onclick*='goView']",  # 클릭 이벤트가 있는 요소들
            ]

            for selector in possible_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    print(f"셀렉터 '{selector}'로 {len(elements)}개 요소 발견")
                    # goView가 포함된 요소만 필터링
                    filtered = []
                    for elem in elements:
                        onclick = elem.get_attribute("onclick") or ""
                        if "goView" in onclick:
                            filtered.append(elem)
                    if filtered:
                        print(f"  그 중 goView가 포함된 정책: {len(filtered)}개")
                        policy_list = filtered
                        break

        print(f"최종 발견된 정책 개수: {len(policy_list)}")

        # 페이지 전체 HTML 구조 확인 (처음 몇 번만)
        if current_page <= 2:
            print("\n=== 페이지 구조 분석 ===")
            try:
                # 전체 본문 영역 찾기
                main_content = driver.find_elements(
                    By.CSS_SELECTOR, ".main-content, .content, #content, .container"
                )
                for i, content in enumerate(main_content):
                    print(
                        f"주요 콘텐츠 영역 {i+1}: {content.tag_name}, 클래스: {content.get_attribute('class')}"
                    )

                # goView가 포함된 모든 요소 찾기
                all_goview_elements = driver.find_elements(
                    By.CSS_SELECTOR, "[onclick*='goView']"
                )
                print(f"goView 이벤트가 있는 전체 요소: {len(all_goview_elements)}개")

                for i, elem in enumerate(all_goview_elements[:5]):  # 처음 5개만 출력
                    onclick = elem.get_attribute("onclick")
                    text = elem.text.strip()[:50]  # 처음 50자만
                    print(
                        f"  {i+1}. {elem.tag_name} - onclick: {onclick}, 텍스트: {text}..."
                    )

            except Exception as debug_e:
                print(f"디버깅 중 오류: {debug_e}")

        for i, policy in enumerate(policy_list, 1):
            try:
                # 정책 전체 텍스트
                full_text = policy.text.strip()

                # 개별 요소들 찾기
                policy_type = ""
                policy_title = ""
                policy_summary = ""
                agency = ""

                # 전체 텍스트를 줄별로 분리하여 파싱
                lines = [line.strip() for line in full_text.split("\n") if line.strip()]

                if lines:
                    # 첫 번째 줄: 정책유형
                    if len(lines) >= 1:
                        policy_type = lines[0]

                    # 두 번째 줄: 정책명
                    if len(lines) >= 2:
                        policy_title = lines[1]

                    # 세 번째 줄 이후: 정책 요약
                    if len(lines) >= 3:
                        policy_summary = " ".join(lines[2:])

                # 정책 ID 추출 및 상세 페이지에서 주관기관 가져오기
                try:
                    link_elem = policy.find_element(By.CSS_SELECTOR, "a")
                    if link_elem:
                        onclick_attr = link_elem.get_attribute("onclick")
                        if onclick_attr and (
                            "view" in onclick_attr.lower()
                            or "goview" in onclick_attr.lower()
                        ):
                            import re

                            match = re.search(r"goView\('([^']+)'\)", onclick_attr)
                            if match:
                                policy_id = match.group(1)
                                print(f"  정책 {i}: {policy_title} (ID: {policy_id})")
                                agency = extract_agency_from_detail_page(policy_id)
                                print(f"    주관기관: {agency}")

                except Exception as link_e:
                    print(f"  정책 {i} 링크 처리 중 오류: {link_e}")

                # 정책 정보 저장
                policy_data = {
                    "type": policy_type,
                    "title": policy_title,
                    "summary": policy_summary,
                    "agency": agency,
                }
                policies.append(policy_data)

            except Exception as e:
                print(f"정책 {i} 처리 중 오류: {e}")
                continue

    except Exception as e:
        print(f"페이지 크롤링 중 오류: {e}")

    return policies


try:
    # URL
    url = "https://youth.seoul.go.kr/infoData/youthPlcyInfo/list2.do?plcyBizId=&tab=003&key=2309160001&sc_detailAt=&pageIndex=1&orderBy=regYmd+desc&blueWorksYn=N&tabKind=003"

    print("페이지 로딩 중...")
    driver.get(url)
    time.sleep(5)

    print("현재 페이지 제목:", driver.title)
    print("현재 URL:", driver.current_url)

    # 전체 정책 수집
    all_policies = []
    current_page = 1
    max_pages = 250  # 기본값, 실제로는 페이지에서 확인

    # CSV 파일 준비
    csv_filename = "all_youth_policies_list2.csv"

    print(f"\n=== 모든 페이지 크롤링 시작 ===")

    # 첫 페이지에서 총 페이지 수 확인
    try:
        # 총 개수나 마지막 페이지 번호 찾기
        page_info_elements = driver.find_elements(
            By.CSS_SELECTOR, ".pagination a, .paging a"
        )
        page_numbers = []
        for elem in page_info_elements:
            text = elem.text.strip()
            if text.isdigit():
                page_numbers.append(int(text))

        if page_numbers:
            visible_max = max(page_numbers)
            print(f"현재 보이는 최대 페이지 번호: {visible_max}")

        # 총 개수 정보 찾기
        total_elements = driver.find_elements(By.CSS_SELECTOR, "body")
        if total_elements:
            body_text = total_elements[0].text
            # "총 1194건" 같은 패턴 찾기
            import re

            total_match = re.search(r"총\s*(\d+)\s*건", body_text)
            if total_match:
                total_count = int(total_match.group(1))
                print(f"총 정책 수: {total_count}")
                # 페이지당 5개씩이라고 가정 (실제 확인 필요)
                max_pages = (total_count // 5) + 1
                print(f"예상 최대 페이지 수: {max_pages}")

    except Exception as e:
        print(f"페이지 정보 확인 중 오류: {e}")

    while True:
        print(f"\n--- 페이지 {current_page} 크롤링 중 ---")

        # 현재 페이지의 URL 저장 (페이지 변경 확인용)
        current_url = driver.current_url

        # 현재 페이지의 정책들 크롤링
        page_policies = crawl_policies_from_current_page()

        # 현재 페이지에서 정책이 없으면 종료
        if not page_policies:
            print("현재 페이지에서 정책을 찾을 수 없습니다. 크롤링을 종료합니다.")
            break

        all_policies.extend(page_policies)

        print(f"페이지 {current_page}에서 {len(page_policies)}개 정책 수집")
        print(f"총 수집된 정책: {len(all_policies)}개")

        # 페이지네이션 정보 확인
        try:
            # 현재 페이지 번호 추출 (URL과 페이지네이션 모두에서)
            current_page_from_url = None
            current_page_from_pagination = None

            # URL에서 pageIndex 추출
            if "pageIndex=" in driver.current_url:
                import re

                match = re.search(r"pageIndex=(\d+)", driver.current_url)
                if match:
                    current_page_from_url = int(match.group(1))
                    print(f"URL에서 추출한 페이지 번호: {current_page_from_url}")

            # 페이지네이션에서 현재 활성화된 페이지 번호 확인
            active_page_elements = driver.find_elements(
                By.CSS_SELECTOR,
                ".pagination .on, .pagination .current, .page_on, .pagination a[style*='background'], .pagination span[style*='background']",
            )

            if not active_page_elements:
                # 다른 방법으로 현재 페이지 찾기 (검은색 배경이나 특별한 스타일)
                all_page_numbers = driver.find_elements(
                    By.CSS_SELECTOR, ".pagination a, .pagination span"
                )
                for elem in all_page_numbers:
                    if elem.text.isdigit() and (
                        "background" in elem.get_attribute("style")
                        or elem.get_attribute("class")
                        and "on" in elem.get_attribute("class")
                    ):
                        active_page_elements = [elem]
                        break

            if active_page_elements:
                page_text = active_page_elements[0].text.strip()
                if page_text.isdigit():
                    current_page_from_pagination = int(page_text)
                    print(
                        f"페이지네이션에서 추출한 현재 페이지: {current_page_from_pagination}"
                    )

            # 총 개수 정보 확인
            total_count_element = driver.find_elements(
                By.CSS_SELECTOR, ".total, .cnt, [class*='total']"
            )
            if total_count_element:
                for elem in total_count_element:
                    if "1194" in elem.text or "총" in elem.text:
                        print(f"전체 개수 정보: {elem.text}")
                        break
        except Exception as e:
            print(f"페이지네이션 정보 확인 중 오류: {e}")

        # 다음 페이지 찾기 및 이동
        try:
            is_last_page = False
            next_page_number = current_page + 1

            # 마지막 페이지 확인 방법들

            # 방법 1: 페이지 239인지 확인 (스크린샷에서 확인된 마지막 페이지)
            if current_page_from_pagination and current_page_from_pagination >= 239:
                print(
                    f"현재 페이지가 {current_page_from_pagination}로 마지막 페이지 범위입니다."
                )
                is_last_page = True

            # 방법 2: URL의 pageIndex 확인
            if current_page_from_url and current_page_from_url >= 239:
                print(
                    f"URL의 페이지 인덱스가 {current_page_from_url}로 마지막 페이지 범위입니다."
                )
                is_last_page = True

            if is_last_page:
                print("마지막 페이지에 도달했습니다.")
                break

            # 다음 페이지 번호를 직접 찾아서 클릭
            print(f"페이지 {next_page_number} 버튼을 찾는 중...")

            # 페이지 번호 버튼 찾기 (여러 방법 시도)
            next_page_button = None

            # 방법 1: 정확한 페이지 번호 텍스트 찾기
            page_number_buttons = driver.find_elements(
                By.CSS_SELECTOR, ".pagination a, .paging a, [class*='page'] a"
            )
            for btn in page_number_buttons:
                if btn.text.strip() == str(next_page_number):
                    onclick = btn.get_attribute("onclick") or ""
                    if onclick and btn.is_enabled() and btn.is_displayed():
                        next_page_button = btn
                        print(
                            f"페이지 {next_page_number} 버튼을 찾았습니다: {onclick[:100]}..."
                        )
                        break

            # 방법 2: onclick에서 페이지 번호 확인
            if not next_page_button:
                all_links = driver.find_elements(
                    By.CSS_SELECTOR, "a[onclick*='egov_link_page']"
                )
                for btn in all_links:
                    onclick = btn.get_attribute("onclick") or ""
                    # onclick에서 페이지 번호 추출
                    import re

                    match = re.search(r"egov_link_page\((\d+)\)", onclick)
                    if match and int(match.group(1)) == next_page_number:
                        if btn.is_enabled() and btn.is_displayed():
                            next_page_button = btn
                            print(
                                f"onclick에서 페이지 {next_page_number} 버튼을 찾았습니다"
                            )
                            break

            # 방법 3: URL 직접 변경으로 이동 (가장 확실한 방법)
            if not next_page_button:
                print(
                    f"페이지 {next_page_number} 버튼을 찾을 수 없어서 URL 직접 변경으로 이동합니다"
                )

            # URL 직접 변경 방식을 우선적으로 사용 (더 안정적)
            print(f"URL 직접 변경으로 페이지 {next_page_number}으로 이동합니다")

            # 현재 URL에서 pageIndex만 변경
            current_url = driver.current_url
            if "pageIndex=" in current_url:
                new_url = re.sub(
                    r"pageIndex=\d+", f"pageIndex={next_page_number}", current_url
                )
            else:
                separator = "&" if "?" in current_url else "?"
                new_url = f"{current_url}{separator}pageIndex={next_page_number}"

            old_url = current_url
            driver.get(new_url)
            current_page = next_page_number
            print(f"URL로 페이지 {next_page_number}으로 이동: {new_url}")
            time.sleep(4)

            # 페이지 이동 성공 확인
            final_url = driver.current_url
            if "pageIndex=" in final_url:
                match = re.search(r"pageIndex=(\d+)", final_url)
                if match:
                    actual_page = int(match.group(1))
                    if actual_page == next_page_number:
                        print(f"페이지 {next_page_number} 이동 성공!")
                        continue
                    else:
                        print(
                            f"예상 페이지: {next_page_number}, 실제 페이지: {actual_page}"
                        )
                        # 실제 페이지가 예상과 다르면 마지막 페이지일 가능성
                        if actual_page < next_page_number:
                            print("마지막 페이지에 도달한 것 같습니다.")
                            break
            continue

            # 다음 페이지로 이동 시도
            if next_page_button:
                print("다음 페이지로 이동 시도...")
                try:
                    # 현재 URL 저장
                    old_url = driver.current_url
                    old_page_index = current_page_from_url or current_page

                    driver.execute_script("arguments[0].click();", next_page_button)
                    current_page = next_page_number

                    # 페이지 로딩 대기
                    time.sleep(4)

                    # 페이지 변경 확인
                    new_url = driver.current_url
                    new_page_index = None

                    if "pageIndex=" in new_url:
                        match = re.search(r"pageIndex=(\d+)", new_url)
                        if match:
                            new_page_index = int(match.group(1))

                    # URL이나 페이지 인덱스가 변경되지 않았으면 마지막 페이지
                    if new_url == old_url or (
                        new_page_index
                        and old_page_index
                        and new_page_index <= old_page_index
                    ):
                        print(
                            f"페이지가 변경되지 않았습니다. (이전: {old_page_index}, 현재: {new_page_index})"
                        )
                        print("마지막 페이지에 도달한 것 같습니다.")
                        break

                    print(f"페이지 이동 완료. 새 페이지 인덱스: {new_page_index}")

                except Exception as click_e:
                    print(f"다음 페이지 버튼 클릭 중 오류: {click_e}")
                    break
            else:
                print("다음 페이지 버튼을 찾을 수 없습니다.")
                break

            # 안전장치들
            if current_page > max_pages:
                print(f"최대 페이지 수({max_pages})에 도달했습니다.")
                break

            if len(all_policies) > 1500:  # 1194개보다 여유있게
                print("예상보다 많은 정책이 수집되었습니다. 크롤링을 중단합니다.")
                break

        except Exception as page_e:
            print(f"페이지 이동 중 오류: {page_e}")
            break

    # CSV 파일로 저장
    print(f"\n=== CSV 파일 저장 중 ===")
    with open(csv_filename, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["번호", "정책유형", "정책명", "정책 요약", "주관기관"])

        for i, policy in enumerate(all_policies, 1):
            writer.writerow(
                [
                    i,
                    policy["type"],
                    policy["title"],
                    policy["summary"],
                    policy["agency"],
                ]
            )

    print(
        f"✅ 총 {len(all_policies)}개의 정책 정보를 '{csv_filename}' 파일로 저장했습니다."
    )
    print(f"크롤링한 페이지 수: {current_page}")

except Exception as e:
    print(f"전체 프로세스 중 오류 발생: {e}")

finally:
    driver.quit()

print("\n스크립트 완료!")
