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


def extract_agency_from_detail_page(policy_url):
    """상세 페이지에서 주관기관 정보 추출"""
    try:
        # 새 탭에서 상세 페이지 열기
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(policy_url)

        # 상세 페이지 로딩 대기
        time.sleep(2)

        agency = "서울시"  # 기본값
        # 주관기관 정보 찾기
        try:
            body_text = driver.find_element(By.TAG_NAME, "body").text

            if (
                "주관 기관" in body_text
                or "담당기관" in body_text
                or "기관" in body_text
            ):
                rows = driver.find_elements(
                    By.CSS_SELECTOR, "table tr, .form-table tr, .info-table tr"
                )

                for row in rows:
                    row_text = row.text.strip()
                    if "주관 기관" in row_text or "담당기관" in row_text:
                        # 행 텍스트에서 직접 추출
                        parts = (
                            row_text.split("주관 기관")
                            if "주관 기관" in row_text
                            else row_text.split("담당기관")
                        )
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
        return "서울시"


def crawl_policies_from_current_page():
    """현재 페이지의 정책들을 크롤링"""
    policies = []

    try:
        # 페이지 로딩 대기
        time.sleep(3)

        # 서울시 정책 페이지에 맞는 셀렉터로 정책 리스트 찾기
        print("정책 리스트 검색 중...")

        # 먼저 정책 리스트 컨테이너를 찾기
        policy_list = []

        # 여러 셀렉터를 시도해서 정책 항목들 찾기
        selectors_to_try = [
            ".policy-list li",
            ".list li",
            "ul li",
            ".content li",
            "tbody tr",
            ".board-list li",
        ]

        for selector in selectors_to_try:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                print(f"셀렉터 '{selector}'로 {len(elements)}개 요소 발견")

                # 필터링 없이 모든 정책 가져오기
                valid_policies = []
                for elem in elements:
                    text = elem.text.strip()

                    # 기본적인 필터링만 - 텍스트가 있고 링크가 있는 것
                    if len(text) > 10 and not any(
                        exclude in text
                        for exclude in [
                            "메뉴",
                            "검색",
                            "로그인",
                            "이전",
                            "다음",
                            "처음",
                            "마지막",
                        ]
                    ):

                        # 링크가 있는지 확인
                        links = elem.find_elements(By.CSS_SELECTOR, "a")
                        if links:
                            valid_policies.append(elem)

                if valid_policies:
                    policy_list = valid_policies
                    print(f"유효한 정책 항목: {len(policy_list)}개")
                    break
                elif len(elements) >= 3:  # 3개 이상이면 사용
                    policy_list = elements
                    print(f"모든 요소 사용: {len(policy_list)}개")
                    break

        print(f"최종 발견된 정책 개수: {len(policy_list)}")

        for i, policy in enumerate(policy_list, 1):
            try:
                # 정책 전체 텍스트
                full_text = policy.text.strip()
                if not full_text:
                    continue

                # 개별 요소들 찾기
                policy_type = ""
                policy_title = ""
                policy_summary = ""
                agency = "서울시"

                # 전체 텍스트를 줄별로 분리하여 파싱
                lines = [line.strip() for line in full_text.split("\n") if line.strip()]

                if lines:
                    # 첫 번째 줄이 정책유형인지 확인
                    first_line = lines[0]
                    policy_types = [
                        "일자리",
                        "주거",
                        "교육",
                        "복지",
                        "문화",
                        "참여",
                        "권리",
                    ]

                    if any(ptype in first_line for ptype in policy_types):
                        policy_type = first_line
                        if len(lines) >= 2:
                            policy_title = lines[1]
                        if len(lines) >= 3:
                            policy_summary = " ".join(lines[2:])
                    else:
                        # 첫 번째 줄이 정책명인 경우
                        policy_title = first_line
                        if len(lines) >= 2:
                            policy_summary = " ".join(lines[1:])

                # 상세 페이지 링크 처리 - 빠르게 진행
                print(f"  정책 {i}: {policy_title}")

                # 정책 정보 저장 - 제목이 있으면 무조건 저장
                if policy_title and len(policy_title) > 2:  # 최소한의 검증만
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
    # 서울시 정책 페이지 URL
    url = "https://youth.seoul.go.kr/infoData/plcyInfo/ctList.do?plcyBizId=&tab=002&key=2309150002&sc_detailAt=&pageIndex=1&orderBy=regYmd+desc&blueWorksYn=N&tabKind=002&sw="

    print("서울시 정책 페이지 로딩 중...")
    driver.get(url)
    time.sleep(5)

    print("현재 페이지 제목:", driver.title)
    print("현재 URL:", driver.current_url)

    # 전체 정책 수집
    all_policies = []
    current_page = 1

    # CSV 파일 준비
    csv_filename = "seoul_city_policies_fixed.csv"

    print(f"\n=== 서울시 정책 크롤링 시작 ===")

    # 첫 페이지에서 총 정책 수 확인
    try:
        total_elements = driver.find_elements(By.CSS_SELECTOR, "body")
        if total_elements:
            body_text = total_elements[0].text
            total_match = re.search(r"총\s*(\d+)\s*건", body_text)
            if total_match:
                total_count = int(total_match.group(1))
                print(f"총 서울시 정책 수: {total_count}")
    except Exception as e:
        print(f"페이지 정보 확인 중 오류: {e}")

    # 연속으로 빈 페이지가 나오면 중단하기 위한 카운터
    empty_page_count = 0
    max_empty_pages = 3

    while current_page <= 100:  # 최대 100페이지까지
        print(f"\n--- 페이지 {current_page} 크롤링 중 ---")

        # 현재 페이지의 정책들 크롤링
        page_policies = crawl_policies_from_current_page()

        if not page_policies:
            empty_page_count += 1
            print(f"빈 페이지 발견 ({empty_page_count}/{max_empty_pages})")

            if empty_page_count >= max_empty_pages:
                print("연속으로 빈 페이지가 나타나서 크롤링을 종료합니다.")
                break
        else:
            empty_page_count = 0  # 정책이 있으면 카운터 리셋
            all_policies.extend(page_policies)
            print(f"페이지 {current_page}에서 {len(page_policies)}개 정책 수집")

        print(f"총 수집된 정책: {len(all_policies)}개")

        # 충분한 정책을 수집했으면 종료
        if len(all_policies) >= 350:
            print("충분한 정책을 수집했습니다.")
            break

        # 다음 페이지로 이동
        try:
            next_page_number = current_page + 1

            # URL 직접 변경으로 다음 페이지 이동
            print(f"페이지 {next_page_number}으로 이동합니다")

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
            time.sleep(3)

            # 페이지 이동 확인
            final_url = driver.current_url
            if "pageIndex=" in final_url:
                match = re.search(r"pageIndex=(\d+)", final_url)
                if match:
                    actual_page = int(match.group(1))
                    if actual_page < next_page_number:
                        print("마지막 페이지에 도달한 것 같습니다.")
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
        f"✅ 총 {len(all_policies)}개의 서울시 정책 정보를 '{csv_filename}' 파일로 저장했습니다."
    )
    print(f"크롤링한 페이지 수: {current_page}")

except Exception as e:
    print(f"전체 프로세스 중 오류 발생: {e}")

finally:
    driver.quit()

print("\n서울시 정책 크롤링 완료!")
