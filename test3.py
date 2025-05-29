from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tempfile, shutil, time

# ✅ 設定路徑
FIREFOX_PATH = "/home/vivian/firefox/firefox"
GECKO_PATH = "/home/vivian/bin/geckodriver"

profile_path = tempfile.mkdtemp()
profile = FirefoxProfile(profile_path)

options = Options()
options.binary_location = FIREFOX_PATH
options.profile = profile.path
options.add_argument("--headless")  # 可取消註解以背景執行

service = Service(executable_path=GECKO_PATH)
driver = webdriver.Firefox(service=service, options=options)

try:
    # ✅ 開啟全聯“生活誌”
    url = "https://www.pxmart.com.tw/campaign/life-will"
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.campaign-tab__wrap"))
    )

    # ✅ 抓分類按鈕
    category_buttons = driver.find_elements(By.CSS_SELECTOR, "div.campaign-tab__wrap button")
    category_map = {}

    for btn in category_buttons:
        category_name = btn.text.strip()
        category_map[category_name] = btn

    print(f"✅ 共找到 {len(category_map)} 個分類：", list(category_map.keys()))

    # ✅ 一一點擊分類 → 擷取商品資訊
    for category_name, button in category_map.items():
        try:
            print(f"\n📂 分類：{category_name}")
            driver.execute_script("arguments[0].click();", button)
            time.sleep(2)  # ⏳ 給 JavaScript 時間載入資料

            # 等待商品載入
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//h5[contains(@class, 'Card_card-title')]"))
            )

            # 擷取所有商品名稱與價格
            titles = driver.find_elements(By.XPATH, "//h5[contains(@class, 'Card_card-title')]")
            prices = driver.find_elements(By.XPATH, "//p[contains(@class, 'Card_card-productPrice')]")

            for i in range(min(len(titles), len(prices))):
                name = titles[i].text.strip()
                price = prices[i].text.strip()
                print(f"  {i+1}. 📦 {name} - 💰 {price}")

        except Exception as e:
            print(f"⚠️ 分類「{category_name}」擷取失敗：", e)

except Exception as e:
    print("❌ 程式整體錯誤：", e)

finally:
    driver.quit()
    shutil.rmtree(profile_path)
