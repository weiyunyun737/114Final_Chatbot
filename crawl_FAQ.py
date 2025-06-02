from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os, shutil, tempfile, time

# ========== Selenium 設定 ==========
FIREFOX_PATH = "/home/vivian/firefox/firefox"
GECKO_PATH = "/home/vivian/bin/geckodriver"
profile_path = tempfile.mkdtemp()
options = Options()
options.binary_location = FIREFOX_PATH
options.add_argument("--headless")
service = Service(executable_path=GECKO_PATH)
driver = webdriver.Firefox(service=service, options=options)

# ========== 頁面開啟 ==========
driver.get("https://www.pxmart.com.tw/customer-service/faq/%E5%85%A8%E6%94%AF%E4%BB%98%E7%9B%B8%E9%97%9C%E5%95%8F%E9%A1%8C")

# ========== 初始化 ==========
texts, metadatas = [], []

try:
    # 抓所有 tab 分類
    def get_tab_buttons():
        return driver.find_elements(By.CSS_SELECTOR, "a[class*='Tab_tab_']")

    tab_buttons_initial = get_tab_buttons()
    print(f"✅ 初次偵測到 {len(tab_buttons_initial)} 個分類")

    for i in range(len(tab_buttons_initial)):
        try:
            # 每次迭代都重新獲取 tab_buttons
            tab_buttons = get_tab_buttons()
            if i >= len(tab_buttons):
                print(f"⚠️ 找不到第 {i+1} 個分類，跳過。")
                continue

            tab = tab_buttons[i]
            category = tab.text.strip()
            print(f"\n 開始擷取分類：{category}")

            driver.execute_script("arguments[0].click();", tab)
            time.sleep(1)

            # 等待 FAQ 出現（用模糊抓法）
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "summary[class*='Accordion']"))
            )

            def get_faq_blocks():
                return driver.find_elements(By.CSS_SELECTOR, "details[class*='Accordion']")

            faq_blocks = get_faq_blocks()
            print(f"✅ 共找到 {len(faq_blocks)} 則問題")

            for block_index in range(len(faq_blocks)):
                try:
                    # 每次迭代都重新獲取 faq_blocks
                    faq_blocks_reloaded = get_faq_blocks()
                    if block_index >= len(faq_blocks_reloaded):
                        print(f"⚠️ 找不到第 {block_index+1} 個 FAQ，跳過。")
                        continue
                    block = faq_blocks_reloaded[block_index]

                    summary = block.find_element(By.CSS_SELECTOR, "summary")
                    driver.execute_script("arguments[0].click();", summary)
                    time.sleep(0.3)

                    q = summary.text.strip()
                    a = block.find_element(By.CSS_SELECTOR, "div[class*='Editor']").text.strip()

                    full_text = f"Q: {q}\nA: {a}"
                    texts.append(full_text)
                    metadatas.append({"type": "faq", "category": category})

                except Exception as e_inner:
                    print(f"⚠️ 單則 FAQ 擷取失敗：{str(e_inner)}")

        except Exception as e:
            print(f"❌ 分類擷取錯誤（第 {i+1} 個）：{str(e)}")

except Exception as e:
    print("❌ 整體錯誤：", str(e))

finally:
    driver.quit()
    shutil.rmtree(profile_path)

# ========== 建立向量 ==========
if texts:
    print("\n易 建立向量庫中...")
    embedding = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"}
    )
    db = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metadatas)
    db.save_local("faiss_index")
    print(f"✅ 已建立 {len(texts)} 筆 FAQ 向量資料並儲存至 faiss_index/")
else:
    print("⚠️ 無 FAQ 資料，向量庫未建立")
