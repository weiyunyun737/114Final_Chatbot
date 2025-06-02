from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import tempfile
import shutil
import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 設定 ===
FIREFOX_PATH = "/home/vivian/firefox/firefox"
GECKO_PATH = "/home/vivian/bin/geckodriver"

# === Headless 模式 + 暫存 Profile 設定 ===
profile_path = tempfile.mkdtemp()
options = Options()
options.binary_location = FIREFOX_PATH
options.add_argument("--headless")

service = Service(executable_path=GECKO_PATH)
driver = webdriver.Firefox(service=service, options=options)

texts = []
metadatas = []

try:
    driver.get("https://www.pxmart.com.tw/campaign/life-will")

    # 等待分類按鈕出現
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//a[contains(@class, 'Button_button')]"))
    )

    category_links = driver.find_elements(By.XPATH, "//a[contains(@class, 'Button_button')]")
    category_map = {}
    for link in category_links:
        name = link.text.strip()
        href = link.get_attribute("href")
        if name and href:
            category_map[name] = href

    print(f"✅ 找到 {len(category_map)} 個分類：", list(category_map.keys()))

    for category_name, category_url in category_map.items():
        print(f"\n🔎 分類：{category_name}")
        driver.get(category_url)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//h5[contains(@class, 'Card_card-title')]"))
            )
        except:
            print("⚠️ 無商品資料")
            continue

        titles = driver.find_elements(By.XPATH, "//h5[contains(@class, 'Card_card-title')]")
        prices = driver.find_elements(By.XPATH, "//p[contains(@class, 'Card_card-productPrice')]")

        for i in range(min(len(titles), len(prices))):
            name = titles[i].text.strip()
            price = prices[i].text.strip()
            full_text = f"商品名稱：{name}，價格：{price}，分類：{category_name}"
            texts.append(full_text)
            metadatas.append({
            	"type": "product",
            	"category": category_name
            })
            
except Exception as e:
    print("❌ 發生錯誤：", e)

finally:
    driver.quit()
    shutil.rmtree(profile_path)

# === 向量建立 ===
print("\n🧠 建立向量庫中...")
embedding = HuggingFaceEmbeddings(
    #model_name="sentence-transformers/all-mpnet-base-v2",
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"device": "cpu"}  # 可改為 "cuda" 如使用 GPU
)

db = FAISS.from_texts(texts=texts, embedding=embedding, metadatas=metadatas)
db.save_local("faiss_index")
print("✅ 向量庫已儲存至 faiss_index/")
print("✅ 向量庫筆數：", db.index.ntotal)
'''
docs = db.similarity_search("洗髮", k=20)
for i, doc in enumerate(docs):
    print(f"{i+1}.", doc.page_content)
'''
