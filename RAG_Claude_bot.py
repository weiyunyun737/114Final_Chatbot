from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
import os

# === key  ===
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
API_KEY = "你的 OpenRouter API Key"
MODEL = "anthropic/claude-3-sonnet"  # 也可改成 claude-3-haiku

# === 建立嵌入模型與資料庫 ===
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

# === 模擬一些資料 ===
texts = [
    "全聯每週三會員日有指定商品折扣。",
    "加入 PX Pay 可享更多全聯點數回饋。",
    "中元節有生鮮特賣活動，請關注官方公告。",
]
metadatas = [{"source": "活動1"}, {"source": "活動2"}, {"source": "活動3"}]

# 僅第一次執行才加入資料
if len(vectordb.get()['documents']) == 0:
    vectordb.add_texts(texts, metadatas=metadatas)
    print("✅ 已加入初始資料")

# === 使用者輸入 ===
query = input("請輸入你的問題：\n> ")

# === 查詢語意相似內容 ===
docs = vectordb.similarity_search(query, k=3)
context = "\n".join([doc.page_content for doc in docs])

# === 組合 prompt 丟給 Claude ===
prompt = f"""你是一位全聯的客服人員。請根據以下內部資料回答顧客的問題：

【客服資料】
{context}

【顧客問題】
{query}

請用簡潔、親切的語氣回答。
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)

if response.status_code == 200:
    print("\n🤖 Claude 回覆：\n")
    print(response.json()["choices"][0]["message"]["content"])
else:
    print("❌ 回覆錯誤：", response.text)
