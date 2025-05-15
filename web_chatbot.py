import os
import streamlit as st
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


# ✅ Streamlit 頁面設定
st.set_page_config(page_title="客服小幫手", page_icon="💬")
st.markdown("<h1 style='font-size:30px; color:#F63366;'>客服機器人小幫手 💬</h1>", unsafe_allow_html=True)
st.markdown("您好，有任何問題都可以問我喔！")

# ✅ API Key 設定
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ 請先設定你的 OpenRouter API Key（環境變數或 secrets）")
    st.stop()

# ✅ FAQ 快取資料
faq_responses = {
    "如何聯絡客服": "您可以透過 support@example.com 聯絡我們的客服團隊。",
    "營業時間": "我們的客服營業時間為週一至週五 09:00 至 18:00。",
    "你是誰": "我是 OpenRouter 聊天客服機器人，隨時為您服務。",
}

# ✅ 初始化嵌入模型與向量資料庫（模型下載到local）
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectordb = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

# ✅ 初始模擬資料（只執行一次）
if len(vectordb.get()['documents']) == 0:
    texts = [
        "全聯每週三會員日有指定商品折扣。",
        "加入 PX Pay 可享更多全聯點數回饋。",
        "中元節有生鮮特賣活動，請關注官方公告。",
    ]
    metadatas = [{"source": "活動1"}, {"source": "活動2"}, {"source": "活動3"}]
    vectordb.add_texts(texts, metadatas=metadatas)

# ✅ 定義語意查詢 + Claude 回覆的函式
def query_with_rag_claude(query: str, api_key: str, model="anthropic/claude-3-haiku") -> str:
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""你是一位全聯的客服人員。請根據以下資料回答顧客的問題：

【內部資料】
{context}

【顧客問題】
{query}

請用親切、簡潔的方式回答。
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}]
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return "❌ 錯誤：" + res.text

# ✅ 使用者輸入區
user_input = st.text_input("請輸入您的問題：", key="user_input")

if user_input:
    # 先查 FAQ 快取
    if user_input in faq_responses:
        st.success(f"🤖：{faq_responses[user_input]}")
    else:
        with st.spinner("🤔 正在查找資料與撰寫回覆中..."):
            reply = query_with_rag_claude(user_input, OPENROUTER_API_KEY)
            st.success(f"🤖：{reply}")
