#fg
import os
import streamlit as st
import requests

# ✅ 強制 Streamlit watcher 關閉（避免 torch crash）
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ✅ 頁面設定
st.set_page_config(page_title="客服test", page_icon="💬")
st.markdown("<h1 style='font-size:30px; color:#F63366;'>客服test 💬</h1>", unsafe_allow_html=True)
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
    "你是誰": "我是全聯線上文字客服，隨時為您服務。",
}

# ✅ 載入 FAISS 與 Embedding 模型
try:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError as e:
    st.error(f"❌ 模組載入失敗：{e}")
    st.stop()

try:
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# 向量資料庫初始化（僅讀取既有的 faiss_index）
persist_path = "faiss_index"
if os.path.exists(persist_path):
    try:
        vectordb = FAISS.load_local(persist_path, embedding, allow_dangerous_deserialization=True)
        print("✅ 向量庫載入成功，資料筆數：", vectordb.index.ntotal)
    except Exception as e:
        st.error(f"❌ 載入向量庫失敗：{e}")
        st.stop()
else:
    st.error("❌ 未找到向量資料庫，請先執行爬蟲並建立 faiss_index/")
    st.stop()

# ✅ Claude 回答函式（RAG）
def query_with_rag_claude(query: str, api_key: str, model="anthropic/claude-3-haiku") -> str:
    #docs = vectordb.similarity_search(query, k=5)
    #context = "\n".join([doc.page_content for doc in docs])
    docs = vectordb.similarity_search(query, k=5)
    st.write("🔍 系統查得相近資料：")
    for doc in docs:
    	st.markdown(f"- `{doc.page_content}`")
context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""你是一位全聯的客服人員。請根據以下資料回答顧客的問題：

【內部資料】
{context}

【顧客問題】
{query}

請用親切的方式回答。
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

# ✅ 多輪對話初始化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ✅ 使用者輸入
user_input = st.chat_input("請輸入您的問題：")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 回傳 FAQ 或 Claude 回覆
    if user_input in faq_responses:
        reply = faq_responses[user_input]
    else:
        with st.spinner("🧠 正在查找資料與撰寫回覆中..."):
            reply = query_with_rag_claude(user_input, OPENROUTER_API_KEY)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ✅ 顯示對話歷史
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

