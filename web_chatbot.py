#fg
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ✅ 頁面設定
st.set_page_config(page_title="客服 test", page_icon="💬")
st.markdown("<h1 style='font-size:30px; color:#F63366;'>客服 test 💬</h1>", unsafe_allow_html=True)
st.markdown("您好，有任何問題都可以問我喔！")

# ✅ 初始化對話紀錄
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ✅ 清除對話按鈕
if st.button("🗑️ 清除對話紀錄"):
    st.session_state["chat_history"] = []
    st.experimental_rerun()

# ✅ API Key 設定
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ 請先設定你的 OpenRouter API Key（環境變數或 secrets）")
    st.stop()

# ✅ 初始化嵌入模型（強制 CPU 避免錯誤）
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

# ✅ 向量資料庫初始化
persist_path = "faiss_index"
if os.path.exists(persist_path):
    vectordb = FAISS.load_local(persist_path, embedding, allow_dangerous_deserialization=True)
else:
    texts = [
        "全聯每週三會員日有指定商品折扣。",
        "加入 PX Pay 可享更多全聯點數回饋。",
        "中元節有生鮮特賣活動，請關注官方公告。",
    ]
    vectordb = FAISS.from_texts(texts, embedding)
    vectordb.save_local(persist_path)

# ✅ 查詢 Claude 並保留多輪對話
def query_with_rag_claude(user_query: str, api_key: str, model="anthropic/claude-3-haiku") -> str:
    docs = vectordb.similarity_search(user_query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    system_msg = {
        "role": "system",
        "content": f"""你是一位全聯客服人員。請根據以下內部資料回答顧客問題：

【內部資料】
{context}

請用親切、簡潔的方式回答。
"""}

    # 將系統設定 + 多輪歷史 + 新問題組成完整對話
    messages = [system_msg] + st.session_state["chat_history"] + [
        {"role": "user", "content": user_query}
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages
    }

    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()["choices"][0]["message"]["content"]
    else:
        return "❌ 回覆錯誤：" + res.text

# ✅ 使用者輸入區
user_input = st.text_input("請輸入您的問題：", key="user_input")

if user_input:
    with st.spinner("🤔 正在查找資料與撰寫回覆中..."):
        reply = query_with_rag_claude(user_input, OPENROUTER_API_KEY)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})

# ✅ 顯示對話紀錄
st.divider()
st.markdown("### 💬 對話紀錄")
for msg in st.session_state["chat_history"]:
    if msg["role"] == "user":
        st.markdown(f"🧑‍💼 **你**：{msg['content']}")
    elif msg["role"] == "assistant":
        st.markdown(f"🤖 **客服**：{msg['content']}")

