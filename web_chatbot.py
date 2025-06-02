import os
import streamlit as st
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ 強制 Streamlit watcher 關閉（避免 torch crash）
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# ✅ 頁面設定
st.set_page_config(page_title="全聯客服", page_icon=":)")
st.markdown("<h1 style='font-size:30px; color:#F63366;'>全聯客服 </h1>", unsafe_allow_html=True)
st.markdown("您好，有任何問題都可以問我喔！")

# ✅ API Key 設定
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("❌ 請先設定你的 OpenRouter API Key（環境變數或 secrets）")
    st.stop()

# ✅ FAQ 快取資料
faq_responses = {
    "如何聯絡客服": "您可以透過 support@pxmart.com.tw 聯絡我們的客服團隊。",
    "營業時間": "我們的客服營業時間為週一至週日 09:00 至 22:00。",
    "你是誰": "我是全聯線上文字客服，隨時為您服務。",
}

# ✅ 載入 FAISS 與 Embedding 模型
try:
    embedding = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cpu"}
    )
except ImportError as e:
    st.error(f"❌ 模組載入失敗：{e}")
    st.stop()
except Exception as e:
    st.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# ✅ 載入 FAISS 向量庫 (FAQ)
persist_path_faq = "faiss_index_faq"
vectordb_faq = None
if os.path.exists(persist_path_faq):
    try:
        vectordb_faq = FAISS.load_local(persist_path_faq, embedding, allow_dangerous_deserialization=True)
        st.success(f"✅ FAQ 向量庫載入成功，資料筆數：{vectordb_faq.index.ntotal if vectordb_faq else 0}")
    except Exception as e:
        st.error(f"❌ 載入 FAQ 向量庫失敗：{e}")
else:
    st.warning(f"⚠️ 未找到 FAQ 向量資料庫：{persist_path_faq}/")

# ✅ 載入 FAISS 向量庫 (Products)
persist_path_product = "faiss_index_product"
vectordb_product = None
if os.path.exists(persist_path_product):
    try:
        vectordb_product = FAISS.load_local(persist_path_product, embedding, allow_dangerous_deserialization=True)
        st.success(f"✅ 商品向量庫載入成功，資料筆數：{vectordb_product.index.ntotal if vectordb_product else 0}")
    except Exception as e:
        st.error(f"❌ 載入商品向量庫失敗：{e}")
else:
    st.warning(f"⚠️ 未找到商品向量資料庫：{persist_path_product}/")

# ✅ Claude 回答函式（RAG，同時查詢兩個向量庫）
def query_with_rag_claude(query: str, api_key: str, model="anthropic/claude-3-haiku") -> str:
    combined_docs = []
    k = 10  # 您可以調整每個向量庫的檢索數量

    if vectordb_faq:
        docs_faq = vectordb_faq.similarity_search(query, k=k)
        combined_docs.extend(docs_faq)

    if vectordb_product:
        docs_product = vectordb_product.similarity_search(query, k=k)
        combined_docs.extend(docs_product)

    with st.expander("系統查得相近資料："):
    	for doc in combined_docs:
            source_type = doc.metadata.get('type', 'unknown')
            st.markdown(f"- `{doc.page_content}` (來源: `{source_type}`) ")

    context = "\n".join([doc.page_content for doc in combined_docs])

    prompt = f"""你是一位全聯客服人員。請只根據下列內部資料回答問題，並清楚列出找到的相關資訊來源（常見問題或商品資訊）。若資料中沒有，請說明無法查得，但不要假設。

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
        with st.spinner("易 正在查找資料與撰寫回覆中..."):
            reply = query_with_rag_claude(user_input, OPENROUTER_API_KEY)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ✅ 顯示對話歷史
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
