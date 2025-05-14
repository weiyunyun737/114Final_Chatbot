import os
import streamlit as st
import requests
import json

# ✅ UI設定
st.set_page_config(page_title="客服小幫手", page_icon="💬")

st.markdown(
  "<h1 style='font-size:30px; color:#F63366;'>客服機器人小幫手 💬</h1>",
  unsafe_allow_html=True
)
st.markdown("您好，有任何問題都可以問我喔！")


# ✅ FAQ 快取資料
faq_responses = {
    "如何聯絡客服": "您可以透過 support@example.com 聯絡我們的客服團隊。",
    "營業時間": "我們的客服營業時間為週一至週五 09:00 至 18:00。",
    "你是誰": "我是 OpenRouter 聊天客服機器人，隨時為您提供協助！"
}

# ✅ 取得 API Key，並提醒未設定
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
if not OPENROUTER_API_KEY:
    st.error("❌ 請先設定你的 OpenRouter API Key（環境變數或 secrets）")
    st.stop()

MODEL_NAME = "anthropic/claude-3-haiku"

# ✅ 初始化聊天歷史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ 輸入欄位
user_input = st.chat_input("請輸入您的問題...")

# ✅ 串流回覆函式
def ask_openrouter_stream(user_input):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://example.com",
        "X-Title": "streamlit-chatbot"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": "你是一位友善又專業的中文客服助手，請使用繁體中文回答問題。"}] + st.session_state["messages"],
        "stream": True
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        response.raise_for_status()

        full_reply = ""
        for line in response.iter_lines(decode_unicode=False):
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                    if "content" in delta:
                        full_reply += delta["content"]
                        yield delta["content"]
                except Exception as e:
                    print("解析錯誤：", e)
                    continue
    except Exception as e:
        yield f"⚠️ 錯誤：{str(e)}"

# ✅ 顯示對話歷史
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ✅ 處理輸入訊息
if user_input and user_input.strip():
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ✅ FAQ 快取命中檢查
    faq_hit = next((v for k, v in faq_responses.items() if k in user_input), None)

    with st.chat_message("assistant"):
        if faq_hit:
            st.markdown(faq_hit)
            st.session_state["messages"].append({"role": "assistant", "content": faq_hit})
        else:
            reply = ""
            reply_container = st.empty()
            for chunk in ask_openrouter_stream(user_input):
                reply += chunk
                reply_container.markdown(reply + "▌")
            reply_container.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
