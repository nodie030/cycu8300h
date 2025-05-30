import os
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
from flask import Flask, request, abort
from dotenv import load_dotenv
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError

load_dotenv()
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# 讀取資料
qa_df = pd.read_csv("通通夠QA資料庫.csv")
qa_chunks = [f"Q: {row['question']}\nA: {row['answer']}" for _, row in qa_df.iterrows()]

event_df = pd.read_csv("cycu_tag_activities.csv")
event_chunks = [
    f"活動標題：{row['標題']}\n時間：{row['時間']}\n標籤：{row['標籤']}\n說明：{row['活動描述']}"
    for _, row in event_df.iterrows()
]

teacher_df = pd.read_csv("cycu_teachers.csv")
teacher_chunks = [
    f"教師姓名：{row['姓名']}\n職稱：{row['職稱']}\n辦公處：{row['辦公處']}\n電話：{row['電話']}\nE-mail：{row['E-mail']}"
    for _, row in teacher_df.iterrows()
]

course_df = pd.read_csv("cycu_teacher_courses.csv")
course_chunks = [
    # f"課程名稱：{row['course_name']}\n授課教師：{row['teacher']}\n時間：{row['time']}\n地點：{row['location']}"
    f"課程名稱：{row['課程名稱']}\n授課教師：{row['教師']}\n課程代碼：{row['課程代碼']}"
    for _, row in course_df.iterrows()
]

# 合併所有段落文字
all_chunks = qa_chunks + event_chunks + teacher_chunks + course_chunks

# 建立 embedding
embeddings = [client.embeddings.create(model="text-embedding-ada-002", input=chunk).data[0].embedding for chunk in all_chunks]
embedding_array = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(len(embedding_array[0]))
index.add(embedding_array)
chunk_lookup = all_chunks
user_histories = {}

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_input = event.message.text
    user_id = event.source.user_id
    history = user_histories.get(user_id, [])

    # 搜尋相關語意內容
    user_embedding = client.embeddings.create(model="text-embedding-ada-002", input=user_input).data[0].embedding
    D, I = index.search(np.array([user_embedding], dtype="float32"), k=5)
    related_info = "\n---\n".join([chunk_lookup[i] for i in I[0]])

    # 給 GPT 的提示內容
    messages = [
        {"role": "system", "content": (
            "你是中原大學通識中心的 GPT 助手【通通夠】🎓，請根據下列提供的活動、課程、教師與 QA 資料回答問題，"
            "回答需語意理解、整合內容，不可虛構。若資料不足請回應：『這部分我不太清楚，請洽通識中心人員。電話：03-265-6853』。"
            "請使用溫柔、活潑、貼近學生的語氣，多插入 emoji。"
            f"相關資料如下：\n{related_info}"
        )},
        *history[-6:],
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )

    reply = response.choices[0].message.content.strip()
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    user_histories[user_id] = history[-10:]

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(port=5000)