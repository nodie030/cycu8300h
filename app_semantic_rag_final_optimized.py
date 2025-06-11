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
from docx import Document

load_dotenv()
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app = Flask(__name__)

# 工具：將長文字切段（每段300字）
def split_text(text, max_length=300):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

# 讀取 QA
qa_df = pd.read_csv("通通夠QA資料庫.csv")
qa_chunks = []
for _, row in qa_df.iterrows():
    content = f"Q: {row['question']}\nA: {row['answer']}"
    qa_chunks.extend(split_text(content))

# 讀取活動
event_df = pd.read_csv("cycu_tag_activities.csv")
event_chunks = []
for _, row in event_df.iterrows():
    content = f"活動標題：{row['標題']}\n時間：{row['時間']}\n標籤：{row['標籤']}\n說明：{row['活動描述']}"
    event_chunks.extend(split_text(content))

# 讀取教師資料
teacher_df = pd.read_csv("cycu_teachers.csv")
teacher_chunks = []
for _, row in teacher_df.iterrows():
    content = f"教師姓名：{row['姓名']}\n職稱：{row['職稱']}\n辦公處：{row['辦公處']}\n電話：{row['電話']}\nE-mail：{row['E-mail']}"
    teacher_chunks.extend(split_text(content))

# 讀取課程資料
course_df = pd.read_csv("cycu_teacher_courses.csv")
course_chunks = []
grouped = course_df.groupby("教師")["課程名稱"].apply(list)
for 教師, 課程名稱 in grouped.items():
    content = f"教師姓名：{教師}\n授課課程：{', '.join(課程名稱)}"
    course_chunks.extend(split_text(content))

# 讀取並切段 .docx 檔案
docx_texts = []
docx_folder = "./"
for file in os.listdir(docx_folder):
    if file.endswith(".docx"):
        doc = Document(os.path.join(docx_folder, file))
        full_text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
        docx_texts.extend(split_text(f"【文件：{file}】\n{full_text}"))

# 整合所有資料來源
all_chunks = qa_chunks + event_chunks + teacher_chunks + course_chunks + docx_texts

# 建立向量索引
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

    # 語意檢索最相近的 10 段，並限制總字數
    user_embedding = client.embeddings.create(model="text-embedding-ada-002", input=user_input).data[0].embedding
    D, I = index.search(np.array([user_embedding], dtype="float32"), k=10)
    
    selected_chunks = []
    total_length = 0
    for idx in I[0]:
        chunk = chunk_lookup[idx]
        if total_length + len(chunk) <= 1500:  # 限制總長度 1500 字
            selected_chunks.append(chunk)
            total_length += len(chunk)
        else:
            break

    related_info = "\n---\n".join(selected_chunks)

    # 提示給 GPT
    messages = [
        {"role": "system", "content": (
            "你是中原大學通識中心的 GPT 助手【通通夠】🎓，請根據下列提供的活動、課程、教師、QA 與文件內容回答問題，"
            "回答需語意理解、整合內容，不可自行猜測或編造。若資料不足請回應：『這部分我不太清楚，請洽通識中心人員。電話：03-265-6853』。"
            "請使用溫柔、活潑、貼近學生的語氣，盡量多插入 emoji。"
            f"相關資料如下：\n{related_info}"
        )},
        *history[-6:],
        {"role": "user", "content": user_input}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    reply = response.choices[0].message.content.strip()
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": reply})
    user_histories[user_id] = history[-10:]

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

if __name__ == "__main__":
    app.run(port=5000)