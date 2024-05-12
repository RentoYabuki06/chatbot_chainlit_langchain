import json
import ast
import chainlit as cl
import os
from openai import AsyncOpenAI

from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# 環境変数ファイルをロード
load_dotenv()

# 環境変数から API キーを取得
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# 初回起動時に呼び出されてチャットが開始する
@cl.on_chat_start #デコレータ
async def on_chat_start(): # 非同期関数
    model = ChatOpenAI(streaming=True)
    # システム用のメッセージプロンプトテンプレートの準備
    template="You're an helpful assistant. You have to answer the question."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    #人間用のメッセージプロンプトテンプレートの準備
    human_template="{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # プロンプトテンプレートの準備
    prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    # コンポーネントの連結
    message_history = prompt | model | StrOutputParser()
    # 作成したオブジェクトをセッション変数"Conversation history"に保存する
    cl.user_session.set("message_history", message_history)


# ユーザーからのメッセージを受け取ったときに呼び出される関数
@cl.on_message
async def on_message(message: cl.Message): # 非同期関数  cl.Messageタイプのmessageを受け取る
    # 会話履歴を取り出す
    message_history = cl.user_session.get("message_history")  
    # 応答に利用するための空文字列を生成する
    msg = cl.Message(content="")


    async for chunk in message_history.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
    # 部分的なテキストデータをメッセージに追加する
        await msg.stream_token(chunk)
    # 完全な解答をユーザーに送信する
    await msg.send()