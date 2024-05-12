import os # osとやりとりするAPI, ここでは環境変数にopenAIのAPIキーを設定するため
import shutil # 高レベルのファイルやディレクトリの操作。ここでは一時DBの削除に使用
# from PyPDF2 import PdfReader 
from openai import AsyncOpenAI

from langchain.chat_models import ChatOpenAI # チャット用のAIモデル
from langchain.prompts import ChatPromptTemplate # プロンプトのテンプレート
from langchain.schema import StrOutputParser #文字出力パーサー
# Runnable（実行可能オブジェクト）とその設定に関するクラスをインポートします。
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
# 環境変数の設定
from dotenv import load_dotenv
# chainlit（チャットボット作成用フレームワーク）をclとしてインポートします。
import chainlit as cl

TEMP_PDF_PATH = "./doc/doc.pdf"

# 環境変数を`.env`ファイルから読み込む
load_dotenv()

# 環境変数からAPIキーを取得
api_key = os.getenv('OPENAI_API_KEY')
client = AsyncOpenAI(api_key=api_key)

# PDFファイルを開いてテキストを抽出するための非同期関数
async def process_pdf(file):
    file = file[0] if isinstance(file, list) else file
    with open(TEMP_PDF_PATH, 'wb') as f:
        f.write(file.content)
    reader = PdfReader(TEMP_PDF_PATH)
    return ''.join(page.extract_text() for page in reader.pages)

# チャットセッションが開始されたときに呼び出される関数を定義します。このデコレータが関数をイベントハンドラとして登録します。
@cl.on_chat_start
async def on_chat_start():
    # ChatOpenAIモデルをストリーミングモードでインスタンス化します。
    model = ChatOpenAI(streaming=True)
    # チャットプロンプトテンプレートを作成し、AIの役割を定義します。
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Please provide appropriate responses to the questions.",
            ),
            ("human", "{question}"),
        ]
    )
    # プロンプト、モデル、出力パーサーをつなげて実行可能なオブジェクトを作成します。
    runnable = prompt | model | StrOutputParser()
    # この実行可能オブジェクトをユーザーセッションに格納し、後で再利用できるようにします。
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    # ユーザーセッションから実行可能オブジェクトを取得します。
    runnable = cl.user_session.get("runnable")

    # メッセージオブジェクトを作成し、初期コンテンツを空に設定します。
    msg = cl.Message(content="")

    # 取得した実行可能オブジェクトを非同期に実行し、メッセージの内容を質問としてAIに渡します。
    async for chunk in runnable.astream(
        {"question": message.content},
        # 実行時の設定として、Langchainのコールバックハンドラを設定します。
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        # AIからの応答チャンクをメッセージオブジェクトに追加します。
        await msg.stream_token(chunk)

    # 構築したメッセージをユーザーに送信します。
    await msg.send()