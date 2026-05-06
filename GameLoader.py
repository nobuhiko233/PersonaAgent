import os
import sys
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

MAX_GAMES_TO_LOAD = 1000

def main():
    print("1. 正在加载本地 Embedding 模型 (已强行切断网络连接)...")
    model_name_or_path = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-large-zh-v1.5")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name_or_path, 
        model_kwargs={"local_files_only": True}
    )

    print("2. 正在读取 steam_games.csv 数据...")
    loader = CSVLoader(file_path="steam_games.csv", encoding="utf-8")
    docs = loader.load()

    docs = docs[:MAX_GAMES_TO_LOAD]
    print(f"-> 为保证速度，已截取前 {len(docs)} 条游戏数据。")

    print(f"3. 正在构建本地 Chroma 向量数据库 (处理 {len(docs)} 条，请稍候)...")
    Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory="./steam_chroma_db"
    )

    print("✅ 游戏数据向量化并保存成功！")

if __name__ == "__main__":
    main()