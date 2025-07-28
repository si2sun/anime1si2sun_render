import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import json
import unicodedata
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
import cProfile
import pstats
import time
import traceback
import logging # 新增日誌模組

# 導入 Firestore 相關模組
from google.cloud import firestore

# 從同級目錄導入 情感top3提出_dandadan_fast 模組
try:
    from 情感top3提出_dandadan_fast_json import get_top3_emotions_fast
except ImportError:
    # 使用 logging 替代 print
    logging.error("ERROR: 無法導入 '情感top3提出_dandadan_fast_json' 模組。請確保該檔案存在且在可被Python找到的路徑上。")
    sys.exit(1)

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# ====== CORS 配置 (保持不變) ======
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:5000",
    # 部署到 Render 後，需要新增你的 Render 服務網址
    # 例如: "https://your-service-name.onrender.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ====== PostgreSQL 資料庫配置 ======
# 建議從環境變數讀取資料庫憑證
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    except psycopg2.Error as e:
        logging.error(f"資料庫連線錯誤: {e}") # 使用 logging 替代 print
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"無法連接到資料庫或資料庫操作失敗: {e}")
    finally:
        if conn:
            conn.close()

# 變數
AVAILABLE_ANIME_NAMES = []
YOUTUBE_ANIME_EPISODE_URLS = {}
BAHAMUT_ANIME_EPISODE_URLS = {}
ANIME_COVER_IMAGE_URLS = {}
ANIME_TAGS_DB = {}
TAG_COMBINATION_MAPPING = {}
EMOTION_CATEGORY_MAPPING = {}

# Firestore 客戶端變數
db = None

# 建議透過環境變數 GOOGLE_APPLICATION_CREDENTIALS 或 GOOGLE_CLOUD_PROJECT 來初始化 Firestore
# 或將服務帳號 JSON 內容存入環境變數 GOOGLE_APPLICATION_CREDENTIALS_JSON
# 如果檔案存在且路徑正確，可以這樣設定
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "animetext-anime1si2sun.json" 
# 更好的方式是將 JSON 內容讀取進來，或者讓 Firestore Client 自動查找
# 如果 'animetext-anime1si2sun.json' 在應用程式根目錄，並在 Dockerfile 中 COPY 進去，這樣寫可能可以。
# 但更穩健的方案是使用 GOOGLE_APPLICATION_CREDENTIALS_JSON 環境變數 (下面會在 startup_event 處理)

def load_anime_data_from_db():
    print("\n--- 開始從 PostgreSQL 載入動漫數據 ---")
    start_time = time.time()
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    
    # 彻底移除 "ED開始秒數" 的查询
    query = 'SELECT "作品名", "集數", "巴哈動畫瘋網址", "YT網址", "封面圖", "作品分類" FROM anime_url ORDER BY "作品名", "集數";'
    
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("⚠️ 警告：資料庫的 'anime_url' 表中沒有找到任何數據。")
        return

    for row in rows:
        anime_original, episode, bahamut_url, youtube_url, cover_image_val, tags_json = row
        
        anime_normalized = unicodedata.normalize('NFC', str(anime_original).strip())
        AVAILABLE_ANIME_NAMES.append(anime_normalized)
        
        # ====== 修改這裡 ======
        ep_key_raw = episode
        ep_key = ""
        if ep_key_raw is not None:
            try:
                # 嘗試將其轉換為整數，然後再轉為字串
                ep_key = str(int(float(ep_key_raw))).strip()
            except (ValueError, TypeError):
                # 如果轉換失敗，保留原始字串形式
                ep_key = str(ep_key_raw).strip()
        # ====================

        YOUTUBE_ANIME_EPISODE_URLS.setdefault(anime_normalized, {})
        BAHAMUT_ANIME_EPISODE_URLS.setdefault(anime_normalized, {})
        ANIME_TAGS_DB.setdefault(anime_normalized, [])

        if youtube_url:
            yt_url_str = str(youtube_url).strip()
            video_id = None
            if "youtube.com/watch?v=" in yt_url_str: video_id = yt_url_str.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in yt_url_str: video_id = yt_url_str.split("youtu.be/")[-1].split("?")[0]
            if video_id: YOUTUBE_ANIME_EPISODE_URLS[anime_normalized][ep_key] = video_id
        
        if bahamut_url: BAHAMUT_ANIME_EPISODE_URLS[anime_normalized][ep_key] = str(bahamut_url).strip()
        if cover_image_val: ANIME_COVER_IMAGE_URLS[anime_normalized] = str(cover_image_val).strip()
        if tags_json:
            try:
                tags = json.loads(str(tags_json).replace("'", '"'))
                if isinstance(tags, list): ANIME_TAGS_DB[anime_normalized] = tags
            except (json.JSONDecodeError, TypeError): pass

    AVAILABLE_ANIME_NAMES = sorted(list(set(AVAILABLE_ANIME_NAMES)))
    print(f"--- PostgreSQL 數據載入完成，耗時 {time.time() - start_time:.2f} 秒 ---")

@app.on_event("startup")
async def startup_event():
    print(f"伺服器啟動中...")
    load_anime_data_from_db()
    
    global db, TAG_COMBINATION_MAPPING, EMOTION_CATEGORY_MAPPING
    try:
        # <<<<<<< 關鍵修改：在這裡一次性完成驗證和初始化 >>>>>>>
        credentials, project_id = google.auth.default()
        if project_id:
            print(f"INFO: Google Cloud Project ID '{project_id}' 已自動偵測。")
        else:
            # 如果還是找不到，提供一個後備方案或明確的錯誤
            project_id = "animetext" # 或者您的真實 Project ID
            print(f"WARN: 無法自動偵測 Project ID，使用預設值 '{project_id}'。")

        # 將偵測到的 project_id 傳入
        db = firestore.Client(project="animetext", database="anime-label")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        print("INFO: Firestore 初始化成功")
    except Exception as e:
        print("ERROR: Firestore 客戶端初始化失敗:")
        traceback.print_exc()  # 印出完整錯誤堆疊
        sys.exit(1)

    print("\n--- 開始從 Firestore 載入情感映射檔案 ---")
    start_time = time.time()
    try:
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            if '作品分類' in data and '情感分類' in data and isinstance(data['情感分類'], list):
                TAG_COMBINATION_MAPPING[data['作品分類']] = list(set(data['情感分類']))
        
        emotion_label_docs = db.collection('emotion_label').stream()
        for doc in emotion_label_docs:
            data = doc.to_dict()
            if '情感分類' in data and '情緒' in data and isinstance(data['情緒'], list):
                EMOTION_CATEGORY_MAPPING[data['情感分類']] = list(set(data['情緒']))
        
        print("INFO: 情感映射從 Firestore 載入成功。")
    except Exception as e:
        print(f"ERROR: 從 Firestore 載入映射失敗: {e}"); traceback.print_exc(); sys.exit(1)
    print(f"--- 情感映射載入完成，耗時 {time.time() - start_time:.2f} 秒 ---\n")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request): return templates.TemplateResponse("animetop.html", {"request": request})

@app.get("/search_anime_names")
async def search_anime_names(query: str = ""):
    if not query: return []
    return sorted([name for name in AVAILABLE_ANIME_NAMES if query.lower() in name.lower()])

@app.get("/get_emotion_categories")
async def get_emotion_categories():
    if not EMOTION_CATEGORY_MAPPING: raise HTTPException(500, "情感分類映射未成功載入。")
    return sorted(list(EMOTION_CATEGORY_MAPPING.keys()))


@app.get("/get_emotions")
async def get_emotions_api(anime_name: str, custom_emotions: list[str] = Query(None)):
    t_start = time.time()
    print(f"\n--- 收到搜尋請求: '{anime_name}' (時間: {t_start}) ---")
    normalized_name = unicodedata.normalize('NFC', anime_name.strip())
    if normalized_name not in AVAILABLE_ANIME_NAMES: raise HTTPException(404, f"找不到 '{anime_name}' 的數據。")

    t_db_start = time.time()
    copy_sql_query = sql.SQL('COPY (SELECT "彈幕", "label", "label2", "作品名", "集數", "時間", "情緒" FROM anime_danmaku WHERE "作品名" = {anime_name}) TO STDOUT WITH CSV HEADER DELIMITER \',\'').format(anime_name=sql.Literal(normalized_name))
    try:
        buffer = io.StringIO()
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.copy_expert(copy_sql_query, buffer, size=8192)
        buffer.seek(0)
        df_danmaku = pd.read_csv(buffer)
        if df_danmaku.empty: raise HTTPException(404, f"資料庫中沒有找到 '{normalized_name}' 的彈幕數據。")
        df_danmaku['集數'] = df_danmaku['集數'].astype(str)
    except HTTPException as e: raise e
    except Exception as e: raise HTTPException(500, f"讀取彈幕數據時發生錯誤: {e}")
    t_db_end = time.time()
    print(f"  [計時] 資料庫彈幕讀取 (高效模式): {t_db_end - t_db_start:.4f} 秒")

    t_map_start = time.time()
    dynamic_emotion_mapping = {}
    if custom_emotions:
        print(f"INFO: 使用者自訂模式: {custom_emotions}")
        for category in custom_emotions:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
    else:
        print("INFO: 使用預設模式 (最佳完全匹配)")
        tags = ANIME_TAGS_DB.get(normalized_name)
        if not tags: raise HTTPException(404, f"找不到 '{anime_name}' 的分類數據 (tags)。")
        
        anime_tags_set = set(tags)
        best_match_key, max_match_length = None, -1
        for rule_key in TAG_COMBINATION_MAPPING.keys():
            rule_tags_set = set(rule_key.split('|'))
            if rule_tags_set.issubset(anime_tags_set) and len(rule_tags_set) > max_match_length:
                max_match_length = len(rule_tags_set)
                best_match_key = rule_key

        if best_match_key:
            print(f"  -> 最佳完全匹配規則: '{best_match_key}'")
            categories_from_tags = TAG_COMBINATION_MAPPING.get(best_match_key, [])
            for category in categories_from_tags:
                if category in EMOTION_CATEGORY_MAPPING:
                    dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
        if not dynamic_emotion_mapping:
            raise HTTPException(404, f"無法為 '{anime_name}' (Tags: {tags}) 找到任何完全匹配的情感分類定義。")
    t_map_end = time.time()
    print(f"  [計時] 動態情感映射生成: {t_map_end - t_map_start:.4f} 秒")
    print(f"INFO: 動態情感映射生成完成: {list(dynamic_emotion_mapping.keys())}")
    
    t_core_start = time.time()
    try:
        # <<<<<<< 关键修改：呼叫函式时不再传递任何 op/ed 参数 >>>>>>>
        result = get_top3_emotions_fast(
            df=df_danmaku, 
            anime_name=normalized_name, 
            emotion_mapping=dynamic_emotion_mapping
        )
    except Exception as e:
        print(f"ERROR: 核心分析失敗: {e}"); traceback.print_exc(); raise HTTPException(500, "伺服器內部錯誤，情緒分析失敗。")
    t_core_end = time.time()
    print(f"  [計時] 核心情绪分析 (get_top3_emotions_fast): {t_core_end - t_core_start:.4f} 秒")

    if not result: raise HTTPException(404, f"找不到 '{anime_name}' 符合條件的情緒熱點數據。")

    t_sort_start = time.time()
    ordered_final_result = {}
    if not custom_emotions:
        t_top5_start = time.time()
        top_5_moments = get_top5_density_moments(
            df=df_danmaku,
            anime_name=normalized_name
        )
        t_top5_end = time.time()
        print(f"  [計時] TOP 10 弹幕时段计算: {t_top5_end - t_top5_start:.4f} 秒")

        priority_top = ["最精采/激烈的時刻", "LIVE/配樂", "虐點/感動"]
        priority_bottom_key = "彈幕最密集 TOP10"
        
        final_ordered_keys = [key for key in priority_top if key in result]
        other_categories = sorted([key for key in result if key not in priority_top])
        final_ordered_keys.extend(other_categories)
        if top_5_moments:
            final_ordered_keys.append(priority_bottom_key)

        for key in final_ordered_keys:
            if key == priority_bottom_key:
                ordered_final_result[key] = top_5_moments
            elif key in result:
                ordered_final_result[key] = result[key]
    else:
        ordered_final_result = dict(sorted(result.items()))
    t_sort_end = time.time()
    print(f"  [計時] 最终结果排序: {t_sort_end - t_sort_start:.4f} 秒")

    final_output = {
        "youtube_episode_urls": YOUTUBE_ANIME_EPISODE_URLS.get(normalized_name),
        "bahamut_episode_urls": BAHAMUT_ANIME_EPISODE_URLS.get(normalized_name),
        "cover_image_url": ANIME_COVER_IMAGE_URLS.get(normalized_name, ""),
        **ordered_final_result
    }
    
    t_end = time.time()
    print(f"--- 搜尋請求 '{anime_name}' 處理完成，總耗時 {t_end - t_start:.4f} 秒 ---\n")
    return final_output


# 在生產環境中，這個區塊不會被執行，因為 uvicorn 會直接執行 main:app
# if __name__ == '__main__':
#    logging.info("-----------------------------\n")
#    uvicorn.run("main_json:app", host="0.0.0.0", port=5000, reload=True)
