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
from psycopg2 import sql, pool # 導入連線池
from contextlib import contextmanager
import time
import traceback
import logging
import redis # 導入 Redis

# 導入 Firestore 相關模組
from google.cloud import firestore

# 從同級目錄導入 情感top3提出_dandadan_fast 模組
try:
    from 情感top3提出_dandadan_fast_json import get_all_highlights_single_pass
except ImportError:
    logging.error("ERROR: 無法導入 '情感top3提出_dandadan_fast_json' 模組。")
    sys.exit(1)

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# ====== CORS 配置 (保持不變) ======
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "http://127.0.0.1:5000",
    "https://your-service-name.onrender.com", # 請記得換成您 Render 服務的網址
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 為方便測試，可先設為 "*"，正式上線建議使用上面的 origins 列表
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ====== 資料庫與快取配置 ======
# 從環境變數讀取資料庫連線字串 (Render 推薦作法)
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# 全域變數
db_pool = None
redis_client = None
AVAILABLE_ANIME_NAMES = []
YOUTUBE_ANIME_EPISODE_URLS = {}
BAHAMUT_ANIME_EPISODE_URLS = {}
ANIME_COVER_IMAGE_URLS = {}
ANIME_TAGS_DB = {}
TAG_COMBINATION_MAPPING = {}
EMOTION_CATEGORY_MAPPING = {}
db = None # Firestore client

# ====== 連線管理 (使用連線池) ======
@contextmanager
def get_db_connection():
    conn = None
    if not db_pool:
        raise HTTPException(status_code=503, detail="資料庫連線池不可用。")
    try:
        conn = db_pool.getconn()
        yield conn
    except psycopg2.Error as e:
        logging.error(f"從連線池取得連線時出錯: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="資料庫服務暫時不可用。")
    finally:
        if conn:
            db_pool.putconn(conn) # 將連線放回池中，而不是關閉它

# ====== 應用程式啟動與關閉事件 ======
@app.on_event("startup")
async def startup_event():
    logging.info("伺服器啟動中...")
    
    # 初始化資料庫連線池
    global db_pool
    if not DATABASE_URL:
        logging.error("致命錯誤：未設定 DATABASE_URL 環境變數。")
        sys.exit(1)
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
        logging.info("資料庫連線池初始化成功。")
    except psycopg2.Error as e:
        logging.error(f"建立資料庫連線池失敗: {e}")
        sys.exit(1)

    # 初始化 Redis 連線
    global redis_client
    if not REDIS_URL:
        logging.warning("警告：未設定 REDIS_URL 環境變數，快取功能將被禁用。")
    else:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            logging.info("Redis 快取連線成功。")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"無法連線到 Redis，快取功能將被禁用: {e}")
            redis_client = None

    # 載入動漫元數據
    load_anime_data_mapping_from_db()

    # 初始化 Firestore
    global db
    try:
        # ... (您原本的 Firestore 初始化邏輯保持不變) ...
        db = firestore.Client(database="anime-label")
        logging.info("INFO: Firestore 客戶端初始化成功。")
    except Exception as e:
        logging.error(f"ERROR: Firestore 客戶端初始化失敗: {e}")
        # 在生產環境中，您可能不希望因此而退出，取決於您的需求
        # sys.exit(1)
    
    # 載入情感映射
    # ... (您原本的 Firestore 載入邏輯保持不變) ...
    try:
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            tag_key = data.get('作品分類', doc.id)
            categories = data.get('情感分類')
            if tag_key and isinstance(categories, list):
                TAG_COMBINATION_MAPPING[tag_key] = list(set(categories))
        
        emotion_label_docs = db.collection('emotion_label').stream()
        for doc in emotion_label_docs:
            data = doc.to_dict()
            emotion_category_key = data.get('情感分類', doc.id)
            emotions = data.get('情緒')
            if emotion_category_key and isinstance(emotions, list):
                EMOTION_CATEGORY_MAPPING[emotion_category_key] = list(set(emotions))
        logging.info("情感映射檔案從 Firestore 載入完成。")
    except Exception as e:
        logging.error(f"從 Firestore 載入情感映射失敗: {e}")


@app.on_event("shutdown")
def shutdown_event():
    logging.info("伺服器關閉中...")
    if db_pool:
        db_pool.closeall()
        logging.info("資料庫連線池已關閉。")
    if redis_client:
        redis_client.close()
        logging.info("Redis 連線已關閉。")

# `load_anime_data_mapping_from_db` 函式保持不變
def load_anime_data_mapping_from_db():
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    logging.info("從資料庫加載動漫數據映射...")
    # ... (此函式內部邏輯完全不變) ...
    total_process_start_time = time.time()
    AVAILABLE_ANIME_NAMES.clear(); YOUTUBE_ANIME_EPISODE_URLS.clear(); BAHAMUT_ANIME_EPISODE_URLS.clear(); ANIME_COVER_IMAGE_URLS.clear(); ANIME_TAGS_DB.clear()
    unique_anime_names_normalized = set()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL('SELECT "作品名", "集數", "巴哈動畫瘋網址", "YT網址", "封面圖", "作品分類" FROM anime_url ORDER BY "作品名", "集數";'))
                rows = cur.fetchall()
                if not rows: logging.warning("⚠️ 警告：資料庫的 'anime_url' 表中沒有找到任何數據。"); return
                for row in rows:
                    anime_original, episode, bahamut_url, youtube_url, cover_image_val, tags_json = row
                    normalized_anime_name = unicodedata.normalize('NFC', str(anime_original).strip())
                    unique_anime_names_normalized.add(normalized_anime_name)
                    ep_key_raw = episode
                    ep_key = str(int(float(ep_key_raw))).strip() if ep_key_raw is not None and str(ep_key_raw).replace('.', '', 1).isdigit() else str(ep_key_raw).strip()
                    if normalized_anime_name not in YOUTUBE_ANIME_EPISODE_URLS: YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name] = {}
                    if normalized_anime_name not in BAHAMUT_ANIME_EPISODE_URLS: BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name] = {}
                    if youtube_url:
                        yt_url_str = str(youtube_url).strip()
                        video_id = yt_url_str.split('v=')[-1].split('&')[0] if "youtube.com/watch?v=" in yt_url_str else (yt_url_str.split('youtu.be/')[-1].split('?')[0] if "youtu.be/" in yt_url_str else None)
                        if video_id: YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = video_id
                    if bahamut_url: BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = str(bahamut_url).strip()
                    if cover_image_val and normalized_anime_name not in ANIME_COVER_IMAGE_URLS: ANIME_COVER_IMAGE_URLS[normalized_anime_name] = str(cover_image_val).strip()
                    if tags_json and normalized_anime_name not in ANIME_TAGS_DB:
                        tags = json.loads(tags_json.replace("'", '"')) if isinstance(tags_json, str) else (tags_json if isinstance(tags_json, list) else [])
                        if isinstance(tags, list) and all(isinstance(t, str) for t in tags): ANIME_TAGS_DB[normalized_anime_name] = tags
                AVAILABLE_ANIME_NAMES = sorted(list(unique_anime_names_normalized))
                logging.info(f"從資料庫加載完成。總計 {len(AVAILABLE_ANIME_NAMES)} 部動漫。")
    except Exception as e:
        logging.error(f"加載動漫數據映射時發生錯誤: {e}"); traceback.print_exc(); sys.exit(1)
    finally:
        total_process_end_time = time.time(); logging.info(f"--- 資料庫數據載入完成，總耗時 {total_process_end_time - total_process_start_time:.4f} 秒 ---")


# ====== 其他 API 端點 (保持不變) ======
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return FileResponse("static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request): return templates.TemplateResponse("animetop.html", {"request": request})

@app.get("/search_anime_names")
async def search_anime_names(query: str = Query("", description="搜尋動漫名稱的關鍵字")):
    if not query: return []
    return sorted([name for name in AVAILABLE_ANIME_NAMES if query.lower() in name.lower()])

@app.get("/get_emotion_categories")
async def get_emotion_categories():
    if not EMOTION_CATEGORY_MAPPING: raise HTTPException(status_code=500, detail="情感分類映射未成功載入。")
    all_categories = sorted(list(EMOTION_CATEGORY_MAPPING.keys()))
    all_categories.extend(["精彩的戰鬥時段", "TOP 5 彈幕時段"])
    return sorted(list(set(all_categories)))


# ====== 核心 API: 整合快取邏輯 ======
@app.get("/get_emotions")
async def get_emotions_api(
    anime_name: str = Query(..., description="要查詢的動漫名稱"),
    custom_emotions: list[str] = Query(None, description="使用者自訂的情感分類列表")
):
    request_start_time = time.time()
    cache_key = None

    # --- 步驟 1: 檢查快取 ---
    if redis_client:
        emotion_key_part = "default" if not custom_emotions else "|".join(sorted(custom_emotions))
        cache_key = f"anime-result:{anime_name}:{emotion_key_part}"
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logging.info(f"--- 快取命中 (Cache Hit) for key: {cache_key} ---")
                logging.info(f"--- 請求 '{anime_name}' 從快取處理完成，耗時: {time.time() - request_start_time:.4f} 秒 ---\n")
                return json.loads(cached_result)
        except redis.exceptions.RedisError as e:
            logging.error(f"讀取 Redis 快取時出錯: {e}")

    logging.info(f"--- 快取未命中 (Cache Miss) for key: {cache_key or 'N/A'} ---")
    
    # --- 步驟 2: 快取未命中，執行完整分析流程 (和原版相同) ---
    normalized_anime_name = unicodedata.normalize('NFC', anime_name.strip())
    if normalized_anime_name not in AVAILABLE_ANIME_NAMES:
        raise HTTPException(status_code=404, detail=f"抱歉，資料庫中沒有找到 '{anime_name}' 的數據。")

    anime_episode_urls = YOUTUBE_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    bahamut_episode_urls = BAHAMUT_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    cover_image_url = ANIME_COVER_IMAGE_URLS.get(normalized_anime_name, "")
    
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT "彈幕", "label", "label2", "作品名", "集數", "時間", "情緒" FROM anime_danmaku WHERE "作品名" = %s;'), (normalized_anime_name,))
            rows = cur.fetchall()
            if not rows: raise HTTPException(status_code=404, detail=f"抱歉，資料庫中沒有找到 '{normalized_anime_name}' 的彈幕數據。")
            df_danmaku = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
            df_danmaku['集數'] = df_danmaku['集數'].astype(str)
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"從資料庫讀取彈幕數據時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail="讀取彈幕數據失敗。")

    dynamic_emotion_mapping = {}
    if custom_emotions:
        for category in custom_emotions:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
    else:
        tags = ANIME_TAGS_DB.get(normalized_anime_name, [])
        if not tags: raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' 的作品分類數據。")
        collected_emotion_categories = set()
        for mapping_key, categories_list in TAG_COMBINATION_MAPPING.items():
            if set(mapping_key.split('|')).issubset(set(tags)):
                collected_emotion_categories.update(categories_list)
        if not collected_emotion_categories: raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' (分類: {tags}) 對應的情感分類。")
        for category in list(collected_emotion_categories):
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
    
    try:
        result = get_all_highlights_single_pass(df=df_danmaku, anime_name=normalized_anime_name, emotion_mapping=dynamic_emotion_mapping)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤，分析失敗: {e}")

    if not result: raise HTTPException(status_code=404, detail=f"抱歉，沒有找到 '{normalized_anime_name}' 的熱點數據。")
    
    # ... (結果處理和排序邏輯保持不變) ...
    processed_result = {}
    for emotion_category, highlights_list in result.items():
        if isinstance(highlights_list, list) and highlights_list:
            if custom_emotions and emotion_category not in custom_emotions: continue
            processed_highlights = [{k: (int(v) if isinstance(v, np.integer) else v) for k, v in item.items()} for item in highlights_list]
            processed_result[emotion_category] = processed_highlights
    if not processed_result: raise HTTPException(status_code=404, detail=f"抱歉，作品 '{normalized_anime_name}' 處理後沒有發現有效的亮點。")
    ordered_final_result = {}
    if not custom_emotions:
        priority_categories = ["精彩的戰鬥時段", "LIVE/神配樂", "虐點/感動", "突如其來/震驚", "虐點", "爆笑", "劇情高潮/震撼", "最精采/激烈的時刻", "TOP 10 彈幕時段"]
        other_categories_with_counts = sorted([(cat, len(highlights)) for cat, highlights in processed_result.items() if cat not in priority_categories], key=lambda x: x[1], reverse=True)
        top_other_categories = [cat for cat, _ in other_categories_with_counts[:5]]
        ordered_keys = [p_cat for p_cat in priority_categories if p_cat in processed_result]
        ordered_keys.extend([t_cat for t_cat in top_other_categories if t_cat not in ordered_keys])
        ordered_keys.extend(sorted([k for k in processed_result if k not in ordered_keys]))
        for key in ordered_keys: ordered_final_result[key] = processed_result[key]
    else:
        ordered_final_result = dict(sorted(processed_result.items()))

    final_output = {
        "youtube_episode_urls": anime_episode_urls,
        "bahamut_episode_urls": bahamut_episode_urls,
        "cover_image_url": cover_image_url,
        **ordered_final_result
    }

    # --- 步驟 3: 將新結果存入快取 ---
    if redis_client and cache_key:
        try:
            # 將 final_output 轉換為 JSON 字串並存入 Redis，設定過期時間為 24 小時
            redis_client.set(cache_key, json.dumps(final_output), ex=86400)
            logging.info(f"--- 結果已寫入快取，鍵為: {cache_key} ---")
        except redis.exceptions.RedisError as e:
            logging.error(f"寫入 Redis 快取時出錯: {e}")

    logging.info(f"--- 請求 '{anime_name}' 處理完成，總耗時: {time.time() - request_start_time:.4f} 秒 ---\n")
    return final_output

# 在生產環境中，這個區塊不會被執行
# if __name__ == '__main__':
#    logging.info("-----------------------------\n")
#    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 5000)), reload=True)
