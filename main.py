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
from psycopg2 import sql, pool # <--- 導入連線池
from contextlib import contextmanager
import time
import traceback
import logging
import redis # <--- 導入 Redis

# 導入 Firestore 相關模組
from google.cloud import firestore

# 從同級目錄導入 情感top3提出_dandadan_fast 模組
try:
    from 情感top3提出_dandadan_fast_json import get_all_highlights_single_pass
except ImportError:
    logging.error("ERROR: 無法導入 '情感top3提出_dandadan_fast_json' 模組。請確保該檔案存在且在可被Python找到的路徑上。")
    sys.exit(1)

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# ====== CORS 配置 ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允許所有來源，方便開發。正式上線建議換成您的前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# ====== 資料庫與快取配置 (混合模式) ======
# 1. 維持原本的 PostgreSQL 資料庫配置方式
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DATABASE_URL = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# 2. 從環境變數讀取 Redis 連線位址
REDIS_URL = os.getenv("REDIS_URL")

# ====== 全域連線變數 ======
db_pool = None
redis_client = None
db = None # Firestore client

# 全域資料變數
AVAILABLE_ANIME_NAMES = []
YOUTUBE_ANIME_EPISODE_URLS = {}
BAHAMUT_ANIME_EPISODE_URLS = {}
ANIME_COVER_IMAGE_URLS = {}
ANIME_TAGS_DB = {}
TAG_COMBINATION_MAPPING = {}
EMOTION_CATEGORY_MAPPING = {}
BATTLE_SEGMENT_SETTINGS = {} # <--- 新增：儲存戰鬥時段設定

# ====== 連線管理 (使用連線池與 Context Manager) ======
@contextmanager
def get_db_connection():
    if not db_pool:
        raise HTTPException(status_code=503, detail="資料庫連線池不可用。")
    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    except psycopg2.Error as e:
        logging.error(f"從連線池取得連線時出錯: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="資料庫服務暫時不可用。")
    finally:
        if conn:
            db_pool.putconn(conn)

# ====== 應用程式生命週期事件 ======
@app.on_event("startup")
async def startup_event():
    global db_pool, redis_client, db
    logging.info("伺服器啟動中，開始初始化所有服務...")

    # 1. 初始化資料庫連線池
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
        logging.info("資料庫連線池初始化成功。")
    except psycopg2.Error as e:
        logging.error(f"建立資料庫連線池失敗: {e}")
        sys.exit(1)

    # 2. 初始化 Redis 連線
    if not REDIS_URL:
        logging.warning("警告：未設定 REDIS_URL 環境變數，快取功能將被禁用。")
    else:
        try:
            redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            redis_client.ping()
            logging.info(f"Redis 快取連線成功 (Host: {redis_client.connection_pool.connection_kwargs.get('host')})。")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"無法連線到 Redis (URL: {REDIS_URL})，快取功能將被禁用: {e}")
            redis_client = None

    # 3. 載入動漫元數據
    load_anime_data_mapping_from_db()

    # 4. 初始化 Firestore 客戶端
    try:
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            credentials_json = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(credentials_json)
            db = firestore.Client(project=credentials.project_id, credentials=credentials, database="anime-label")
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            db = firestore.Client(database="anime-label")
        else:
            db = firestore.Client(database="anime-label")
        logging.info("Firestore 客戶端初始化成功。")
    except Exception as e:
        logging.error(f"Firestore 客戶端初始化失敗: {e}")
        db = None

    # 5. 從 Firestore 載入情感映射與設定
    if db:
        load_emotion_mappings_from_firestore()

@app.on_event("shutdown")
def shutdown_event():
    logging.info("伺服器關閉中...")
    if db_pool:
        db_pool.closeall()
        logging.info("資料庫連線池已關閉。")
    if redis_client:
        redis_client.close()
        logging.info("Redis 連線已關閉。")

def load_anime_data_mapping_from_db():
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    total_process_start_time = time.time()
    logging.info("INFO: 從資料庫加載動漫數據映射...")
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
                    ep_key_raw = episode; ep_key = ""
                    if ep_key_raw is not None:
                        try: ep_key = str(int(float(ep_key_raw))).strip()
                        except (ValueError, TypeError): ep_key = str(ep_key_raw).strip()
                    if normalized_anime_name not in YOUTUBE_ANIME_EPISODE_URLS: YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name] = {}
                    if normalized_anime_name not in BAHAMUT_ANIME_EPISODE_URLS: BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name] = {}
                    if youtube_url:
                        yt_url_str = str(youtube_url).strip(); video_id = None
                        if "youtube.com/watch?v=" in yt_url_str: video_id = yt_url_str.split("v=")[-1].split("&")[0].split("?")[0]
                        elif "youtu.be/" in yt_url_str: video_id = yt_url_str.split("youtu.be/")[-1].split("?")[0]
                        if video_id: YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = video_id
                    if bahamut_url: BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = str(bahamut_url).strip()
                    if cover_image_val and normalized_anime_name not in ANIME_COVER_IMAGE_URLS: ANIME_COVER_IMAGE_URLS[normalized_anime_name] = str(cover_image_val).strip()
                    if tags_json and normalized_anime_name not in ANIME_TAGS_DB:
                        tags = []
                        if isinstance(tags_json, str):
                            try: tags = json.loads(tags_json.replace("'", '"'))
                            except json.JSONDecodeError: tags = []
                        elif isinstance(tags_json, list): tags = tags_json
                        if isinstance(tags, list) and all(isinstance(t, str) for t in tags): ANIME_TAGS_DB[normalized_anime_name] = tags
                AVAILABLE_ANIME_NAMES = sorted(list(unique_anime_names_normalized))
                logging.info(f"INFO: 從資料庫加載完成。總計 {len(AVAILABLE_ANIME_NAMES)} 部動漫可供搜尋。")
    except Exception as e:
        logging.error(f"ERROR: 加載動漫數據映射時發生未知錯誤: {e}"); traceback.print_exc()
    finally:
        total_process_end_time = time.time()
        logging.info(f"--- 資料庫數據載入完成，總耗時 {total_process_end_time - total_process_start_time:.4f} 秒 ---")

def load_emotion_mappings_from_firestore():
    global TAG_COMBINATION_MAPPING, EMOTION_CATEGORY_MAPPING, BATTLE_SEGMENT_SETTINGS
    logging.info("\n--- 開始從 Firestore 載入情感映射與設定檔案 ---")
    start_mapping_load_time = time.time()
    try:
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            tag_key = data.get('作品分類', doc.id)
            categories = data.get('情感分類')
            if tag_key and isinstance(categories, list):
                # 直接儲存從 Firestore 讀取的完整列表
                TAG_COMBINATION_MAPPING[tag_key] = list(set(categories))
        
        logging.info("INFO: 標籤組合與戰鬥時段設定從 Firestore 'anime_label' 集合載入成功。")
        
        emotion_label_docs = db.collection('emotion_label').stream()
        for doc in emotion_label_docs:
            data = doc.to_dict()
            emotion_category_key = data.get('情感分類', doc.id)
            emotions = data.get('情緒')
            if emotion_category_key and isinstance(emotions, list):
                EMOTION_CATEGORY_MAPPING[emotion_category_key] = list(set(emotions))
        logging.info("INFO: 情緒詞從 Firestore 'emotion_label' 集合載入成功。")

    except Exception as e:
        logging.error(f"ERROR: 從 Firestore 載入情感映射失敗: {e}")
    finally:
        end_mapping_load_time = time.time()
        logging.info(f"--- 情感映射與設定檔案從 Firestore 載入完成，總耗時 {end_mapping_load_time - start_mapping_load_time:.4f} 秒 ---\n")

# ====== API 端點 ======
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
    if not EMOTION_CATEGORY_MAPPING:
        raise HTTPException(status_code=500, detail="情感分類映射未成功載入。")
    all_categories = sorted(list(EMOTION_CATEGORY_MAPPING.keys()))
    all_categories.extend(["精彩的戰鬥時段", "TOP 10 彈幕時段"]) # Note: TOP 5 is a legacy name, analysis does TOP 10.
    return sorted(list(set(all_categories)))

@app.get("/get_emotions")
async def get_emotions_api(
    anime_name: str = Query(..., description="要查詢的動漫名稱"),
    custom_emotions: list[str] = Query(None, description="使用者自訂的情感分類列表")
):
    request_start_time = time.time()
    cache_key = None

    if redis_client:
        emotion_key_part = "default" if not custom_emotions else "|".join(sorted(custom_emotions))
        cache_key = f"anime-result:{anime_name}:{emotion_key_part}"
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logging.info(f"--- 快取命中 (Cache Hit) for key: {cache_key} ---")
                return json.loads(cached_result)
        except redis.exceptions.RedisError as e:
            logging.error(f"讀取 Redis 快取時出錯: {e}")

    logging.info(f"--- 快取未命中 (Cache Miss) for key: {cache_key or 'N/A'}. 開始執行完整分析... ---")
    
    if not anime_name: raise HTTPException(status_code=400, detail="請提供動漫名稱")
    normalized_anime_name = unicodedata.normalize('NFC', anime_name.strip())
    if normalized_anime_name not in AVAILABLE_ANIME_NAMES:
        raise HTTPException(status_code=404, detail=f"資料庫中沒有找到 '{anime_name}' 的數據。")

    anime_episode_urls = YOUTUBE_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    bahamut_episode_urls = BAHAMUT_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    cover_image_url = ANIME_COVER_IMAGE_URLS.get(normalized_anime_name, "")

    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL('SELECT "彈幕", "label", "label2", "作品名", "集數", "時間", "情緒" FROM anime_danmaku WHERE "作品名" = %s;'),(normalized_anime_name,))
            rows = cur.fetchall()
            if not rows: raise HTTPException(status_code=404, detail=f"資料庫中沒有找到 '{normalized_anime_name}' 的彈幕數據。")
            df_danmaku = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
            df_danmaku['集數'] = df_danmaku['集數'].astype(str)
            logging.info(f"INFO: 從資料庫成功載入 {len(df_danmaku)} 筆 '{normalized_anime_name}' 的彈幕數據。")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"從資料庫讀取彈幕數據時發生錯誤: {e}")

    dynamic_emotion_mapping = {}
    should_calculate_battle = False

    if custom_emotions:
        logging.info(f"INFO: 使用者自訂模式，選擇的分類: {custom_emotions}")
        if "精彩的戰鬥時段" in custom_emotions:
            should_calculate_battle = True
        for category in custom_emotions:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
    else:
        logging.info("INFO: 使用預設模式，根據作品分類生成情感映射。")
        tags = ANIME_TAGS_DB.get(normalized_anime_name, [])
        if not tags: raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' 的作品分類數據。")
        anime_tags_set = set(tags)
        collected_emotion_categories = set()
        
        for mapping_key, categories_list in TAG_COMBINATION_MAPPING.items():
            if set(mapping_key.split('|')).issubset(anime_tags_set):
                collected_emotion_categories.update(categories_list)
        
                # 2. 檢查是否包含戰鬥時段指令
        if "精彩的戰鬥時段" in collected_emotion_categories:
            should_calculate_battle = True
            collected_emotion_categories.remove("精彩的戰鬥時段") # 移除指令，它不是一個真正的情感
            logging.info(f"  -> 根據 Firestore 設定，將啟用「精彩的戰鬥時段」分析。")

        for category in list(collected_emotion_categories):
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]

        # 4. 檢查是否有任何分析任務
        if not dynamic_emotion_mapping and not should_calculate_battle:
            raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' (分類: {tags}) 對應的有效情感分類定義。")
    try:
        result = get_all_highlights_single_pass(
            df=df_danmaku, 
            anime_name=normalized_anime_name, 
            emotion_mapping=dynamic_emotion_mapping,
            calculate_battle_segments=should_calculate_battle
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤，分析失敗: {e}")

    if not result: raise HTTPException(status_code=404, detail=f"抱歉，沒有找到 '{normalized_anime_name}' 的熱點數據。")
    
    processed_result = {}
    for emotion_category, highlights_list in result.items():
        if isinstance(highlights_list, list) and highlights_list:
            if custom_emotions and emotion_category not in custom_emotions: continue
            processed_highlights = [{k: (int(v) if isinstance(v, np.integer) else v) for k, v in item.items()} for item in highlights_list]
            processed_result[emotion_category] = processed_highlights
    if not processed_result: raise HTTPException(status_code=404, detail=f"抱歉，作品 '{normalized_anime_name}' 處理後沒有發現有效的亮點。")

    ordered_final_result = {}
    if not custom_emotions:
        priority_categories = ["精彩的戰鬥時段","LIVE/神配樂","虐點/感動","突如其來/震驚","虐點","爆笑","劇情高潮/震撼","最精采/激烈的時刻","TOP 10 彈幕時段"]
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
    
    if redis_client and cache_key:
        try:
            redis_client.set(cache_key, json.dumps(final_output, ensure_ascii=False), ex=86400)
            logging.info(f"--- 結果已成功寫入快取，鍵為: {cache_key} ---")
        except redis.exceptions.RedisError as e:
            logging.error(f"寫入 Redis 快取時出錯: {e}")

    logging.info(f"--- 請求 '{anime_name}' 完整分析處理完成，總耗時: {time.time() - request_start_time:.4f} 秒 ---\n")
    return final_output



