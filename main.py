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
    # <<<<<<<<<<<<<<< 步驟 1: 同時導入 get_top5_density_moments >>>>>>>>>>>>>>>
    from 情感top3提出_dandadan_fast_json import get_top3_emotions_fast, get_top5_density_moments
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
DB_HOST = os.getenv("DB_HOST", "35.223.124.201")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "anime1si2sun")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "lty890509")

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

def load_anime_data_mapping_from_db():
    """
    啟動時從 PostgreSQL 的 anime_url 表加載所有 URL、封面圖資訊和作品分類。
    """
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    
    total_process_start_time = time.time()
    logging.info("INFO: 從資料庫加載動漫數據映射...") 
    
    AVAILABLE_ANIME_NAMES.clear()
    YOUTUBE_ANIME_EPISODE_URLS.clear()
    BAHAMUT_ANIME_EPISODE_URLS.clear()
    ANIME_COVER_IMAGE_URLS.clear()
    ANIME_TAGS_DB.clear()
    
    unique_anime_names_normalized = set()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                    SELECT "作品名", "集數", "巴哈動畫瘋網址", "YT網址", "封面圖", "作品分類"
                    FROM anime_url ORDER BY "作品名", "集數";
                    """)
                )
                rows = cur.fetchall()

                if not rows:
                    logging.warning("⚠️ 警告：資料庫的 'anime_url' 表中沒有找到任何數據。")
                    return

                for row in rows:
                    anime_original, episode, bahamut_url, youtube_url, cover_image_val, tags_json = row
                    normalized_anime_name = unicodedata.normalize('NFC', str(anime_original).strip())
                    unique_anime_names_normalized.add(normalized_anime_name)
                    ep_key_raw = episode
                    ep_key = ""
                    if ep_key_raw is not None:
                        try:
                            ep_key = str(int(float(ep_key_raw))).strip()
                        except (ValueError, TypeError):
                            ep_key = str(ep_key_raw).strip()

                    if normalized_anime_name not in YOUTUBE_ANIME_EPISODE_URLS:
                        YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name] = {}
                    if normalized_anime_name not in BAHAMUT_ANIME_EPISODE_URLS:
                        BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name] = {}

                    if youtube_url:
                        yt_url_str = str(youtube_url).strip()
                        video_id = None
                        if "youtube.com/watch?v=" in yt_url_str:
                            video_id = yt_url_str.split("v=")[-1].split("&")[0].split("?")[0]
                        elif "youtu.be/" in yt_url_str:
                            video_id = yt_url_str.split("youtu.be/")[-1].split("?")[0]
                        
                        if video_id:
                            YOUTUBE_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = video_id

                    if bahamut_url:
                        BAHAMUT_ANIME_EPISODE_URLS[normalized_anime_name][ep_key] = str(bahamut_url).strip()

                    if cover_image_val and normalized_anime_name not in ANIME_COVER_IMAGE_URLS:
                        ANIME_COVER_IMAGE_URLS[normalized_anime_name] = str(cover_image_val).strip()

                    if tags_json and normalized_anime_name not in ANIME_TAGS_DB:
                        tags = []
                        if isinstance(tags_json, str):
                            try:
                                tags = json.loads(tags_json.replace("'", '"'))
                            except json.JSONDecodeError:
                                tags = []
                        elif isinstance(tags_json, list):
                            tags = tags_json
                        
                        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
                            ANIME_TAGS_DB[normalized_anime_name] = tags
                    
                AVAILABLE_ANIME_NAMES = sorted(list(unique_anime_names_normalized))
                logging.info(f"INFO: 從資料庫加載完成。總計 {len(AVAILABLE_ANIME_NAMES)} 部動漫可供搜尋。")

    except HTTPException as e:
        logging.error(f"ERROR: 加載動漫數據映射失敗: {e.detail}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"ERROR: 加載動漫數據映射時發生未知錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        total_process_end_time = time.time()
        logging.info(f"--- 資料庫數據載入完成，總耗時 {total_process_end_time - total_process_start_time:.4f} 秒 ---")


@app.on_event("startup")
async def startup_event():
    logging.info(f"伺服器啟動中，當前時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    load_anime_data_mapping_from_db()

    global db
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
        
        logging.info("INFO: Firestore 客戶端初始化成功。")
    except Exception as e:
        logging.error(f"ERROR: Firestore 客戶端初始化失敗: {e}")
        sys.exit(1)

    logging.info("\n--- 開始從 Firestore 載入情感映射檔案 ---")
    start_mapping_load_time = time.time()

    try:
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            tag_key = data.get('作品分類', doc.id)
            categories = data.get('情感分類')
            if tag_key and isinstance(categories, list):
                TAG_COMBINATION_MAPPING[tag_key] = list(set(categories))
        logging.info("INFO: TAG_COMBINATION_MAPPING 從 Firestore 'anime_label' 集合載入成功。")
    except Exception as e:
        logging.error(f"ERROR: 從 Firestore 'anime_label' 載入 TAG_COMBINATION_MAPPING 失敗: {e}")
        sys.exit(1)

    try:
        emotion_label_docs = db.collection('emotion_label').stream()
        for doc in emotion_label_docs:
            data = doc.to_dict()
            emotion_category_key = data.get('情感分類', doc.id)
            emotions = data.get('情緒')
            if emotion_category_key and isinstance(emotions, list):
                EMOTION_CATEGORY_MAPPING[emotion_category_key] = list(set(emotions))
        logging.info("INFO: EMOTION_CATEGORY_MAPPING 從 Firestore 'emotion_label' 集合載入成功。")
    except Exception as e:
        logging.error(f"ERROR: 從 Firestore 'emotion_label' 載入 EMOTION_CATEGORY_MAPPING 失敗: {e}")
        sys.exit(1)

    end_mapping_load_time = time.time()
    logging.info(f"--- 情感映射檔案從 Firestore 載入完成，總耗時 {end_mapping_load_time - start_mapping_load_time:.4f} 秒 ---\n")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("animetop.html", {"request": request})

@app.get("/search_anime_names")
async def search_anime_names(query: str = Query("", description="搜尋動漫名稱的關鍵字")):
    if not query: return []
    query_lower = query.lower()
    return sorted([name for name in AVAILABLE_ANIME_NAMES if query_lower in name.lower()])

@app.get("/get_emotion_categories")
async def get_emotion_categories():
    if not EMOTION_CATEGORY_MAPPING:
        raise HTTPException(status_code=500, detail="情感分類映射未成功載入。")
    return sorted(list(EMOTION_CATEGORY_MAPPING.keys()))


@app.get("/get_emotions")
async def get_emotions_api(
    anime_name: str = Query(..., description="要查詢的動漫名稱"),
    custom_emotions: list[str] = Query(None, description="使用者自訂的情感分類列表")
):
    request_start_time = time.time()
    logging.info(f"\n--- 收到搜尋請求: '{anime_name}', 開始處理 ---")

    if not anime_name:
        raise HTTPException(status_code=400, detail="請提供動漫名稱")

    normalized_anime_name = unicodedata.normalize('NFC', anime_name.strip())
    
    if normalized_anime_name not in AVAILABLE_ANIME_NAMES:
        raise HTTPException(status_code=404, detail=f"抱歉，資料庫中沒有找到 '{anime_name}' 的數據。")

    anime_episode_urls = YOUTUBE_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    bahamut_episode_urls = BAHAMUT_ANIME_EPISODE_URLS.get(normalized_anime_name, {})
    cover_image_url = ANIME_COVER_IMAGE_URLS.get(normalized_anime_name, "")

    # 從資料庫獲取彈幕
    df_danmaku = pd.DataFrame()
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL('SELECT "彈幕", "label", "label2", "作品名", "集數", "時間", "情緒" FROM anime_danmaku WHERE "作品名" = %s;'),
                (normalized_anime_name,)
            )
            rows = cur.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail=f"抱歉，資料庫中沒有找到 '{normalized_anime_name}' 的彈幕數據。")
            
            column_names = [desc[0] for desc in cur.description]
            df_danmaku = pd.DataFrame(rows, columns=column_names)
            df_danmaku['集數'] = df_danmaku['集數'].astype(str)
            logging.info(f"INFO: 從資料庫成功載入 {len(df_danmaku)} 筆 '{normalized_anime_name}' 的彈幕數據。")

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"ERROR: 從資料庫讀取彈幕數據時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"從資料庫讀取彈幕數據時發生錯誤: {str(e)}")

    # 執行情感分析
    dynamic_emotion_mapping = {}
    if custom_emotions:
        logging.info(f"INFO: 使用者自訂模式，選擇的情感分類: {custom_emotions}")
        for category in custom_emotions:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
        if not dynamic_emotion_mapping:
            raise HTTPException(status_code=404, detail=f"您選擇的分類 {custom_emotions} 均無效。")
    else:
        logging.info("INFO: 使用預設模式，根據作品分類生成情感映射。")
        tags = ANIME_TAGS_DB.get(normalized_anime_name, [])
        if not tags:
            raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' 的作品分類數據。")

        anime_tags_set = set(tags)
        collected_emotion_categories = set()
        for mapping_key, categories_list in TAG_COMBINATION_MAPPING.items():
            mapping_tags_set = set(mapping_key.split('|'))
            if mapping_tags_set and mapping_tags_set.issubset(anime_tags_set):
                logging.info(f"  -> 匹配到標籤組合 '{mapping_key}'，加入情感分類: {categories_list}")
                collected_emotion_categories.update(categories_list)

        if not collected_emotion_categories:
            raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' (作品分類: {tags}) 對應的情感分類定義。")

        categories = list(collected_emotion_categories)
        for category in categories:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
        
        if not dynamic_emotion_mapping:
            raise HTTPException(status_code=404, detail=f"根據作品分類 '{tags}' 組合出的情感分類 ({categories}) 沒有對應的原始情緒詞。")

    try:
        # 呼叫主分析函式
        result = get_top3_emotions_fast(df_danmaku, normalized_anime_name, emotion_mapping=dynamic_emotion_mapping)
        
        # <<<<<<<<<<<<<<< 步驟 2: 額外呼叫彈幕密度分析函式 >>>>>>>>>>>>>>>
        top_5_moments = get_top5_density_moments(df_danmaku, normalized_anime_name)
        
        # <<<<<<<<<<<<<<< 步驟 3: 將彈幕密度結果整合進來 >>>>>>>>>>>>>>>
        if top_5_moments:
            result["TOP 10 彈幕時段"] = top_5_moments
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤，分析失敗: {str(e)}")

    if not result:
        raise HTTPException(status_code=404, detail=f"抱歉，沒有找到 '{normalized_anime_name}' 的熱點數據。")

    # 結果處理和排序
    processed_result = {}
    for emotion_category, highlights_list in result.items():
        if isinstance(highlights_list, list) and highlights_list:
            processed_highlights = []
            for item in highlights_list:
                processed_item = {k: (int(v) if isinstance(v, np.integer) else v) for k, v in item.items()}
                processed_highlights.append(processed_item)
            processed_result[emotion_category] = processed_highlights

    if not processed_result:
        raise HTTPException(status_code=404, detail=f"抱歉，作品 '{normalized_anime_name}' 處理後沒有發現有效的亮點。")

    # 根據模式決定排序方式
    ordered_final_result = {}
    if not custom_emotions:
        # <<<<<<<<<<<<<<< 步驟 4: 將 "TOP 5 彈幕時段" 加入優先排序 >>>>>>>>>>>>>>>
        priority_categories = [ "精彩的戰鬥時段", "虐點/爆點","TOP 10 彈幕時段"]
        
        other_categories_with_counts = [(cat, len(highlights)) for cat, highlights in processed_result.items() if cat not in priority_categories]
        top_other_categories = [cat for cat, _ in sorted(other_categories_with_counts, key=lambda x: x[1], reverse=True)[:5]]
        
        ordered_keys = []
        for p_cat in priority_categories:
            if p_cat in processed_result: ordered_keys.append(p_cat)
        for t_cat in top_other_categories:
            if t_cat not in ordered_keys: ordered_keys.append(t_cat)
        
        remaining_keys = sorted([k for k in processed_result if k not in ordered_keys])
        ordered_keys.extend(remaining_keys)

        for key in ordered_keys:
            ordered_final_result[key] = processed_result[key]
    else:
        # 自訂模式：按字母順序排序
        ordered_final_result = dict(sorted(processed_result.items()))

    # 組合最終輸出
    final_output = {
        "youtube_episode_urls": anime_episode_urls, 
        "bahamut_episode_urls": bahamut_episode_urls, 
        "cover_image_url": cover_image_url,
        **ordered_final_result
    }
    
    logging.info(f"--- 請求 '{anime_name}' 處理完成，總耗時: {time.time() - request_start_time:.4f} 秒 ---\n")
    return final_output

# if __name__ == '__main__':
#    logging.info("-----------------------------\n")
#    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)




