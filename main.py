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

def load_anime_data_mapping_from_db():
    """
    啟動時從 PostgreSQL 的 anime_url 表加載所有 URL、封面圖資訊和作品分類。
    """
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    
    total_process_start_time = time.time()
    logging.info("INFO: 從資料庫加載動漫數據映射...") # 使用 logging 替代 print
    
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
                    logging.warning("⚠️ 警告：資料庫的 'anime_url' 表中沒有找到任何數據。") # 使用 logging 替代 print
                    return

                for row in rows:
                    anime_original, episode, bahamut_url, youtube_url, cover_image_val, tags_json = row

                    normalized_anime_name = unicodedata.normalize('NFC', str(anime_original).strip())
                    unique_anime_names_normalized.add(normalized_anime_name)

                    ep_key_raw = episode
                    ep_key = ""
                    if ep_key_raw is not None:
                        try:
                        # 嘗試將其轉換為整數，然後再轉為字串
                            ep_key = str(int(float(ep_key_raw))).strip()
                        except (ValueError, TypeError):
                            # 如果轉換失敗，保留原始字串形式
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
                logging.info(f"INFO: 從資料庫加載完成。總計 {len(AVAILABLE_ANIME_NAMES)} 部動漫可供搜尋。") # 使用 logging 替代 print

    except HTTPException as e:
        logging.error(f"ERROR: 加載動漫數據映射失敗: {e.detail}") # 使用 logging 替代 print
        sys.exit(1)
    except Exception as e:
        logging.error(f"ERROR: 加載動漫數據映射時發生未知錯誤: {e}") # 使用 logging 替代 print
        traceback.print_exc()
        sys.exit(1)
    finally:
        total_process_end_time = time.time()
        logging.info(f"--- 資料庫數據載入完成，總耗時 {total_process_end_time - total_process_start_time:.4f} 秒 ---") # 使用 logging 替代 print


@app.on_event("startup")
async def startup_event():
    logging.info(f"伺服器啟動中，當前時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}") # 使用 logging 替代 print
    load_anime_data_mapping_from_db()

    global db
    try:
        # 嘗試從環境變數 GOOGLE_APPLICATION_CREDENTIALS_JSON 載入憑證
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"):
            credentials_json = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON"))
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_info(credentials_json)
            db = firestore.Client(project=credentials.project_id, credentials=credentials, database="anime-label")
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # 如果是檔案路徑，確保 Dockerfile 有 COPY 進去
            db = firestore.Client(database="anime-label")
        else:
            # 如果沒有特定憑證，嘗試預設方式 (例如 GKE, Cloud Run 的服務帳號)
            db = firestore.Client(database="anime-label")
        
        logging.info("INFO: Firestore 客戶端初始化成功。") # 使用 logging 替代 print
    except Exception as e:
        logging.error(f"ERROR: Firestore 客戶端初始化失敗: {e}") # 使用 logging 替代 print
        sys.exit(1)

    logging.info("\n--- 開始從 Firestore 載入情感映射檔案 ---") # 使用 logging 替代 print
    start_mapping_load_time = time.time()

    try:
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            tag_key = data.get('作品分類', doc.id)
            categories = data.get('情感分類')
            if tag_key and isinstance(categories, list):
                TAG_COMBINATION_MAPPING[tag_key] = list(set(categories))
        logging.info("INFO: TAG_COMBINATION_MAPPING 從 Firestore 'anime_label' 集合載入成功。") # 使用 logging 替代 print
    except Exception as e:
        logging.error(f"ERROR: 從 Firestore 'anime_label' 載入 TAG_COMBINATION_MAPPING 失敗: {e}") # 使用 logging 替代 print
        sys.exit(1)

    try:
        emotion_label_docs = db.collection('emotion_label').stream()
        for doc in emotion_label_docs:
            data = doc.to_dict()
            emotion_category_key = data.get('情感分類', doc.id)
            emotions = data.get('情緒')
            if emotion_category_key and isinstance(emotions, list):
                EMOTION_CATEGORY_MAPPING[emotion_category_key] = list(set(emotions))
        logging.info("INFO: EMOTION_CATEGORY_MAPPING 從 Firestore 'emotion_label' 集合載入成功。") # 使用 logging 替代 print
    except Exception as e:
        logging.error(f"ERROR: 從 Firestore 'emotion_label' 載入 EMOTION_CATEGORY_MAPPING 失敗: {e}") # 使用 logging 替代 print
        sys.exit(1)

    end_mapping_load_time = time.time()
    logging.info(f"--- 情感映射檔案從 Firestore 載入完成，總耗時 {end_mapping_load_time - start_mapping_load_time:.4f} 秒 ---\n") # 使用 logging 替代 print


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

# <<<<<<< 新增 API 端點 >>>>>>>
@app.get("/get_emotion_categories")
async def get_emotion_categories():
    """
    提供所有可用的頂層情感分類，供前端的自訂選項使用。
    """
    if not EMOTION_CATEGORY_MAPPING:
        raise HTTPException(status_code=500, detail="情感分類映射未成功載入。")
    return sorted(list(EMOTION_CATEGORY_MAPPING.keys()))


@app.get("/get_emotions")
async def get_emotions_api(
    anime_name: str = Query(..., description="要查詢的動漫名稱"),
    # <<<<<<< 關鍵修改：新增可選的 custom_emotions 參數 >>>>>>>
    custom_emotions: list[str] = Query(None, description="使用者自訂的情感分類列表")
):
    request_start_time = time.time()
    logging.info(f"\n--- 收到搜尋請求: '{anime_name}', 開始處理 ---") # 使用 logging 替代 print

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
            # 確保 '集數' 是字串類型，以匹配 URL 字典的鍵
            df_danmaku['集數'] = df_danmaku['集數'].astype(str)
            logging.info(f"INFO: 從資料庫成功載入 {len(df_danmaku)} 筆 '{normalized_anime_name}' 的彈幕數據。") # 使用 logging 替代 print

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"ERROR: 從資料庫讀取彈幕數據時發生錯誤: {str(e)}") # 使用 logging 替代 print
        raise HTTPException(status_code=500, detail=f"從資料庫讀取彈幕數據時發生錯誤: {str(e)}")

    # <<<<<<< 關鍵修改：根據 custom_emotions 動態生成 emotion_mapping >>>>>>>
    dynamic_emotion_mapping = {}
    if custom_emotions:
        # --- 使用者自訂模式 ---
        logging.info(f"INFO: 使用者自訂模式，選擇的情感分類: {custom_emotions}") # 使用 logging 替代 print
        for category in custom_emotions:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
        if not dynamic_emotion_mapping:
            raise HTTPException(status_code=404, detail=f"您選擇的分類 {custom_emotions} 均無效。")
    else:
        # --- 預設模式 (基於作品分類，合併所有匹配項) ---
        logging.info("INFO: 使用預設模式，根據作品分類生成情感映射。") # 使用 logging 替代 print
        tags = ANIME_TAGS_DB.get(normalized_anime_name, [])
        if not tags:
            raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' 的作品分類數據。")

        anime_tags_set = set(tags)
        # 使用 set 來自動處理合併後重複的情感分類
        collected_emotion_categories = set()

        for mapping_key, categories_list in TAG_COMBINATION_MAPPING.items():
            # 將 "奇幻|冒險" 這樣的 key 轉換成 set
            mapping_tags_set = set(mapping_key.split('|'))
            
            # 確保 mapping_key 不為空，並且是 anime_tags_set 的子集
            if mapping_tags_set and mapping_tags_set.issubset(anime_tags_set):
                # 如果匹配，就把這個 key 對應的所有情感分類都加到 set 中
                logging.info(f"  -> 匹配到標籤組合 '{mapping_key}'，加入情感分類: {categories_list}")
                collected_emotion_categories.update(categories_list)

        if not collected_emotion_categories:
            # 如果遍歷完所有組合後，一個匹配項都找不到
            raise HTTPException(status_code=404, detail=f"找不到作品 '{anime_name}' (作品分類: {tags}) 對應的情感分類定義。")

        # 將集合轉換為列表，以便後續處理
        categories = list(collected_emotion_categories)
        
        for category in categories:
            if category in EMOTION_CATEGORY_MAPPING:
                dynamic_emotion_mapping[category] = EMOTION_CATEGORY_MAPPING[category]
        
        if not dynamic_emotion_mapping:
            raise HTTPException(status_code=404, detail=f"根據作品分類 '{tags}' 組合出的情感分類 ({categories}) 沒有對應的原始情緒詞。")

    # 執行核心情緒分析
    try:
        result = get_top3_emotions_fast(df_danmaku, normalized_anime_name, emotion_mapping=dynamic_emotion_mapping)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"伺服器內部錯誤，情緒分析失敗: {str(e)}")

    if not result:
        raise HTTPException(status_code=404, detail=f"抱歉，沒有找到 '{normalized_anime_name}' 的情緒熱點數據。")

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

    # <<<<<<< 關鍵修改：根據模式決定排序方式 >>>>>>>
    ordered_final_result = {}
    if not custom_emotions:
        # 預設模式：使用優先級排序
        priority_categories = ["TOP 10 彈幕時段", "最精采/激烈的時刻", "虐點/感動"]
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
    
    logging.info(f"--- 請求 '{anime_name}' 處理完成，總耗時: {time.time() - request_start_time:.4f} 秒 ---\n") # 使用 logging 替代 print
    return final_output

# 在生產環境中，這個區塊不會被執行，因為 uvicorn 會直接執行 main:app
# if __name__ == '__main__':
#    logging.info("-----------------------------\n")
#    uvicorn.run("main_json:app", host="0.0.0.0", port=5000, reload=True)


