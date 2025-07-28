import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import json
import unicodedata
import time
import traceback
import psycopg2
from psycopg2 import sql
from contextlib import contextmanager
import io

# 確保這些導入是正確且不重複的
import google.auth
from google.cloud import firestore
from google.oauth2 import service_account # <-- 這是關鍵的修改
from google.cloud.firestore_v1.base_client import BaseClient

# 從同級目錄導入 情感top3提出_dandadan_fast 模組
try:
    from 情感top3提出_dandadan_fast_json import get_top3_emotions_fast, get_top5_density_moments
except ImportError:
    print("ERROR: 無法導入 '情感top3提出_dandadan_fast_json' 模組中的函式。")
    sys.exit(1)

app = FastAPI()

# ====== CORS 配置 ======
# 生產環境允許所有來源，但限制特定方法和標頭
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境可改為具體域名
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=".")

# ====== PostgreSQL 資料庫配置 ======
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

@contextmanager
def get_db_connection():
    """建立並管理 PostgreSQL 資料庫連接的上下文管理器。"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        yield conn
    finally:
        if conn:
            conn.close()

# 全域變數，用於儲存已載入的動漫數據和 URL
AVAILABLE_ANIME_NAMES = []
YOUTUBE_ANIME_EPISODE_URLS = {} # {動畫名: {集數: video_id}}
BAHAMUT_ANIME_EPISODE_URLS = {} # {動畫名: {集數: url}}
ANIME_COVER_IMAGE_URLS = {} # {動畫名: 封面圖URL}
ANIME_TAGS_DB = {} # {動畫名: [tag1, tag2]}

# Firestore 客戶端
db: BaseClient = None # 在 startup_event 中初始化

# 情感映射
TAG_COMBINATION_MAPPING = {} # {作品分類: [情感分類1, 情感分類2]}
EMOTION_CATEGORY_MAPPING = {} # {情感分類: [情緒1, 情緒2]}

def load_anime_data_from_db():
    print("\n--- 開始從 PostgreSQL 載入動漫數據 ---")
    start_time = time.time()
    global AVAILABLE_ANIME_NAMES, YOUTUBE_ANIME_EPISODE_URLS, BAHAMUT_ANIME_EPISODE_URLS, ANIME_COVER_IMAGE_URLS, ANIME_TAGS_DB
    
    # 徹底移除 "ED開始秒數" 的查询
    query = 'SELECT "作品名", "集數", "巴哈動畫瘋網址", "YT網址", "封面圖", "作品分類" FROM anime_url ORDER BY "作品名", "集數";'
    
    temp_available_anime_names = set() # 用 set 避免重複
    
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        rows = cur.fetchall()

    if not rows:
        print("⚠️ 警告：資料庫的 'anime_url' 表中沒有找到任何數據。")
        return

    for row in rows:
        anime_original, episode, bahamut_url, youtube_url, cover_image_val, tags_json = row
        
        anime_normalized = unicodedata.normalize('NFC', str(anime_original).strip())
        temp_available_anime_names.add(anime_normalized)
        
        # ====== 修改這裡：處理集數的標準化 ======
        ep_key_raw = episode
        ep_key = ""
        if ep_key_raw is not None:
            try:
                # 嘗試將浮點數轉換為整數，然後再轉為字串，確保為 "1" 而非 "1.0"
                ep_key = str(int(float(ep_key_raw))).strip()
            except (ValueError, TypeError):
                # 如果轉換失敗 (例如本身就是字串或非數字)，則直接轉字串
                ep_key = str(ep_key_raw).strip()
        # =========================================

        YOUTUBE_ANIME_EPISODE_URLS.setdefault(anime_normalized, {})
        BAHAMUT_ANIME_EPISODE_URLS.setdefault(anime_normalized, {})
        ANIME_TAGS_DB.setdefault(anime_normalized, [])

        if youtube_url:
            yt_url_str = str(youtube_url).strip()
            video_id = None
            # 修改 YT 網址解析邏輯以適應您的數據格式
            # 假設您的 YT 網址格式可能為 https://www.youtube.com/watch?v=VIDEO_ID 或其他
            if "youtube.com/watch?v=" in yt_url_str:
                video_id = yt_url_str.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in yt_url_str:
                video_id = yt_url_str.split("youtu.be/")[-1].split("?")[0]
            # 處理 Googleusercontent 代理的 URL
            elif "youtube.com/watch?v=" in yt_url_str:
                video_id = yt_url_str.split("v=")[-1].split("&")[0]
            elif "youtu.be/" in yt_url_str:
                video_id = yt_url_str.split("youtu.be/")[-1].split("?")[0]

            if video_id:
                YOUTUBE_ANIME_EPISODE_URLS[anime_normalized][ep_key] = video_id
        
        if bahamut_url: BAHAMUT_ANIME_EPISODE_URLS[anime_normalized][ep_key] = str(bahamut_url).strip()
        if cover_image_val: ANIME_COVER_IMAGE_URLS[anime_normalized] = str(cover_image_val).strip()
        if tags_json:
            try:
                tags = json.loads(str(tags_json).replace("'", '"'))
                if isinstance(tags, list): ANIME_TAGS_DB[anime_normalized] = tags
            except (json.JSONDecodeError, TypeError): pass

    AVAILABLE_ANIME_NAMES = sorted(list(temp_available_anime_names))
    print(f"--- PostgreSQL 數據載入完成，耗時 {time.time() - start_time:.2f} 秒 ---")

@app.on_event("startup")
async def startup_event():
    print(f"伺服器啟動中...")
    load_anime_data_from_db()

    global db, TAG_COMBINATION_MAPPING, EMOTION_CATEGORY_MAPPING
    try:
        firestore_credentials_json = os.getenv("FIRESTORE_CREDENTIALS_JSON")
        if firestore_credentials_json:
            credentials_info = json.loads(firestore_credentials_json)
            # ====== 關鍵修改在這裡 ======
            db = firestore.Client(database="anime-label", credentials=service_account.Credentials.from_service_account_info(credentials_info))
            # ==========================
            print("INFO: Firestore 客戶端初始化成功 (從環境變數)。")
        else:
            print("ERROR: 未找到 FIRESTORE_CREDENTIALS_JSON 環境變數。請在部署環境中設定此變數。")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Firestore 客戶端初始化失敗: {e}"); traceback.print_exc(); sys.exit(1) # 添加 traceback 方便偵錯

    print("\n--- 開始從 Firestore 載入情感映射檔案 ---")
    start_time = time.time()
    try:
        # 載入 anime_label collection
        anime_label_docs = db.collection('anime_label').stream()
        for doc in anime_label_docs:
            data = doc.to_dict()
            if '作品分類' in data and '情感分類' in data and isinstance(data['情感分類'], list):
                TAG_COMBINATION_MAPPING[data['作品分類']] = list(set(data['情感分類']))
        
        # 載入 emotion_label collection
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
async def read_root(request: Request):
    """主頁面，顯示動漫搜尋介面。"""
    return templates.TemplateResponse("animetop.html", {"request": request})

@app.get("/search_anime_names")
async def search_anime_names(query: str = Query(..., min_length=1)):
    """提供動漫名稱的自動完成建議。"""
    normalized_query = unicodedata.normalize('NFC', query.strip().lower())
    suggestions = [
        name for name in AVAILABLE_ANIME_NAMES
        if normalized_query in unicodedata.normalize('NFC', name.lower())
    ]
    return {"suggestions": sorted(list(set(suggestions)))}

@app.get("/anime_danmaku/{anime_name}/{episode_number_str}", response_model=dict)
async def get_anime_danmaku(
    anime_name: str,
    episode_number_str: str, # 接收字串，以便處理 "2f" 這種情況
    analysis_window: int = Query(60, description="彈幕分析的窗口大小（秒）"),
    min_gap: int = Query(300, description="不同熱點之間的最小間隔（秒）"),
    top_n: int = Query(5, description="每個情感類別要顯示的熱點數量"),
    custom_mode: bool = Query(False, description="是否啟用自訂排序模式")
):
    """
    根據動漫名稱和集數，獲取情感分析熱點。
    """
    normalized_name = unicodedata.normalize('NFC', anime_name.strip())
    
    # 針對集數進行標準化，確保查找時格式一致
    ep_key = ""
    if episode_number_str is not None:
        try:
            ep_key = str(int(float(episode_number_str))).strip()
        except (ValueError, TypeError):
            ep_key = episode_number_str.strip() # 如果無法轉為數字，則保留原始字串 (例如 "2f")

    print(f"\n[GET /anime_danmaku/{normalized_name}/{ep_key}] 請求接收，參數: window={analysis_window}, gap={min_gap}, top_n={top_n}, custom_mode={custom_mode}")
    
    # 檢查動漫是否存在
    if normalized_name not in AVAILABLE_ANIME_NAMES:
        print(f"  [錯誤] 動漫 '{normalized_name}' 不存在於資料庫中。")
        raise HTTPException(status_code=404, detail=f"動漫 '{anime_name}' 不存在。")

    # 檢查集數是否存在於 URL 數據中 (用於提供 URL)
    yt_url = YOUTUBE_ANIME_EPISODE_URLS.get(normalized_name, {}).get(ep_key)
    bahamut_url = BAHAMUT_ANIME_EPISODE_URLS.get(normalized_name, {}).get(ep_key)

    # 從資料庫獲取彈幕數據
    danmaku_data = []
    t_db_start = time.time()
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            # 使用 parameterized query 防止 SQL 注入
            cur.execute(
                sql.SQL('SELECT "彈幕", "label", "label2", "作品名", "集數", "時間", "情緒" FROM danmaku_data WHERE "作品名" = %s AND "集數" = %s;')
                .format(table=sql.Identifier('danmaku_data')),
                (anime_original, episode_number_str) # 這裡使用原始的 episode_number_str，因為 DB 裡可能存的是原始值
            )
            rows = cur.fetchall()
            if not rows:
                print(f"  [警告] 動漫 '{normalized_name}' 第 {ep_key} 集沒有彈幕數據。")
                # 即使沒有彈幕數據，也返回基礎信息
                return {
                    "anime_name": normalized_name,
                    "episode": ep_key,
                    "youtube_episode_urls": yt_url,
                    "bahamut_episode_urls": bahamut_url,
                    "cover_image_url": ANIME_COVER_IMAGE_URLS.get(normalized_name),
                    "tags": ANIME_TAGS_DB.get(normalized_name, []),
                    "emotional_hotspots": {},
                    "message": "沒有找到該集數的彈幕數據。"
                }
            
            # 將查詢結果轉換為列表，然後創建 DataFrame
            df_columns = ["彈幕", "label", "label2", "作品名", "集數", "時間", "情緒"]
            danmaku_data = pd.DataFrame(rows, columns=df_columns)
            
    except Exception as e:
        print(f"  [錯誤] 從資料庫獲取彈幕數據失敗: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="無法從資料庫獲取彈幕數據。")
    t_db_end = time.time()
    print(f"  [計時] 從資料庫獲取彈幕數據: {t_db_end - t_db_start:.4f} 秒")
    
    # 情感分析
    result = {}
    top_5_moments = []
    
    # 使用從 Firestore 載入的映射
    emotion_mapping = TAG_COMBINATION_MAPPING.get(ANIME_TAGS_DB.get(normalized_name, ["未知分類"])[0], {})
    final_emotion_mapping = {}
    for tag_key, emotion_categories in emotion_mapping.items():
        for category in emotion_categories:
            if category in EMOTION_CATEGORY_MAPPING:
                final_emotion_mapping.setdefault(tag_key, []).extend(EMOTION_CATEGORY_MAPPING[category])

    # 執行情感熱點分析
    t_emotions_start = time.time()
    try:
        result = get_top3_emotions_fast(
            df=danmaku_data,
            anime_name=normalized_name,
            emotion_mapping=final_emotion_mapping,
            analysis_window=analysis_window,
            min_gap=min_gap,
            top_n=top_n
        )
    except Exception as e:
        print(f"  [錯誤] 情感熱點分析失敗: {e}")
        traceback.print_exc()
        result = {} # 即使失敗也確保 result 是空的字典
    t_emotions_end = time.time()
    print(f"  [計時] 情感熱點分析: {t_emotions_end - t_emotions_start:.4f} 秒")

    # 執行 TOP N 彈幕密度時段計算
    t_top5_start = time.time()
    try:
        top_5_moments = get_top5_density_moments(
            df=danmaku_data,
            anime_name=normalized_name
        )
    except Exception as e:
        print(f"  [錯誤] 彈幕密度時段計算失敗: {e}")
        traceback.print_exc()
        top_5_moments = [] # 即使失敗也確保 top_5_moments 是空的列表
    t_top5_end = time.time()
    print(f"  [計時] TOP 10 弹幕时段计算: {t_top5_end - t_top5_start:.4f} 秒")
    
    t_sort_start = time.time()
    ordered_final_result = {}
    if not custom_mode:
        priority_top = ["最精采/激烈的時刻", "LIVE/配樂", "虐點/感動"]
        priority_bottom_key = "彈幕最密集 TOP10"
        
        final_ordered_keys = []
        for key in priority_top:
            if key in result:
                final_ordered_keys.append(key)
        
        other_categories = sorted([
            key for key in result 
            if key not in priority_top
        ])
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
        "anime_name": normalized_name,
        "episode": ep_key, # 返回標準化後的集數鍵
        "youtube_episode_urls": yt_url,
        "bahamut_episode_urls": bahamut_url,
        "cover_image_url": ANIME_COVER_IMAGE_URLS.get(normalized_name),
        "tags": ANIME_TAGS_DB.get(normalized_name, []),
        "emotional_hotspots": ordered_final_result,
        "message": "數據已成功載入並分析。"
    }
    print(f"[GET /anime_danmaku/{normalized_name}/{ep_key}] 請求處理完成。")
    return final_output

if __name__ == "__main__":
    # 在本地運行時，確保有默認值或者在 .env 檔案中設定環境變數
    # 如果本地測試需要用到 Firestore，請確保 animetext-anime1si2sun.json 在根目錄，
    # 或者手動設定 os.environ["FIRESTORE_CREDENTIALS_JSON"]
    # For local development, you might want to load .env file
    # from dotenv import load_dotenv
    # load_dotenv()

    uvicorn.run(app, host="0.0.0.0", port=8000)
