import pandas as pd
import numpy as np
import io
import time
import json
import logging
from collections import defaultdict, Counter

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_highlights_single_pass(
    df: pd.DataFrame, 
    anime_name: str, 
    emotion_mapping: dict,
    analysis_window: int = 60, 
    min_gap: int = 240, 
    top_n: int = 5
):
    """
    【最終效能優化版】 - V15
    對數據預處理進行全面向量化，取代 apply()，大幅提升效能。
    """
    logging.info("\n--- 開始執行最終版高效單次掃描分析 (V15) ---")
    start_time = time.time()

    # --- 1. 數據預處理 (全面向量化) ---
    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty: return {}

    # <<<<<<<<<<<<<<< 效能優化點 1: 向量化時間轉換 >>>>>>>>>>>>>>>
    time_parts = df_anime['時間'].str.split(':', expand=True).astype(int)
    df_anime['秒數'] = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
    
    BATTLE_KEYWORDS = [
        "經費", "帥", "運鏡", "666", "作畫", "燃", "神", "分鏡", "高能", 
        "外掛", "爆", "炸", "猛", "777", "速度", "流暢", "魄力", "優雅", 
        "BGM", "打鬥", "強"
    ]
    keyword_regex = '|'.join(BATTLE_KEYWORDS)
    df_anime['is_battle'] = df_anime['彈幕'].str.contains(keyword_regex, na=False)

    # <<<<<<<<<<<<<<< 效能優化點 2: 向量化情緒分類 >>>>>>>>>>>>>>>
    # 創建一個從 '情緒' -> '情感分類' 的反向映射字典
    emotion_to_category_map = {emotion: category 
                               for category, emotions in emotion_mapping.items() 
                               for emotion in emotions}
    # 使用高效的 .map() 方法進行分類
    df_anime['情緒分類'] = df_anime['情緒'].map(emotion_to_category_map)
    
    all_highlights = defaultdict(list)
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()

    preprocess_end_time = time.time()
    logging.info(f"--- 數據預處理完成，耗時 {preprocess_end_time - start_time:.2f} 秒 ---")

    # --- 2. 單次滑動窗口掃描 (NumPy 高效版) ---
    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        logging.info(f"  -> 正在掃描第 {ep} 集...")
        
        ep_seconds = group_df['秒數'].to_numpy(dtype=np.int32)
        ep_emotions = group_df['情緒分類'].to_numpy()
        ep_is_battle = group_df['is_battle'].to_numpy()
        ep_is_signin = (group_df['情緒'].to_numpy() == '簽到')
        
        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            mask = (ep_seconds >= t_start) & (ep_seconds < t_end)
            total_count = np.sum(mask)
            
            if total_count < 10:
                continue

            window_emotions = ep_emotions[mask]
            emotion_counts = Counter(cat for cat in window_emotions if pd.notna(cat))

            for emotion_category, count in emotion_counts.items():
                min_count_threshold = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count_threshold: continue
                rate = count / total_count
                if emotion_category == "LIVE/神配樂" and rate < 0.3: continue
                score = count * rate
                all_highlights[emotion_category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})
            
            battle_count = np.sum(ep_is_battle[mask])
            if battle_count > 5:
                 all_highlights["精彩的戰鬥時段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

            density_count = total_count - np.sum(ep_is_signin[mask])
            if density_count > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})

    scan_end_time = time.time()
    logging.info(f"--- 全集掃描完成，耗時 {scan_end_time - preprocess_end_time:.2f} 秒 ---")
    
    # --- 3. 結果後處理 ---
    final_result = {}
    for category, highlights in all_highlights.items():
        if not highlights: continue
        
        highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        
        selected_list = []
        if category == "TOP 10 彈幕時段":
            current_top_n = 10
        elif "劇情高潮" in category:
            current_top_n = 10
        elif "虐點/感動" in category:
            current_top_n = 7
        elif category == "精彩的戰鬥時段":
            current_top_n = 7
        else:
            current_top_n = 5
      
        episode_quota_tracker = defaultdict(int)
    
        for _, row in highlights_df.iterrows():
            if len(selected_list) >= current_top_n: break
            
            ep_row = row['集數']
            if episode_quota_tracker[ep_row] >= 2 and category not in ["精彩的戰鬥時段", "TOP 10 彈幕時段"]:
                continue
    
            is_conflict = any(r['集數'] == ep_row and abs(r['start_second'] - row['start_second']) < min_gap for r in selected_list)
            if not is_conflict:
                selected_list.append(row.to_dict())
                episode_quota_tracker[ep_row] += 1
    
        if not selected_list: continue
    
        def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        def format_episode(ep):
            try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
            except (ValueError, TypeError): return str(ep)
    
        output_list = []
        final_window = 30 if category not in ["精彩的戰鬥時段", "TOP 10 彈幕時段"] else analysis_window
        
        for r in selected_list:
            item = {
                '集數': format_episode(r['集數']),
                '時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + final_window)}",
                'start_second': int(r['start_second'])
            }
            if 'rate' in r:
                item['熱度分數'] = round(r['score'], 2)
                item['彈幕數量'] = int(r['count'])
                item['情緒佔比'] = f"{r['rate']:.1%}"
            else:
                item['彈幕數量'] = int(r['score'])
            output_list.append(item)
            
        final_result[category] = output_list
    
    logging.info(f"--- 全部分析完成，總耗時 {time.time() - start_time:.2f} 秒 ---")
    return final_result

# 測試用區塊
if __name__ == '__main__':
    pass











