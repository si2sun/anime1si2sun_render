import pandas as pd
import numpy as np
import io
import time
import json
import logging
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_highlights_single_pass(
    df: pd.DataFrame, 
    anime_name: str, 
    emotion_mapping: dict,
    # <<< 關鍵修改 1：新增參數，預設為 False >>>
    calculate_battle_segments: bool = False,
    analysis_window: int = 60, 
    min_gap: int = 240, 
    top_n: int = 5
):
    """
    【最終效能優化版】 - V14
    結合單次掃描架構與 NumPy 高速陣列運算，並支援可選的戰鬥時段分析。
    """
    logging.info("\n--- 開始執行最終版高效單次掃描分析 (V14) ---")
    start_time = time.time()

    # --- 1. 數據預處理 (不變) ---
    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty: return {}

    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except: return 0
    df_anime['秒數'] = df_anime['時間'].apply(time_to_seconds)
    
    # 僅在需要時才建立 is_battle 欄位，節省資源
    if calculate_battle_segments:
        BATTLE_KEYWORDS = [
            "經費", "運鏡", "666", "作畫", "燃", "分鏡", "高能", 
            "外掛", "爆", "炸", "猛", "777", "速度", "流暢", "魄力", "優雅", 
            "BGM", "打鬥", "強","太帥","超帥","星爆", "雞皮疙瘩","頭皮發麻","神仙打架","優秀",
            "畫面","帥啊","前方","名場面","精采"
        ]
        keyword_regex = '|'.join(BATTLE_KEYWORDS)
        df_anime['is_battle'] = df_anime['彈幕'].str.contains(keyword_regex, na=False)

    def classify_emotion(e):
        for cat, e_list in emotion_mapping.items():
            if e in e_list: return cat
        return None
    df_anime['情緒分類'] = df_anime['情緒'].apply(classify_emotion)
    
    all_highlights = defaultdict(list)
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()

    # --- 2. 單次滑動窗口掃描 (NumPy 高效版) ---
    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        logging.info(f"  -> 正在掃描第 {ep} 集...")
        
        ep_seconds = group_df['秒數'].to_numpy(dtype=np.int32)
        ep_emotions = group_df['情緒分類'].to_numpy()
        # 僅在需要時才讀取 is_battle 數據
        ep_is_battle = group_df['is_battle'].to_numpy() if calculate_battle_segments else None
        ep_is_signin = (group_df['情緒'].to_numpy() == '簽到')
        
        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            mask = (ep_seconds >= t_start) & (ep_seconds < t_end)
            total_count = np.sum(mask)
            
            if total_count < 10:
                continue

            # a) 計算情感分類熱度 (不變)
            window_emotions = ep_emotions[mask]
            emotion_counts = Counter(cat for cat in window_emotions if pd.notna(cat))
            for emotion_category, count in emotion_counts.items():
                # ... (此處邏輯不變) ...
                min_count_threshold = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count_threshold: continue
                rate = count / total_count
                if emotion_category == "LIVE/神配樂" and rate < 0.3: continue
                score = count * rate
                all_highlights[emotion_category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})
            
            # <<< 關鍵修改 2：只有在指令為 True 時才計算 >>>
            if calculate_battle_segments and ep_is_battle is not None:
                battle_count = np.sum(ep_is_battle[mask])
                if battle_count > 5:
                     all_highlights["精彩的戰鬥/競技片段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

            # c) 計算 TOP 彈幕密度 (不變，永遠執行)
            density_count = total_count - np.sum(ep_is_signin[mask])
            if density_count > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})

    scan_end_time = time.time()
    logging.info(f"--- 全集掃描完成，耗時 {scan_end_time - start_time:.2f} 秒 ---")
    
    # --- 3. 結果後處理 ---
    final_result = {}
    for category, highlights in all_highlights.items():
        if not highlights: continue
        
        highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        
        selected_list = []
        if category == "TOP 10 彈幕時段":
            current_top_n = 10
        elif "劇情高潮/震撼" in category:
            current_top_n = 10
        elif "虐點/感動" in category:
            current_top_n = 7
        elif category in "精彩的戰鬥/競技片段":
            current_top_n = 7
        elif "LIVE/神配樂" in category:
            current_top_n = 7
        else: # 其他所有情感分類的預設值
            current_top_n = 5
      
        episode_quota_tracker = defaultdict(int)
    
        for _, row in highlights_df.iterrows():
            if len(selected_list) >= current_top_n: break
            
            ep_row = row['集數']
            if episode_quota_tracker[ep_row] >= 2 and category not in ["精彩的戰鬥/競技片段", "TOP 10 彈幕時段"]:
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
        final_window = 30 if category not in ["精彩的戰鬥/競技片段", "TOP 10 彈幕時段"] else analysis_window
        
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
    # 這裡可以放置您的測試數據和呼叫邏輯，以便獨立測試此檔案
    pass






