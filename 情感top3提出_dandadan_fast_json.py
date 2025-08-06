import pandas as pd
import numpy as np
import io
import time
import json
import logging
from collections import defaultdict, Counter
from io import StringIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_highlights_single_pass(
    df: pd.DataFrame, 
    anime_name: str, 
    emotion_mapping: dict,
    calculate_battle_segments: bool = False,
    analysis_window: int = 60, 
    min_gap: int = 240, 
    top_n: int = 5
):
    """
    【最終修正版】 - V15 (獨立計算)
    確保每個情感分類的計算過程完全獨立，不受其他分類影響。
    """
    logging.info(f"\n--- 開始執行分析 (戰鬥時段分析: {'啟用' if calculate_battle_segments else '禁用'}) ---")
    start_time = time.time()

    # --- 1. 數據預處理 ---
    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty: return {}

    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except: return 0
    df_anime['秒數'] = df_anime['時間'].apply(time_to_seconds)
    
    if calculate_battle_segments:
        BATTLE_KEYWORDS = [
            "經費", "運鏡", "66666", "作畫", "燃", "分鏡", "高能", 
            "外掛", "爆", "炸", "猛", "速度", "流暢", "魄力", "優雅", 
            "BGM", "打鬥", "強","太帥","超帥","星爆", "雞皮疙瘩","頭皮發麻","神仙打架","優秀", "歐拉歐拉歐拉","無駄無駄無駄",
            "畫面","帥啊","前方","名場面","精采", "atomic", "Explosion", "Excalibur", "領域展開",
            "地爆天星","神羅天征", "Enuma", "c8763"
        ]
        keyword_regex = '|'.join(BATTLE_KEYWORDS)
        df_anime['is_battle'] = df_anime['彈幕'].str.contains(keyword_regex, na=False)

    # <<< 關鍵修改 1：不再需要 classify_emotion 和 '情緒分類' 欄位 >>>
    
    all_highlights = defaultdict(list)
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()
    ep_numpy_data = {}

    # --- 2. 單次滑動窗口掃描 ---
    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        logging.info(f"  -> 正在掃描第 {ep} 集...")
        
        ep_seconds = group_df['秒數'].to_numpy(dtype=np.int32)
        ep_is_battle = group_df['is_battle'].to_numpy() if calculate_battle_segments else None
        ep_is_signin = (group_df['情緒'].to_numpy() == '簽到')
        # 直接使用原始的情緒數據
        ep_raw_emotions = group_df['情緒'].to_numpy()
        
        if calculate_battle_segments and ep_is_battle is not None:
            ep_numpy_data[ep] = {'seconds': ep_seconds, 'is_battle': ep_is_battle}
        
        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            time_mask = (ep_seconds >= t_start) & (ep_seconds < t_end)
            total_count = np.sum(time_mask)
            
            if total_count < 10: continue

            # <<< 關鍵修改 2：在這裡循環每個分類，進行獨立計算 >>>
            window_raw_emotions = ep_raw_emotions[time_mask]

            # a) 計算情感分類熱度
            for category, emotion_list in emotion_mapping.items():
                # 為當前分類建立獨立的情緒遮罩
                emotion_mask = np.isin(window_raw_emotions, emotion_list)
                count = np.sum(emotion_mask)

                min_count_threshold = 7 if "虐點/感動" in category or "劇情高潮" in category else 5
                if count < min_count_threshold: continue

                rate = count / total_count
                if category == "LIVE/神配樂" and rate < 0.3: continue
                
                score = count * rate
                all_highlights[category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})

            # b) 計算精彩戰鬥時段 (邏輯不變)
            if calculate_battle_segments and ep_is_battle is not None:
                battle_count = np.sum(ep_is_battle[time_mask])
                if battle_count > 5:
                     all_highlights["精彩的戰鬥/競技片段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

            # c) 計算 TOP 彈幕密度 (邏輯不變)
            density_count = total_count - np.sum(ep_is_signin[time_mask])
            if density_count > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})
    
    # --- 3. 結果後處理與精煉 (此部分邏輯完全不變) ---
    final_result = {}
    # ... (此處的後處理、精煉、格式化邏輯與上一版完全相同，故省略) ...
    for category, highlights in all_highlights.items():
        if not highlights: continue
        highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        selected_list = []
        if category == "TOP 10 彈幕時段": current_top_n = 10
        elif "劇情高潮/震撼" in category: current_top_n = 10
        elif "虐點/感動" in category: current_top_n = 7
        elif category == "精彩的戰鬥/競技片段": current_top_n = 7
        elif "LIVE/神配樂" in category: current_top_n = 7
        else: current_top_n = 5
        episode_quota_tracker = defaultdict(int)
        for _, row in highlights_df.iterrows():
            if len(selected_list) >= current_top_n: break
            ep_row = row['集數']
            if episode_quota_tracker[ep_row] >= 2 and category not in ["精彩的戰鬥/競技片段", "TOP 10 彈幕時段"]: continue
            is_conflict = any(r['集數'] == ep_row and abs(r['start_second'] - row['start_second']) < min_gap for r in selected_list)
            if not is_conflict:
                selected_list.append(row.to_dict())
                episode_quota_tracker[ep_row] += 1
        if not selected_list: continue
        processing_list = []
        if category == "精彩的戰鬥/競技片段" and ep_numpy_data:
            for r in selected_list:
                ep_row = r['集數']; coarse_start = int(r['start_second'])
                numpy_data = ep_numpy_data.get(ep_row)
                if not numpy_data:
                    processing_list.append(r); continue
                mask_60s = (numpy_data['seconds'] >= coarse_start) & (numpy_data['seconds'] < coarse_start + 60)
                battle_ts_in_window = numpy_data['seconds'][mask_60s & numpy_data['is_battle']]
                refined_window_size = 45; best_start = coarse_start; max_count = -1
                if battle_ts_in_window.size > 0:
                    for sub_window_start in range(coarse_start, coarse_start + (60 - refined_window_size) + 1):
                        sub_window_end = sub_window_start + refined_window_size
                        current_count = np.sum((battle_ts_in_window >= sub_window_start) & (battle_ts_in_window < sub_window_end))
                        if current_count > max_count:
                            max_count = current_count; best_start = sub_window_start
                refined_r = r.copy(); refined_r['start_second'] = best_start
                processing_list.append(refined_r)
        else:
            processing_list = selected_list
        def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        def format_episode(ep):
            try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
            except (ValueError, TypeError): return str(ep)
        output_list = []
        if category == "精彩的戰鬥/競技片段": final_window = 45
        elif category == "TOP 10 彈幕時段": final_window = analysis_window
        else: final_window = 30
        for r in processing_list:
            item = {'集數': format_episode(r['集數']),'時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + final_window)}",'start_second': int(r['start_second'])}
            if 'rate' in r:
                item['熱度分數'] = round(r['score'], 2); item['彈幕數量'] = int(r['count']); item['情緒佔比'] = f"{r['rate']:.1%}"
            else:
                item['彈幕數量'] = int(r['score'])
            output_list.append(item)
        final_result[category] = output_list
    
    logging.info(f"--- 全部分析與精煉完成，總耗時 {time.time() - start_time:.2f} 秒 ---")
    return final_result
