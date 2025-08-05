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
    calculate_battle_segments: bool = False,
    analysis_window: int = 60, 
    min_gap: int = 240, 
    top_n: int = 5
):
    """
    【最終效能優化版】 - V14 (兩階段精煉版)
    """
    logging.info("\n--- 開始執行最終版高效單次掃描分析 (V14) ---")
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
            "經費", "運鏡", "666", "作畫", "燃", "分鏡", "高能", 
            "外掛", "爆", "炸", "猛", "速度", "流暢", "魄力", "優雅", 
            "BGM", "打鬥", "強","太帥","超帥","星爆", "雞皮疙瘩","頭皮發麻","神仙打架","優秀", "歐拉歐拉歐拉","無駄無駄無駄",
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
    
    # <<< 新增：為了第二階段精煉，預先儲存每集的 NumPy 陣列 >>>
    ep_numpy_data = {}

    # --- 2. 單次滑動窗口掃描 (粗篩) ---
    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        logging.info(f"  -> 正在粗篩第 {ep} 集...")
        
        ep_seconds = group_df['秒數'].to_numpy(dtype=np.int32)
        ep_emotions = group_df['情緒分類'].to_numpy()
        ep_is_battle = group_df['is_battle'].to_numpy() if calculate_battle_segments else None
        ep_is_signin = (group_df['情緒'].to_numpy() == '簽到')
        
        # 儲存 NumPy 陣列以供後續精煉使用
        if calculate_battle_segments and ep_is_battle is not None:
            ep_numpy_data[ep] = {'seconds': ep_seconds, 'is_battle': ep_is_battle}
        
        for t_start in range(0, max_time - analysis_window + 1):
            # ... (此處的粗篩邏輯完全不變) ...
            t_end = t_start + analysis_window
            mask = (ep_seconds >= t_start) & (ep_seconds < t_end)
            total_count = np.sum(mask)
            if total_count < 10: continue
            window_emotions = ep_emotions[mask]
            emotion_counts = Counter(cat for cat in window_emotions if pd.notna(cat))
            for emotion_category, count in emotion_counts.items():
                min_count_threshold = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count_threshold: continue
                rate = count / total_count
                if emotion_category == "LIVE/神配樂" and rate < 0.3: continue
                score = count * rate
                all_highlights[emotion_category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})
            if calculate_battle_segments and ep_is_battle is not None:
                battle_count = np.sum(ep_is_battle[mask])
                if battle_count > 5:
                     all_highlights["精彩的戰鬥/競技片段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})
            density_count = total_count - np.sum(ep_is_signin[mask])
            if density_count > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})

    scan_end_time = time.time()
    logging.info(f"--- 全集粗篩完成，耗時 {scan_end_time - start_time:.2f} 秒 ---")
    
    # --- 3. 結果後處理與精煉 ---
    final_result = {}
    for category, highlights in all_highlights.items():
        if not highlights: continue
        
        highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        
        # ... (選出 top_n 和避免衝突的邏輯不變) ...
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

        # <<< 這裡是唯一的重大修改點 >>>
        # 根據分類決定要處理的列表 (精煉或不精煉)
        processing_list = []
        if category == "精彩的戰鬥/競技片段" and ep_numpy_data:
            logging.info(f"  -> 開始精煉「{category}」的 {len(selected_list)} 個候選片段...")
            for r in selected_list:
                ep_row = r['集數']
                coarse_start = int(r['start_second'])
                
                numpy_data = ep_numpy_data.get(ep_row)
                if not numpy_data:
                    processing_list.append(r) # 若找不到數據，使用原始結果
                    continue

                # 1. 建立 60 秒粗篩窗口的遮罩
                mask_60s = (numpy_data['seconds'] >= coarse_start) & (numpy_data['seconds'] < coarse_start + 60)
                
                # 2. 獲取此 60 秒內所有戰鬥彈幕的時間戳
                battle_ts_in_window = numpy_data['seconds'][mask_60s & numpy_data['is_battle']]
                
                best_30s_start = coarse_start # 預設值
                max_30s_count = -1

                # 3. 在 60 秒內滑動 30 秒窗口進行精煉
                if battle_ts_in_window.size > 0:
                    # 滑動窗口的起點從 coarse_start 到 coarse_start + 30
                    for sub_window_start in range(coarse_start, coarse_start + 46):
                        sub_window_end = sub_window_start + 45
                        current_count = np.sum((battle_ts_in_window >= sub_window_start) & (battle_ts_in_window < sub_window_end))
                        if current_count > max_30s_count:
                            max_30s_count = current_count
                            best_30s_start = sub_window_start
                
                # 儲存精煉後的結果
                refined_r = r.copy()
                refined_r['start_second'] = best_30s_start
                processing_list.append(refined_r)
        else:
            # 其他所有分類直接使用原始的 selected_list
            processing_list = selected_list

        # --- 格式化輸出 ---
        def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        def format_episode(ep):
            try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
            except (ValueError, TypeError): return str(ep)
    
        output_list = []
        # 現在所有片段的時長都由這個統一邏輯決定
        final_window = 30 if category != "TOP 10 彈幕時段" else analysis_window
        
        for r in processing_list: # 使用 processing_list 進行最終輸出
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
    
    logging.info(f"--- 全部分析與精煉完成，總耗時 {time.time() - start_time:.2f} 秒 ---")
    return final_result







