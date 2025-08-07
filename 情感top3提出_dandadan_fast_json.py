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
    【最終畢業版】 - V16 (通用兩階段精煉)
    所有情感與戰鬥分類都採用60秒粗篩->濃縮精煉的模式，以獲得最高質量的亮點。
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

    all_highlights = defaultdict(list)
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()
    
    # <<< 關鍵修改 1：擴展 ep_numpy_data，儲存所有需要精煉的原始數據 >>>
    ep_numpy_data = {}

    # --- 2. 單次滑動窗口掃描 (粗篩) ---
    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        logging.info(f"  -> 正在粗篩第 {ep} 集...")
        
        ep_seconds = group_df['秒數'].to_numpy(dtype=np.int32)
        ep_raw_emotions = group_df['情緒'].to_numpy()
        ep_is_signin = (ep_raw_emotions == '簽到')
        
        # 儲存這一集的所有 NumPy 陣列，以供後續精煉使用
        ep_numpy_data[ep] = {'seconds': ep_seconds, 'emotions': ep_raw_emotions}
        if calculate_battle_segments:
            ep_numpy_data[ep]['is_battle'] = group_df['is_battle'].to_numpy()

        for t_start in range(0, max_time - analysis_window + 1):
            time_mask = (ep_seconds >= t_start) & (ep_seconds < t_start + analysis_window)
            total_count = np.sum(time_mask)
            if total_count < 10: continue

            # a) 獨立計算每個情感分類的熱度分數
            window_raw_emotions = ep_raw_emotions[time_mask]
            for category, emotion_list in emotion_mapping.items():
                emotion_mask = np.isin(window_raw_emotions, emotion_list)
                count = np.sum(emotion_mask)
                min_count_threshold = 7 if "虐點/感動" in category or "劇情高潮" in category else 5
                if count < min_count_threshold: continue
                rate = count / total_count
                if category == "LIVE/神配樂" and rate < 0.3: continue
                score = count * rate
                all_highlights[category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})

            # b) 計算戰鬥時段熱度
            if calculate_battle_segments and 'is_battle' in ep_numpy_data[ep]:
                battle_count = np.sum(ep_numpy_data[ep]['is_battle'][time_mask])
                if battle_count > 5:
                     all_highlights["精彩的戰鬥/競技片段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

            # c) 計算彈幕密度
            density_count = total_count - np.sum(ep_is_signin[time_mask])
            if density_count > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})
    
    scan_end_time = time.time()
    logging.info(f"--- 全集粗篩完成，耗時 {scan_end_time - start_time:.2f} 秒 ---")

    # --- 3. 結果後處理與通用精煉 ---
    final_result = {}
    for category, highlights in all_highlights.items():
        if not highlights: continue
        
        highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        
        # ... (選出 top_n 和避免衝突的邏輯不變) ...
        selected_list = []; episode_quota_tracker = defaultdict(int)
        if category == "TOP 10 彈幕時段": current_top_n = 10
        elif "劇情高潮/震撼" in category: current_top_n = 10
        elif "虐點/感動" in category: current_top_n = 7
        elif category == "精彩的戰鬥/競技片段": current_top_n = 7
        elif "LIVE/神配樂" in category: current_top_n = 7
        else: current_top_n = 5
        for _, row in highlights_df.iterrows():
            if len(selected_list) >= current_top_n: break
            ep_row = row['集數']
            if episode_quota_tracker[ep_row] >= 2 and category not in ["精彩的戰鬥/競技片段", "TOP 10 彈幕時段"]: continue
            is_conflict = any(r['集數'] == ep_row and abs(r['start_second'] - row['start_second']) < min_gap for r in selected_list)
            if not is_conflict:
                selected_list.append(row.to_dict()); episode_quota_tracker[ep_row] += 1
        if not selected_list: continue

        # <<< 關鍵修改 2：通用精煉邏輯 >>>
        processing_list = []
        if category == "TOP 10 彈幕時段":
            # 彈幕密度不需精煉
            processing_list = selected_list
        else:
            # 所有其他分類（戰鬥和情感）都需要精煉
            logging.info(f"  -> 開始精煉「{category}」的 {len(selected_list)} 個候選片段...")
            
            # 根據分類決定精煉時長
            refined_window_size = 45 if category == "精彩的戰鬥/競技片段" else 30

            for r in selected_list:
                ep_row = r['集數']; coarse_start = int(r['start_second'])
                numpy_data = ep_numpy_data.get(ep_row)
                if not numpy_data:
                    processing_list.append(r); continue
                
                # 在60秒粗篩窗口內，找出所有相關的彈幕時間點
                window_mask_60s = (numpy_data['seconds'] >= coarse_start) & (numpy_data['seconds'] < coarse_start + 60)
                
                ts_to_check = np.array([])
                if category == "精彩的戰鬥/競技片段":
                    if 'is_battle' in numpy_data:
                        battle_mask = numpy_data['is_battle'][window_mask_60s]
                        ts_to_check = numpy_data['seconds'][window_mask_60s][battle_mask]
                else:
                    # 這是情感分類，找出對應的情緒詞
                    emotion_keywords = emotion_mapping.get(category, [])
                    if emotion_keywords:
                        emotion_mask = np.isin(numpy_data['emotions'][window_mask_60s], emotion_keywords)
                        ts_to_check = numpy_data['seconds'][window_mask_60s][emotion_mask]
                
                # 執行精煉掃描
                best_start = coarse_start; max_count = -1
                if ts_to_check.size > 0:
                    for sub_window_start in range(coarse_start, coarse_start + (60 - refined_window_size) + 1):
                        sub_window_end = sub_window_start + refined_window_size
                        current_count = np.sum((ts_to_check >= sub_window_start) & (ts_to_check < sub_window_end))
                        if current_count > max_count:
                            max_count = current_count; best_start = sub_window_start
                
                refined_r = r.copy(); refined_r['start_second'] = best_start
                processing_list.append(refined_r)
        
        # --- 格式化輸出 ---
        def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        def format_episode(ep):
            try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
            except (ValueError, TypeError): return str(ep)
        
        output_list = []
        if category == "精彩的戰鬥/競技片段" or category == "放閃/心動/害羞": final_window = 45
        elif category == "TOP 10 彈幕時段": final_window = analysis_window
        else: final_window = 30
        
        for r in processing_list:
            item = {'集數': format_episode(r['集數']),'時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + final_window)}",'start_second': int(r['start_second'])}
            if 'rate' in r:
                item['熱度分數'] = round(r['score'], 2)
                item['彈幕數量'] = int(r['count'])
                # 將 rate 格式化為帶有一位小數的百分比字串
                item['情緒佔比'] = f"{r['rate']:.1%}"
            else:
                # 處理戰鬥時段和彈幕密度這種沒有 rate 的情況
                item['彈幕數量'] = int(r['score'])
                item['熱度分數'] = round(r['score'], 2)
            output_list.append(item)
            
        final_result[category] = output_list
    
    logging.info(f"--- 全部分析與精煉完成，總耗時 {time.time() - start_time:.2f} 秒 ---")
    return final_result


if __name__ == '__main__':
    # ... (此處的獨立測試腳本完全不變) ...
    pass


