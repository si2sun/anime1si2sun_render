import pandas as pd
import numpy as np
import io
import time
import json
import logging # 新增日誌模組

# 配置日誌 (如果 main.py 沒有統一配置，這裡也可以配置)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_top5_density_moments(df: pd.DataFrame, anime_name: str, 
                             analysis_window: int = 60, top_n: int = 10, 
                             min_gap: int = 300):
    """
    計算整部動畫中，（剔除'簽到'後）彈幕密度最高的前 N 個時段。
    分析范围为整部影片（0 秒到结尾）。
    """
    logging.info("\n--- 開始計算 TOP 5 彈幕密度榜 (完整扫描) ---") # 使用 logging 替代 print
    
    df_anime = df[df['作品名'] == anime_name].copy()
    if '情緒' in df_anime.columns:
        df_anime = df_anime[df_anime['情緒'] != '簽到']
    if df_anime.empty: return []
        
    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except: return 0
    df_anime['秒數'] = df_anime['時間'].apply(time_to_seconds)

    all_danmaku_seconds = {ep: group['秒數'].to_numpy(dtype=np.int32) for ep, group in df_anime.groupby('集數')}
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()
    
    all_highlights = []
    
    for ep, seconds_in_ep in all_danmaku_seconds.items():
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            count = np.sum((seconds_in_ep >= t_start) & (seconds_in_ep < t_end))
            
            if count > 10:
                all_highlights.append({'集數': ep, 'start_second': t_start, '彈幕數量': count})
    
    if not all_highlights: return []
        
    highlights_df = pd.DataFrame(all_highlights).sort_values(by='彈幕數量', ascending=False)
    final_selected_list = []
    
    for _, row in highlights_df.iterrows():
        if len(final_selected_list) >= top_n: break
        is_conflict = any(r['集數'] == row['集數'] and abs(r['start_second'] - row['start_second']) < min_gap for r in final_selected_list)
        if not is_conflict:
            final_selected_list.append(row.to_dict())

    def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
    def format_episode(ep):
        try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
        except (ValueError, TypeError): return str(ep)

    output_list = [{'集數': format_episode(r['集數']), '時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + analysis_window)}", '彈幕數量': int(r['彈幕數量']), 'start_second': int(r['start_second'])} for r in final_selected_list]
    logging.info("--- TOP 5 彈幕密度榜計算完成 ---\n") # 使用 logging 替代 print
    return output_list
      
def get_top3_emotions_fast(
    df: pd.DataFrame, 
    anime_name: str, 
    emotion_mapping: dict, 
    analysis_window: int = 60, 
    min_gap: int = 240, 
    top_n: int = 5
):
    """
    【滑動窗口 & 熱度分數版】 - V11 (最终精细版)
    - 采用差异化占比门槛：仅对 'LIVE/神配乐' 分类施加 30% 门槛，其他分类无此限制。
    """
    logging.debug(f"DEBUG: 執行最终精细版分析函式 (V11)...") # 使用 logging 替代 print
    start_time = time.time()

    # --- 步骤 1: 数据预处理与分类 (保持不变) ---
    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty: return {}

    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except: return 0
    df_anime['秒數'] = df_anime['時間'].apply(time_to_seconds)

    def classify_emotion(e):
        for cat, e_list in emotion_mapping.items():
            if e in e_list: return cat
        return None
    df_anime['情緒分類'] = df_anime['情緒'].apply(classify_emotion)
    
    df_classified = df_anime.dropna(subset=['情緒分類', '秒數']).sort_values('秒數')
    if df_classified.empty: return {}
    
    all_danmaku_seconds = {ep: group['秒數'].to_numpy(dtype=np.int32) for ep, group in df_anime.groupby('集數')}
    
    output_result = {}
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()

    # --- 步骤 2: 对每种情绪类别进行分析 ---
    for emotion_category, emotion_list in emotion_mapping.items():
        logging.info(f"  -> 開始分析情緒分類: 【{emotion_category}】") # 使用 logging 替代 print
        
        group_df = df_classified[df_classified['情緒分類'] == emotion_category]
        emotion_seconds_by_ep = {ep: g['秒數'].to_numpy(dtype=np.int32) for ep, g in group_df.groupby('集數')}
        
        all_potential_highlights = []
        
        # --- 步骤 3: 逐集进行滑动窗口扫描 ---
        for ep, emotion_seconds in emotion_seconds_by_ep.items():
            max_time_to_use = episode_max_times.get(ep, 0)
            if not max_time_to_use or max_time_to_use < analysis_window: continue
            
            total_seconds_in_ep = all_danmaku_seconds.get(ep, np.array([]))

            for t_start in range(0, max_time_to_use - analysis_window + 1):
                t_end = t_start + analysis_window
                
                MIN_TOTAL_COMMENTS = 20
                total_count = np.sum((total_seconds_in_ep >= t_start) & (total_seconds_in_ep < t_end))
                if total_count < MIN_TOTAL_COMMENTS: continue

                count = np.sum((emotion_seconds >= t_start) & (emotion_seconds < t_end))
                
                min_count_threshold = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count_threshold: continue

                rate = count / total_count
                
                # <<<<<<< 关键修改：差异化占比门槛 >>>>>>>
                if emotion_category == "LIVE/神配樂":
                    MIN_RATE_THRESHOLD = 0.3
                    if rate < MIN_RATE_THRESHOLD:
                        continue
                
                score = count * rate
                
                all_potential_highlights.append({
                    '集數': ep, 'start_second': t_start, 'score': score, 
                    'count': count, 'rate': rate
                })

        if not all_potential_highlights: continue
        
        highlights_df = pd.DataFrame(all_potential_highlights).sort_values(by='score', ascending=False)
        final_selected_list = []
        episode_quota_tracker = {}
        MAX_PER_EPISODE = 2
        
        if "虐點/感動" in emotion_category: current_top_n = 7
        elif "劇情高潮/震撼" in emotion_category: current_top_n = 10
        elif "LIVE/神配樂" in emotion_category: current_top_n = 7
        else: current_top_n = top_n
        
        for _, row in highlights_df.iterrows():
            if len(final_selected_list) >= current_top_n: break
            
            ep = row['集數']
            
            if episode_quota_tracker.get(ep, 0) >= MAX_PER_EPISODE:
                continue

            is_conflict = any(r['集數'] == ep and abs(r['start_second'] - row['start_second']) < min_gap for r in final_selected_list)
            if is_conflict:
                continue

            final_selected_list.append(row.to_dict())
            episode_quota_tracker[ep] = episode_quota_tracker.get(ep, 0) + 1

        if final_selected_list:
            REFINED_WINDOW = 30
            refined_list = []
            for window_60s in final_selected_list:
                ep = window_60s['集數']
                seconds_to_refine = emotion_seconds_by_ep.get(ep, np.array([]))
                
                best_30s_start, max_count_in_30s = window_60s['start_second'], -1
                for t_start_30s in range(window_60s['start_second'], window_60s['start_second'] + analysis_window - REFINED_WINDOW + 1):
                    t_end_30s = t_start_30s + REFINED_WINDOW
                    count_in_30s = np.sum((seconds_to_refine >= t_start_30s) & (seconds_to_refine < t_end_30s))
                    if count_in_30s > max_count_in_30s:
                        max_count_in_30s = count_in_30s
                        best_30s_start = t_start_30s
                
                window_60s['start_second'] = best_30s_start
                refined_list.append(window_60s)
            
            refined_list.sort(key=lambda x: x['score'], reverse=True)
            
            def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
            def format_episode(ep):
                try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
                except (ValueError, TypeError): return str(ep)
                
            output_list = [{'集數': format_episode(r['集數']), '時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + REFINED_WINDOW)}", '熱度分數': round(r['score'], 2), '彈幕數量': int(r['count']), '情緒佔比': f"{r['rate']:.1%}", 'start_second': int(r['start_second'])} for r in refined_list]
            output_result[emotion_category] = output_list
    
    logging.debug(f"\nDEBUG: 全部分析完成，總耗時 {time.time() - start_time:.2f} 秒。") # 使用 logging 替代 print
    return output_result 
# 階段 4: 主程式執行區塊 (所有測試程式碼都應該放在這裡)
# ==============================================================================
if __name__ == "__main__":
    
    # --- 測試用的資料和設定 ---
    csv_data="""彈幕,label,label2,作品名,集數,時間,情緒
    """ # 這裡應該放一些測試資料，否則 df 會是空的

    # 為確保測試能運行，提供一個最小的範例數據
    csv_data = """彈幕,label,label2,作品名,集數,時間,情緒
    彈幕1,label_a,label2_x,佐賀偶像是傳奇 捲土重來,1,00:01:00,感動
    彈幕2,label_b,label2_y,佐賀偶像是傳奇 捲土重來,1,00:01:10,正面/其他
    彈幕3,label_c,label2_z,佐賀偶像是傳奇 捲土重來,1,00:01:20,稱讚
    彈幕4,label_d,label2_a,佐賀偶像是傳奇 捲土重來,1,00:01:30,感動
    彈幕5,label_e,label2_b,佐賀偶像是傳奇 捲土重來,1,00:01:40,正面/其他
    彈幕6,label_f,label2_c,佐賀偶像是傳奇 捲土重來,2,00:02:00,感動
    彈幕7,label_g,label2_d,佐賀偶像是傳奇 捲土重來,2,00:02:15,稱讚
    彈幕8,label_h,label2_e,佐賀偶像是傳奇 捲土重來,2,00:02:30,感動
    彈幕9,label_i,label2_f,佐賀偶像是傳奇 捲土重來,2,00:02:45,簽到
    彈幕10,label_j,label2_g,佐賀偶像是傳奇 捲土重來,2,00:02:50,正面/其他
    """
    df = pd.read_csv(io.StringIO(csv_data))
    
    emotion_mapping={
        "LIVE/神配樂": [
            "日語歌詞/外語梗句", "誓言", "感動","稱讚","正面/其他",'強烈稱讚'
        ],
        "劇情高潮/震撼": [
            "劇情高潮", "震撼", "伏筆"
        ]
    }
    
    ANIME_NAME_TO_ANALYZE = "佐賀偶像是傳奇 捲土重來"
    ANALYSIS_WINDOW_SECONDS = 60 
    
    logging.info("="*60) # 使用 logging 替代 print
    logging.info(f"開始分析動畫: 《{ANIME_NAME_TO_ALYZE}》") # 使用 logging 替代 print
    logging.info("="*60) # 使用 logging 替代 print
    
    # 測試主函式
    emotional_hotspots = get_top3_emotions_fast(
        df=df,
        anime_name=ANIME_NAME_TO_ANALYZE,
        emotion_mapping=emotion_mapping,
        analysis_window=ANALYSIS_WINDOW_SECONDS,
        min_gap=300,
        top_n=5
    )
    
    # 測試 TOP 5 密度榜函式
    top_5_moments = get_top5_density_moments(
        df=df,
        anime_name=ANIME_NAME_TO_ANALYZE,
        min_gap=300
    )

    logging.info("\n" + "="*60) # 使用 logging 替代 print
    logging.info("      分析完成！最終情感熱點報告如下：") # 使用 logging 替代 print
    logging.info("="*60 + "\n") # 使用 logging 替代 print
    
    if emotional_hotspots:
        logging.info("--- 情感分類熱點 ---") # 使用 logging 替代 print
        logging.info(json.dumps(emotional_hotspots, indent=4, ensure_ascii=False)) # 使用 logging 替代 print

    if top_5_moments:
        logging.info("\n--- TOP 5 彈幕時段 ---") # 使用 logging 替代 print
        logging.info(json.dumps({"TOP 5 彈幕時段": top_5_moments}, indent=4, ensure_ascii=False)) # 使用 logging 替代 print

    if not emotional_hotspots and not top_5_moments:
        logging.info("非常抱歉，根據目前的設定，找不到任何熱點。") # 使用 logging 替代 print