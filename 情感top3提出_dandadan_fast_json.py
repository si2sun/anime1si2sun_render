--- START OF FILE 情感top3提出_dandadan_fast_json.py ---

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

# <<<<<<<<<<<<<<<<<<<< 新增函式：抓取精彩戰鬥時段 >>>>>>>>>>>>>>>>>>>>>>
def get_exciting_battle_moments(df: pd.DataFrame, anime_name: str, 
                                keywords: list, analysis_window: int = 60, 
                                top_n: int = 5, min_gap: int = 300):
    """
    根據指定的關鍵字列表，計算戰鬥/作畫最精彩的時段。
    - df: 包含彈幕的 DataFrame。
    - anime_name: 要分析的動漫名稱。
    - keywords: 用於識別戰鬥/作畫精彩片段的關鍵字列表。
    - analysis_window: 分析窗口大小（秒）。
    - top_n: 返回的時段數量。
    - min_gap: 兩個時段之間的最小間隔（秒）。
    """
    logging.info("\n--- 開始計算『精彩的戰鬥時段』---")
    
    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty:
        return []

    # 過濾包含關鍵字的彈幕
    keyword_regex = '|'.join(keywords)
    df_battle = df_anime[df_anime['彈幕'].str.contains(keyword_regex, na=False)].copy()

    if df_battle.empty:
        logging.info("--- 未找到與戰鬥/作畫相關的關鍵字彈幕，跳過計算 ---")
        return []

    # 時間轉換為秒
    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except: return 0
    df_battle['秒數'] = df_battle['時間'].apply(time_to_seconds)

    battle_seconds_by_ep = {ep: group['秒數'].to_numpy(dtype=np.int32) for ep, group in df_battle.groupby('集數')}
    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()

    all_highlights = []
    
    # 滑動窗口分析
    for ep, seconds_in_ep in battle_seconds_by_ep.items():
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window: continue
        
        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            # 計算在60秒窗口內，包含關鍵字的彈幕數量
            count = np.sum((seconds_in_ep >= t_start) & (seconds_in_ep < t_end))
            
            # 設置一個最低門檻，避免零星的關鍵字也被計入
            if count > 5:
                all_highlights.append({'集數': ep, 'start_second': t_start, '關鍵字彈幕數': count})
    
    if not all_highlights: return []
        
    # 排序並選出 top_n
    highlights_df = pd.DataFrame(all_highlights).sort_values(by='關鍵字彈幕數', ascending=False)
    final_selected_list = []
    
    for _, row in highlights_df.iterrows():
        if len(final_selected_list) >= top_n: break
        is_conflict = any(r['集數'] == row['集數'] and abs(r['start_second'] - row['start_second']) < min_gap for r in final_selected_list)
        if not is_conflict:
            final_selected_list.append(row.to_dict())

    # 格式化輸出
    def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
    def format_episode(ep):
        try: num_ep = float(ep); return str(int(num_ep)) if num_ep == int(num_ep) else str(num_ep)
        except (ValueError, TypeError): return str(ep)

    output_list = [{'集數': format_episode(r['集數']), '時段': f"{seconds_to_time_str(r['start_second'])}~{seconds_to_time_str(r['start_second'] + analysis_window)}", '彈幕數量': int(r['關鍵字彈幕數']), 'start_second': int(r['start_second'])} for r in final_selected_list]
    logging.info("--- 『精彩的戰鬥時段』計算完成 ---\n")
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
    【滑動窗口 & 熱度分數版】 - V12 (整合戰鬥時段分析)
    - 採用差异化占比门槛：仅对 'LIVE/神配乐' 分类施加 30% 门槛，其他分类无此限制。
    - 新增“精彩的戰鬥時段”分析。
    """
    logging.debug(f"DEBUG: 執行整合版分析函式 (V12)...") # 使用 logging 替代 print
    start_time = time.time()
    
    # <<<<<<<<<<<<<<< 精彩戰鬥時段的關鍵字列表 >>>>>>>>>>>>>>>
    BATTLE_KEYWORDS = [
        "經費", "帥", "運鏡", "666", "作畫", "燃", "神", "分鏡", "高能", 
        "外掛", "爆", "炸", "猛", "777", "速度", "流暢", "魄力", "優雅", 
        "BGM", "打鬥"
    ]

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
    if df_classified.empty:
        # 即使沒有情感分類，仍然可能可以計算戰鬥時段
        logging.warning(f"警告: 作品 '{anime_name}' 沒有可供分析的情緒標籤數據。")
    
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
    
    # <<<<<<<<<<<<<<<<< 在此處呼叫新的戰鬥時段分析函式 >>>>>>>>>>>>>>>>>
    battle_moments = get_exciting_battle_moments(df, anime_name, keywords=BATTLE_KEYWORDS)
    if battle_moments:
        # 將結果存入 output_result 中，key 為 "精彩的戰鬥時段"
        output_result["精彩的戰鬥時段"] = battle_moments

    logging.debug(f"\nDEBUG: 全部分析完成，總耗時 {time.time() - start_time:.2f} 秒。") # 使用 logging 替代 print
    return output_result

# 階段 4: 主程式執行區塊 (所有測試程式碼都應該放在這裡)
# ==============================================================================
if __name__ == "__main__":
    
    # --- 測試用的資料和設定 ---
    # 使用您提供的更豐富的數據來進行測試
    csv_data="""彈幕,label,label2,作品名,集數,時間,情緒
星爆氣流斬！！！,0,1.0,進擊的巨人,22,0:05:21,稱讚
難道是傳說中的阿邦神速斬?！,4,5.0,進擊的巨人,22,0:05:21,疑問
Star Burst Stream,3,99.0,進擊的巨人,22,0:05:21,負面/其他
以為是在玩MHW雙刀鬼人模式,1,99.0,進擊的巨人,22,0:05:22,中立/其他
帥帥的,0,1.0,進擊的巨人,22,0:05:22,稱讚
高潮了!!!,0,3.0,進擊的巨人,22,0:05:22,強烈稱讚
阿卡曼發威,0,4.0,進擊的巨人,22,0:05:22,嘲笑/吐槽梗 
幹 帥死 我愛了,2,6.0,進擊的巨人,22,0:05:22,感動
戰鬥陀螺,1,0.0,進擊的巨人,22,0:05:23,理性講述
48763,1,0.0,進擊的巨人,22,0:05:23,理性講述
前方作畫高能,0,1.0,進擊的巨人,11,0:21:06,稱讚
前面作畫完全對的起「霸權」2個字,1,3.0,進擊的巨人,11,0:21:07,認真講述/感嘆
這段神爆了,0,3.0,進擊的巨人,11,0:21:08,強烈稱讚
此段作為江原康之,1,0.0,進擊的巨人,11,0:21:08,理性講述
高能,0,1.0,進擊的巨人,11,0:21:08,稱讚
荒木哲郎的分鏡真的很猛,0,3.0,進擊的巨人,11,0:21:09,強烈稱讚
我看到進擊的經費,1,0.0,進擊的巨人,11,0:21:09,理性講述
經費在燃燒,1,0.0,進擊的巨人,11,0:21:09,理性講述
這種運鏡超花錢的,2,0.0,進擊的巨人,11,0:21:09,非客觀不滿
這段運鏡超棒,0,3.0,進擊的巨人,11,0:21:09,強烈稱讚
蜘蛛人,1,0.0,進擊的巨人,11,0:21:10,理性講述
飆車搂,0,2.0,進擊的巨人,11,0:21:10,雅夏方面
帥死了,0,3.0,進擊的巨人,11,0:21:10,強烈稱讚
經費之燃燒,0,1.0,進擊的巨人,11,0:21:10,稱讚
這段動畫真的很神，看好幾次,3,6.0,進擊的巨人,11,0:21:11,感動
帥,0,1.0,進擊的巨人,11,0:21:11,稱讚
散場的擁抱~~經費在燃燒~~,0,2.0,進擊的巨人,11,0:21:12,雅夏方面
ＢＧＭ很可以,0,1.0,進擊的巨人,11,0:21:12,稱讚
太神啦,0,3.0,進擊的巨人,11,0:21:12,強烈稱讚
這叫欲揚先抑啊,1,3.0,進擊的巨人,11,0:21:13,認真講述/感嘆
好帥,0,1.0,進擊的巨人,11,0:21:13,稱讚
經費在燃燒,1,0.0,進擊的巨人,11,0:21:14,理性講述
每個都在拼技術的,0,4.0,進擊的巨人,11,0:21:16,嘲笑/吐槽梗 
轉場帥,0,1.0,進擊的巨人,11,0:21:17,稱讚
潮到不行啊!!!,2,4.0,進擊的巨人,11,0:21:17,生氣
運鏡好帥!!!,0,1.0,進擊的巨人,11,0:21:17,稱讚
我比較好奇動畫師的肝還好嗎,4,2.0,進擊的巨人,11,0:21:17,指事句/反問句
這段用VR一定很爽,0,3.0,進擊的巨人,11,0:21:18,強烈稱讚
好帥,0,1.0,進擊的巨人,11,0:21:18,稱讚
古寧頭保衛戰,1,0.0,進擊的巨人,11,0:21:18,理性講述
這分鏡有夠屌,0,3.0,進擊的巨人,11,0:21:18,強烈稱讚
這飛的過程好可怕呀,2,3.0,進擊的巨人,11,0:21:19,害怕
看得豪爽啊這幕,0,3.0,進擊的巨人,11,0:21:19,強烈稱讚
嗨,0,1.0,進擊的巨人,11,0:21:20,稱讚
太有錢了吧,0,3.0,進擊的巨人,11,0:21:20,強烈稱讚
霸權社真的猛,0,3.0,進擊的巨人,11,0:21:20,強烈稱讚
神作畫,0,3.0,進擊的巨人,11,0:21:20,強烈稱讚
超帥.,0,3.0,進擊的巨人,11,0:21:21,強烈稱讚
如果是我一定華麗轉身撞牆,2,2.0,進擊的巨人,11,0:21:22,客觀不滿/無言/尷尬
帥之運鏡,0,1.0,進擊的巨人,11,0:21:22,稱讚
"""
    df = pd.read_csv(io.StringIO(csv_data))
    
    emotion_mapping={
        "劇情高潮/震撼": [
            "劇情高潮", "震撼", "伏筆", "強烈稱讚", "稱讚"
        ],
        "感動/淚點": ["感動", "悲傷"]
    }
    
    ANIME_NAME_TO_ANALYZE = "進擊的巨人"
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("="*60)
    logging.info(f"開始分析動畫: 《{ANIME_NAME_TO_ANALYZE}》")
    logging.info("="*60)
    
    # 測試主函式
    emotional_hotspots = get_top3_emotions_fast(
        df=df,
        anime_name=ANIME_NAME_TO_ANALYZE,
        emotion_mapping=emotion_mapping,
        analysis_window=60,
        min_gap=240,
        top_n=5
    )

    logging.info("\n" + "="*60)
    logging.info("      分析完成！最終熱點報告如下：")
    logging.info("="*60 + "\n")
    
    if emotional_hotspots:
        logging.info(json.dumps(emotional_hotspots, indent=4, ensure_ascii=False))

    if not emotional_hotspots:
        logging.info("非常抱歉，根據目前的設定，找不到任何熱點。")

