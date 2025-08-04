import pandas as pd
import numpy as np
import time
import logging
from collections import defaultdict

# 設定日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_all_highlights_single_pass(
    df: pd.DataFrame, 
    anime_name: str, 
    emotion_mapping: dict,
    analysis_window: int = 60, 
    min_gap: int = 240
):
    """
    【V15 - 最佳化單核心版】
    單集逐步掃描 + NumPy 高速運算 + 篩除無效集數 + 精簡 Counter 操作
    """
    logging.info("\n--- 開始執行最終版高效單次掃描分析 (V15) ---")
    start_time = time.time()

    df_anime = df[df['作品名'] == anime_name].copy()
    if df_anime.empty:
        return {}

    def time_to_seconds(t):
        if pd.isna(t): return 0
        try:
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        except:
            return 0

    df_anime['秒數'] = df_anime['時間'].apply(time_to_seconds)
    df_anime['is_signin'] = df_anime['情緒'] == '簽到'

    battle_keywords = ["經費", "帥", "運鏡", "666", "作畫", "燃", "分鏡", "高能", "外掛", "爆", "炸", "猛", "777", "速度", "流暢", "魄力", "優雅", "BGM", "打鬥", "強"]
    df_anime['is_battle'] = df_anime['彈幕'].str.contains('|'.join(battle_keywords), na=False)

    def classify_emotion(e):
        for cat, e_list in emotion_mapping.items():
            if e in e_list:
                return cat
        return None
    df_anime['情緒分類'] = df_anime['情緒'].apply(classify_emotion)

    episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()
    all_highlights = defaultdict(list)

    for ep, group_df in df_anime.groupby('集數'):
        max_time = episode_max_times.get(ep, 0)
        if not max_time or max_time < analysis_window or group_df.shape[0] < 30:
            continue

        logging.info(f"  -> 正在掃描第 {ep} 集...")

        seconds = group_df['秒數'].to_numpy(np.int32)
        emotions = group_df['情緒分類'].to_numpy()
        is_battle = group_df['is_battle'].to_numpy()
        is_signin = group_df['is_signin'].to_numpy()

        for t_start in range(0, max_time - analysis_window + 1):
            t_end = t_start + analysis_window
            mask = (seconds >= t_start) & (seconds < t_end)
            total_count = np.sum(mask)
            if total_count < 10:
                continue

            window_emotions = emotions[mask]
            valid_emotions = window_emotions[pd.notna(window_emotions)]
            unique, counts = np.unique(valid_emotions, return_counts=True)
            emotion_counts = dict(zip(unique, counts))

            for emotion_category, count in emotion_counts.items():
                min_count = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count:
                    continue
                rate = count / total_count
                if emotion_category == "LIVE/神配樂" and rate < 0.3:
                    continue
                score = count * rate
                all_highlights[emotion_category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})

            battle_count = np.sum(is_battle[mask])
            if battle_count > 5:
                all_highlights["精彩的戰鬥時段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

            density = total_count - np.sum(is_signin[mask])
            if density > 10:
                all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density})

    logging.info(f"--- 全集掃描完成，耗時 {time.time() - start_time:.2f} 秒 ---")

    # 結果整理階段
    final_result = {}
    for category, highlights in all_highlights.items():
        if not highlights:
            continue

        df_cat = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
        top_n = 10 if category in ["TOP 10 彈幕時段", "劇情高潮/震撼"] else 7 if category in ["虐點/感動", "精彩的戰鬥時段"] else 5

        selected, quota = [], defaultdict(int)
        for _, row in df_cat.iterrows():
            if len(selected) >= top_n:
                break
            ep_row = row['集數']
            if quota[ep_row] >= 2 and category not in ["精彩的戰鬥時段", "TOP 10 彈幕時段"]:
                continue
            conflict = any(r['集數'] == ep_row and abs(r['start_second'] - row['start_second']) < min_gap for r in selected)
            if not conflict:
                selected.append(row.to_dict())
                quota[ep_row] += 1

        if not selected:
            continue

        def seconds_to_time_str(s): s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"
        def format_episode(ep):
            try: num = float(ep); return str(int(num)) if num == int(num) else str(num)
            except: return str(ep)

        out = []
        final_window = 30 if category not in ["精彩的戰鬥時段", "TOP 10 彈幕時段"] else analysis_window
        for r in selected:
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
            out.append(item)
        final_result[category] = out

    logging.info(f"--- 全部分析完成，總耗時 {time.time() - start_time:.2f} 秒 ---")
    return final_result
