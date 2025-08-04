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

# 定義戰鬥關鍵字
BATTLE_KEYWORDS = [
    "經費", "帥", "運鏡", "666", "作畫", "燃", "分鏡", "高能", 
    "外掛", "爆", "炸", "猛", "777", "速度", "流暢", "魄力", "優雅", 
    "BGM", "打鬥" , "強"
]
keyword_regex = '|'.join(BATTLE_KEYWORDS)
df_anime['is_battle'] = df_anime['彈幕'].str.contains(keyword_regex, na=False)

# 預先分類情緒
def classify_emotion(e):
    for cat, e_list in emotion_mapping.items():
        if e in e_list: return cat
    return None
df_anime['情緒分類'] = df_anime['情緒'].apply(classify_emotion)

all_highlights = defaultdict(list)
episode_max_times = df_anime.groupby('集數')['秒數'].max().to_dict()
# 將秒數設為索引並排序，這是高效切片窗口的關鍵
df_anime_sorted = df_anime.set_index('秒數').sort_index()

# --- 2. 單次滑動窗口掃描 ---
for ep, group_df_ep in df_anime_sorted.groupby('集數'):
    max_time = episode_max_times.get(ep, 0)
    if not max_time or max_time < analysis_window: continue
    
    logging.info(f"  -> 正在掃描第 {ep} 集...")
    
    # 遍歷每一秒作為窗口的起點
    for t_start in range(0, max_time - analysis_window + 1):
        t_end = t_start + analysis_window
        
        # 高效獲取窗口內的數據
        window_df = group_df_ep.loc[t_start:t_end-1]
        total_count = len(window_df)
        
        if total_count < 10:
            continue

        # a) 計算情感分類熱度
        if '情緒分類' in window_df.columns:
            emotion_counts = window_df['情緒分類'].value_counts()
            for emotion_category, count in emotion_counts.items():
                if pd.isna(emotion_category): continue
                
                min_count_threshold = 7 if "虐點/感動" in emotion_category or "劇情高潮" in emotion_category else 5
                if count < min_count_threshold: continue

                rate = count / total_count
                
                # 對特定分類應用佔比門檻
                if emotion_category == "LIVE/神配樂" and rate < 0.3:
                    continue

                score = count * rate
                all_highlights[emotion_category].append({'集數': ep, 'start_second': t_start, 'score': score, 'count': count, 'rate': rate})
        
        # b) 計算精彩戰鬥時段
        battle_count = window_df['is_battle'].sum()
        if battle_count > 5:
             all_highlights["精彩的戰鬥時段"].append({'集數': ep, 'start_second': t_start, 'score': battle_count})

        # c) 計算 TOP 5 彈幕密度
        density_count = len(window_df[window_df['情緒'] != '簽到'])
        if density_count > 10:
            all_highlights["TOP 10 彈幕時段"].append({'集數': ep, 'start_second': t_start, 'score': density_count})

logging.info(f"--- 全集掃描完成，耗時 {time.time() - start_time:.2f} 秒 ---")

# --- 3. 結果後處理 ---
final_result = {}
for category, highlights in all_highlights.items():
    if not highlights: continue
    
    highlights_df = pd.DataFrame(highlights).sort_values(by='score', ascending=False)
    
    selected_list = []
    current_top_n = 10 if "劇情高潮" in category else 5
    if "虐點/感動" in category: current_top_n = 7
    if category in ["精彩的戰鬥時段"]: current_top_n = 5
    if category in ["TOP 10 彈幕時段"]: current_top_n = 5
  
    
    # 每個類別每集最多取2個
    episode_quota_tracker = defaultdict(int)

    for _, row in highlights_df.iterrows():
        if len(selected_list) >= current_top_n: break
        
        ep_row = row['集數']
        # 檢查單集配額
        if episode_quota_tracker[ep_row] >= 2 and category not in ["精彩的戰鬥時段", "TOP 10 彈幕時段"]:
            continue

        # 檢查時間間隔衝突
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
    # 情感分類做30秒精煉，其他分類顯示原始60秒
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






