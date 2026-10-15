import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
import time
import re
from bs4 import BeautifulSoup
import google.generativeai as genai

# ==========================================
# 0. 初期設定とAPI設定
# ==========================================
st.set_page_config(page_title="地方競馬 AI ROIオプティマイザ", page_icon="🏇", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 永続化（JSONファイル保存）ロジック
# ==========================================
RULES_FILE = "rules.json"
DEFAULT_RULES = [
    "【展開・位置取り】地方の小回りコースではテンのスピード（前走通過順1〜2番手）を最優先で評価すること。",
    "【馬場状態・時計勝負】重・不良馬場では極端に時計が速くなるため、良馬場の上がり性能よりも「持ち時計」を強力加点すること。"
]

def load_rules():
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_RULES
    return DEFAULT_RULES

def save_rules(rules):
    try:
        with open(RULES_FILE, "w", encoding="utf-8") as f:
            json.dump(rules, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"ルールの保存に失敗しました: {e}")

if "feedback_rules" not in st.session_state:
    st.session_state["feedback_rules"] = load_rules()
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = {}

# ==========================================
# 1. AI予測 & 統合ロジック
# ==========================================
def deduplicate_and_merge_rules(existing_rules, new_rule):
    if not existing_rules:
        return [new_rule], "新規追加"
    system_prompt = """
    あなたは地方競馬AIのナレッジベース管理モジュールです。
    【既存のルール一覧】と【追加したい新しいルール】を分析し、同種のものは統合し、新しいものは末尾に追加してください。
    JSONフォーマットのみを出力してください: {"status": "MERGED", "merged_rules": ["..."], "reason": "..."}
    """
    prompt_input = f"【既存ルール】:\n" + "\n".join(existing_rules) + f"\n\n【新ルール】:\n{new_rule}"
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        res = model.generate_content(prompt_input, generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1))
        res_json = json.loads(res.text)
        return res_json.get("merged_rules", existing_rules + [new_rule]), res_json.get("reason", "処理完了")
    except Exception:
        if new_rule not in existing_rules: return existing_rules + [new_rule], "新規追加"
        return existing_rules, "同種ルールあり"

def get_gemini_prediction(combined_race_text, feedback_rules=None):
    rules_text = "\n".join([f"- {r}" for r in feedback_rules]) if feedback_rules else "- 特になし"
    system_prompt = f"""
    あなたは地方競馬の確率論・展開読み・時計解析に精通したプロデータサイエンティストです。
    提供された【出馬表データ】を精査し、各馬の勝率、レース展開、荒れ度、印、推奨買い目を分析してください。
    
    【🚨絶対厳守】
    1. オッズの捏造禁止。見つからない場合は `0.0` とすること。
    2. 馬名は実在するもののみ使用。印(◎◯▲△)は最低4頭出力。
    3. volatility_score は必ず 1, 2, 3 のいずれかの数値を出力すること (1: 堅実/低波乱, 2: 中波乱, 3: 大荒れ/超高波乱)。
    
    【学習済みルール（最優先）】\n{rules_text}
    
    JSONフォーマットのみを出力:
    {{
      "race_name": "レース名", 
      "volatility_score": 3,
      "volatility_level": "🔥🔥🔥 超高波乱", 
      "volatility_reason": "理由",
      "pace_prediction": "展開予測", 
      "pace_reason": "理由",
      "predictions": [ {{"mark": "◎", "horse_number": 1, "horse_name": "馬名", "predicted_win_rate": 0.3, "current_odds": 0.0, "reason": "理由"}} ],
      "recommended_umaren": [ {{"combination": "1-2", "predicted_rate": 0.15, "current_odds": 0.0, "reason": "理由"}} ],
      "recommended_sanrenpuku": [ {{"combination": "1-2-3", "predicted_rate": 0.08, "current_odds": 0.0, "reason": "理由"}} ]
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        res = model.generate_content(combined_race_text, generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1))
        return json.loads(res.text)
    except Exception as e:
        return {"error": str(e)}

def analyze_race_result(result_text, previous_prediction=None):
    compare_prompt = f"【事前予想】\n{json.dumps(previous_prediction, ensure_ascii=False)}\n上記予想と実際の結果を照合し誤差を分析せよ。" if previous_prediction else ""
    system_prompt = f"""
    あなたは地方競馬予想AIの自己改善プログラムです。実際のレース結果を分析してください。
    {compare_prompt}
    今後の予想ロジック精度を上げるための追加ルール文を提案してください。
    また、実際の「馬連の的中組み合わせ（馬番）」と「その配当金額」を正確に抽出してください。
    
    JSONのみ出力: 
    {{
      "race_summary": "結果", 
      "actual_umaren": "実際の馬連組み合わせ (例: 1-2)",
      "actual_umaren_payout": "実際の馬連配当 (例: 1,500円)",
      "winning_factors": "勝因", 
      "ai_reflection": "反省点", 
      "suggested_rule": "新ルール案"
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        res = model.generate_content(result_text, generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1))
        return json.loads(res.text)
    except Exception as e:
        return {"error": str(e)}

def fetch_text_from_url(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.content, 'html.parser')
        for tag in soup(["script", "style", "noscript"]): tag.extract()
        return soup.get_text(separator=' ', strip=True)
    except Exception:
        return ""

def is_target_race(score, target_setting):
    score = int(score) if str(score).isdigit() else 2
    if target_setting == "すべて対象": return True
    if target_setting == "中波乱以上(回収率重視)" and score >= 2: return True
    if target_setting == "大荒れのみ(穴党)" and score == 3: return True
    if target_setting == "堅実のみ(本命党)" and score == 1: return True
    return False

def calculate_kelly_bet(predicted_win_rate, odds, bankroll, kelly_fraction=0.25):
    if odds <= 1.0: return {"action": "オッズ確認要", "bet_amount": 0, "ev": 0.0}
    ev = predicted_win_rate * odds
    if ev <= 1.0: return {"action": "見送り", "bet_amount": 0, "ev": round(ev, 2)}
    f = ((odds - 1.0) * predicted_win_rate - (1.0 - predicted_win_rate)) / (odds - 1.0)
    bet = int((bankroll * (f * kelly_fraction)) // 100 * 100)
    return {"action": "買い" if bet > 0 else "見送り", "bet_amount": max(bet, 100 if bet>0 else 0), "ev": round(ev, 2)}

def check_umaren_hit(ai_combinations, actual_combination):
    """馬連の組み合わせ（例: '1-2' と '2-1'）の順不同マッチング判定"""
    try:
        if not actual_combination or actual_combination == "-": return False
        actual_set = set(re.findall(r'\d+', actual_combination))
        if len(actual_set) != 2: return False
        
        for ai_combo in ai_combinations:
            ai_set = set(re.findall(r'\d+', str(ai_combo)))
            if ai_set == actual_set:
                return True
        return False
    except:
        return False

# ==========================================
# 2. UI画面構成
# ==========================================
st.sidebar.title("🏇 地方競馬 AI ROI")
page = st.sidebar.radio("メニュー", ["🛠️ 1R単体 分析", "🔄 指定日一括 AI予想＆抽出", "📈 過去データ バックテスト自動学習"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 馬券購入ターゲット設定")
target_volatility = st.sidebar.radio(
    "狙うレースの波乱度",
    ["すべて対象", "中波乱以上(回収率重視)", "大荒れのみ(穴党)", "堅実のみ(本命党)"],
    index=1
)
st.sidebar.caption("※設定した波乱度のレースだけが抽出されます。")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 適用中ルール (自動保存済み)")
if st.session_state["feedback_rules"]:
    for i, r in enumerate(st.session_state["feedback_rules"]): st.sidebar.caption(f"{i+1}. {r}")
if st.sidebar.button("ルール初期化"):
    st.session_state["feedback_rules"] = DEFAULT_RULES
    save_rules(DEFAULT_RULES)
    st.rerun()

if page == "🛠️ 1R単体 分析":
    st.title("🎯 単一レース分析")
    target_url = st.text_input("出馬表URL", value="https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=2026%2F10%2F14&k_raceNo=1&k_babaCode=24")
    if st.button("AI予想を実行"):
        with st.spinner("解析中..."):
            text = fetch_text_from_url(target_url)
            res = get_gemini_prediction(text, st.session_state["feedback_rules"])
            if "error" not in res:
                st.session_state["last_prediction"]["single"] = res
                
                score = res.get("volatility_score", 2)
                if is_target_race(score, target_volatility):
                    st.success("🟢 **このレースは設定条件に合致する【勝負レース】です！** 購入を検討してください。")
                else:
                    st.warning("🔴 **このレースは見送りを推奨します。**（設定した波乱度と一致しません）")
                
                st.write(f"**判定波乱度:** {res.get('volatility_level')}")
                st.write(f"**理由:** {res.get('volatility_reason')}")
                
                st.markdown("### 🏇 出走馬・印別予測勝率")
                preds = res.get("predictions", [])
                if preds:
                    table_preds = []
                    for item in preds:
                        rate = item.get("predicted_win_rate", 0)
                        odds = item.get("current_odds", 0.0)
                        kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                        table_preds.append({
                            "印": item.get("mark", "-"),
                            "馬番": item.get("horse_number", "-"),
                            "馬名": item.get("horse_name", "-"),
                            "AI予測勝率": f"{rate*100:.1f}%",
                            "単勝オッズ": f"{odds}倍" if odds > 0 else "取得不可",
                            "期待値(EV)": kelly["ev"] if odds > 0 else "-",
                            "単勝判定": kelly["action"],
                            "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "-",
                            "評価理由": item.get("reason", "")
                        })
                    st.dataframe(pd.DataFrame(table_preds), use_container_width=True)

                st.markdown("### 🎟️ おすすめ馬連ペア")
                umaren_list = res.get("recommended_umaren", [])
                if umaren_list:
                    table_umaren = []
                    for item in umaren_list:
                        combo = item.get("combination")
                        rate = item.get("predicted_rate", 0)
                        odds = item.get("current_odds", 0.0)
                        kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                        table_umaren.append({
                            "馬連ペア": combo,
                            "的中確率": f"{rate*100:.1f}%",
                            "想定オッズ": f"{odds}倍" if odds > 0 else "取得不可",
                            "期待値(EV)": kelly["ev"] if odds > 0 else "-",
                            "馬連判定": kelly["action"],
                            "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "-"
                        })
                    st.dataframe(pd.DataFrame(table_umaren), use_container_width=True)
            else:
                st.error("解析エラーが発生しました。")

elif page == "🔄 指定日一括 AI予想＆抽出":
    st.title("🔄 1日全レース 一括AI予想 ＆ 結果照合")
    base_url = st.text_input("1Rの出馬表 または 結果URLを入力", value="https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=2026%2F10%2F15&k_raceNo=1&k_babaCode=24")
    max_race = st.slider("処理する最終レース番号", min_value=1, max_value=12, value=12)
    
    col1, col2 = st.columns(2)
    if col1.button("🚀 一括AI予想 ＆ 買い目リスト作成"):
        progress_bar = st.progress(0)
        target_races = []
        all_umaren_preds = []
        
        for race_no in range(1, max_race + 1):
            current_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', base_url)
            text = fetch_text_from_url(current_url)
            if text:
                res = get_gemini_prediction(text, st.session_state["feedback_rules"])
                st.session_state["last_prediction"][f"{race_no}R"] = res
                
                if "error" not in res:
                    score = res.get("volatility_score", 2)
                    if is_target_race(score, target_volatility):
                        target_races.append(f"{race_no}R ({res.get('volatility_level')})")
                    
                    # 馬連予想の集計
                    for u in res.get("recommended_umaren", []):
                        all_umaren_preds.append({
                            "レース": f"{race_no}R",
                            "馬連ペア": u.get("combination", "-"),
                            "的中確率": f"{u.get('predicted_rate', 0)*100:.1f}%",
                            "推奨理由": u.get("reason", "")
                        })
            
            time.sleep(1.5)
            progress_bar.progress(race_no / max_race)
            
        st.success("✅ 一括予想完了！")
        
        # 勝負レースのピックアップ表示
        st.markdown("---")
        st.header(f"💰 買うべき勝負レース一覧 (設定: {target_volatility})")
        if target_races:
            for tr in target_races: st.markdown(f"- **{tr}**")
        else:
            st.warning("設定条件に合致するレースがありませんでした。資金温存を推奨します。")
            
        st.markdown("---")
        st.header("🎟️ 本日のAI推奨 馬連一覧")
        if all_umaren_preds:
            st.dataframe(pd.DataFrame(all_umaren_preds), use_container_width=True)
        else:
            st.write("馬連の推奨データがありません。")

    if col2.button("🏁 レース終了後：一括結果照合＆学習"):
        st.info("※レース結果に基づいてAI予想との答え合わせを行います。")
        progress_bar = st.progress(0)
        all_suggested_rules = []
        comparison_results = []
        
        for race_no in range(1, max_race + 1):
            res_url = base_url.replace("DebaTable", "RaceMarkTable")
            current_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', res_url)
            text = fetch_text_from_url(current_url)
            prev_pred = st.session_state["last_prediction"].get(f"{race_no}R", None)
            
            if text:
                analysis = analyze_race_result(text, previous_prediction=prev_pred)
                if "error" not in analysis:
                    if analysis.get('suggested_rule'):
                        all_suggested_rules.append(analysis['suggested_rule'])
                    
                    # 馬連の的中・不的中判定
                    actual_umaren = analysis.get("actual_umaren", "-")
                    actual_payout = analysis.get("actual_umaren_payout", "-")
                    
                    ai_umaren_list = [u.get("combination") for u in prev_pred.get("recommended_umaren", [])] if prev_pred else []
                    ai_umaren_str = ", ".join([str(a) for a in ai_umaren_list]) if ai_umaren_list else "予想なし"
                    
                    is_hit = check_umaren_hit(ai_umaren_list, actual_umaren)
                    hit_mark = "🎯 的中" if is_hit else "❌ 不的中"
                    if actual_umaren == "-": hit_mark = "不明"
                    
                    comparison_results.append({
                        "レース": f"{race_no}R",
                        "AI予想馬連": ai_umaren_str,
                        "実際の馬連結果": actual_umaren,
                        "配当": actual_payout,
                        "結果": hit_mark
                    })
                    
            time.sleep(1.5)
            progress_bar.progress(race_no / max_race)
            
        if comparison_results:
            st.markdown("---")
            st.header("📊 馬連 予想 vs 結果 比較レポート")
            st.dataframe(pd.DataFrame(comparison_results), use_container_width=True)
            
        if all_suggested_rules:
            new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], "\n".join(all_suggested_rules))
            st.session_state["feedback_rules"] = new_rules
            save_rules(new_rules)
            st.success("✅ 全レースの答え合わせと学習が完了し、`rules.json` に永続保存しました！")

elif page == "📈 過去データ バックテスト自動学習":
    st.title("📈 過去レース バックテスト自動学習マシーン")
    past_base_url = st.text_input("過去レース 1Rの出馬表URLを入力", placeholder="例: .../DebaTable?k_raceDate=2026%2F09%2F01&k_raceNo=1&k_babaCode=24")
    past_max_race = st.slider("学習するレース数", min_value=1, max_value=12, value=12)
    
    if st.button("🧠 過去データから全自動でAIをトレーニングする"):
        if "k_raceNo=" not in past_base_url:
            st.error("URLに `k_raceNo=` が含まれていません。")
        else:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            all_suggested_rules = []
            
            for race_no in range(1, past_max_race + 1):
                progress_text.text(f"【{race_no}R】 出馬表解析中 ＆ 結果照合中... (AIトレーニング実行中)")
                deba_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', past_base_url)
                deba_text = fetch_text_from_url(deba_url)
                mock_pred = None
                if deba_text:
                    mock_pred = get_gemini_prediction(deba_text, st.session_state["feedback_rules"])
                time.sleep(1.5)
                
                res_url = deba_url.replace("DebaTable", "RaceMarkTable")
                res_text = fetch_text_from_url(res_url)
                if res_text:
                    analysis = analyze_race_result(res_text, previous_prediction=mock_pred)
                    if "error" not in analysis and analysis.get('suggested_rule'):
                        all_suggested_rules.append(analysis['suggested_rule'])
                time.sleep(1.5)
                progress_bar.progress(race_no / past_max_race)
            
            progress_text.text("✅ 全レースのバックテスト・トレーニングが完了しました！")
            
            if all_suggested_rules:
                with st.spinner("ルールを統合し、データベースに保存しています..."):
                    new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], "\n".join(all_suggested_rules))
                    st.session_state["feedback_rules"] = new_rules
                    save_rules(new_rules)
                st.success("🎉 バックテストで得た教訓をすべて `rules.json` に統合・保存しました！")
                st.rerun()
