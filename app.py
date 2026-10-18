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
st.set_page_config(page_title="名古屋競馬 AI ROIオプティマイザ", page_icon="🏇", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 永続化（JSONファイル保存）ロジック
# ==========================================
RULES_FILE = "rules.json"
DEFAULT_RULES = [
    "【名古屋・新コース形態】2022年移転後の新コースは直線が300mと西日本最長クラスのため、旧コースのような前残り一辺倒ではなく、過去3走平均の上がり3F上位の差し・追込馬の勝率・期待値を高く評価すること。",
    "【名古屋・1500m戦の展開】主力距離である1500m戦は最初のコーナーまで距離がありペースが速くなりやすいため、逃げ・先行馬が競り合う場合は中団待機の馬を大幅に加点すること。",
    "【名古屋・920m短距離】920mの超短距離戦では圧倒的にテンのスピード（スタートからのダッシュ力・前走の通過順1〜2番手）を最優先で評価すること。",
    "【馬場状態・時計勝負】重・不良馬場では極端に時計が速くなる傾向があるため、良馬場の上がり性能よりも「持ち時計（走破タイム実績）」の優秀な馬を強力加点すること。"
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
    あなたは地方名古屋競馬AIのナレッジベース管理モジュールです。
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
    あなたは地方競馬・名古屋競馬場の確率論・展開読み・時計解析に精通したプロデータサイエンティストです。
    提供された【出馬表データ】を精査し、各馬の勝率、レース展開、荒れ度、印、推奨買い目を分析してください。
    
    【🚨絶対厳守の追加事項】
    1. オッズの捏造禁止。見つからない場合は `0.0` とすること。
    2. 印(◎◯▲△)は最低4頭出力。
    3. volatility_score は必ず 1, 2, 3 のいずれかの数値を出力すること (1: 堅実, 2: 中波乱, 3: 大荒れ)。
    4. 【調子の算出】: オッズが0.0の場合でも評価できるよう、純粋に「過去の競走成績」から算出した該当レースにおける各馬の「調子評価・純粋な能力評価」を記述。
    5. 【レース傾向と4角位置】: 「当該レースの展開傾向」と、各馬の「最終コーナー（4角）を通過する際の位置取り確率（%表記）」を出力。
    
    【学習済みルール（最優先）】\n{rules_text}
    
    JSONフォーマットのみを出力:
    {{
      "race_name": "レース名", 
      "volatility_score": 2,
      "volatility_level": "🔥🔥 中波乱", 
      "volatility_reason": "波乱の理由",
      "race_trend": "当該レースの傾向",
      "pace_prediction": "ハイペース想定", 
      "pace_reason": "展開理由",
      "predictions": [ 
        {{
          "mark": "◎", "horse_number": 1, "horse_name": "馬名", "predicted_win_rate": 0.3, "current_odds": 0.0, 
          "condition_eval": "S (近3走連続で上がり最速)", "corner_4_position": "先頭〜2番手 (確率: 75%)", "reason": "本命理由"
        }} 
      ],
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
    あなたは地方名古屋競馬予想AIの自己改善プログラムです。実際のレース結果を分析してください。
    {compare_prompt}
    
    今後の名古屋予想ロジック精度を上げるための追加ルール文を提案してください。
    
    【🚨重要考察：能力 vs オッズ】
    「AIが算出した純粋な能力評価・勝率予測」と「実際の結果（および最終オッズ・人気）」を比較してください。
    世間のオッズ（人気）よりも、AIの能力ベースの予想の方が結果を正確に捉えていたか（妙味があったか）、それともオッズ通りの実力決着でAIが過信していたかを分析し、「ability_vs_odds_insight」として出力してください。
    
    JSONのみ出力: 
    {{
      "race_summary": "結果", 
      "actual_umaren": "実際の馬連組み合わせ (例: 1-2)",
      "actual_umaren_payout": "実際の馬連配当 (例: 1,500円)",
      "winning_factors": "勝因(名古屋のコースバイアスなど)", 
      "ability_vs_odds_insight": "【能力評価 vs オッズ】の比較分析インサイト",
      "ai_reflection": "AI予測の反省点・ズレの原因", 
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
st.sidebar.title("🏇 名古屋競馬 AI ROI")
page = st.sidebar.radio("メニュー", ["🛠️ 1R単体 分析", "🔄 指定日一括 AI予想＆抽出", "📈 過去レース 予想＆結果シミュレーター"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 馬券購入ターゲット設定")
target_volatility = st.sidebar.radio(
    "狙うレースの波乱度",
    ["すべて対象", "中波乱以上(回収率重視)", "大荒れのみ(穴党)", "堅実のみ(本命党)"],
    index=1
)

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

st.sidebar.markdown("---")
st.sidebar.subheader("💾 ルールのバックアップと復元")
rules_json_str = json.dumps(st.session_state["feedback_rules"], ensure_ascii=False, indent=2)
st.sidebar.download_button(label="📥 現在のルールをPCに保存", data=rules_json_str, file_name="nagoya_rules_backup.json", mime="application/json")
uploaded_file = st.sidebar.file_uploader("📤 保存したルールを復元", type=["json"])
if uploaded_file is not None:
    try:
        loaded_rules = json.load(uploaded_file)
        if isinstance(loaded_rules, list):
            st.session_state["feedback_rules"] = loaded_rules
            save_rules(loaded_rules)
            st.sidebar.success("ルールを復元しました！")
    except Exception:
        st.sidebar.error("読み込み失敗")

default_nagoya_url = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=2026%2F10%2F19&k_raceNo=1&k_babaCode=24"

if page == "🛠️ 1R単体 分析":
    st.title("🎯 名古屋 単一レース分析")
    target_url = st.text_input("出馬表URL (名古屋)", value=default_nagoya_url)
    if st.button("AI予想を実行"):
        with st.spinner("名古屋コース特性と過去の成績から解析中..."):
            text = fetch_text_from_url(target_url)
            res = get_gemini_prediction(text, st.session_state["feedback_rules"])
            if "error" not in res:
                st.session_state["last_prediction"]["single"] = res
                score = res.get("volatility_score", 2)
                if is_target_race(score, target_volatility):
                    st.success("🟢 **このレースは設定条件に合致する【勝負レース】です！**")
                else:
                    st.warning("🔴 **このレースは見送りを推奨します。**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"⚡ **判定波乱度:** {res.get('volatility_level')}\n\n**波乱の理由:** {res.get('volatility_reason')}")
                    st.success(f"🏁 **展開予想:** {res.get('pace_prediction')}\n\n**展開理由:** {res.get('pace_reason')}")
                with col2:
                    st.warning(f"📈 **当該レースの傾向 (トラックバイアス):**\n\n{res.get('race_trend', 'データなし')}")
                
                st.markdown("### 🏇 出走馬・印別予測勝率 ＆ 成績に基づく調子評価")
                preds = res.get("predictions", [])
                if preds:
                    table_preds = []
                    for item in preds:
                        rate = item.get("predicted_win_rate", 0)
                        odds = item.get("current_odds", 0.0)
                        kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                        table_preds.append({
                            "印": item.get("mark", "-"), "馬番": item.get("horse_number", "-"), "馬名": item.get("horse_name", "-"),
                            "AI予測勝率": f"{rate*100:.1f}%", "調子評価": item.get("condition_eval", "-"), "4角位置予測(%)": item.get("corner_4_position", "-"),
                            "単勝オッズ": f"{odds}倍" if odds > 0 else "取得不可", "期待値(EV)": kelly["ev"] if odds > 0 else "-",
                            "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "オッズ確認要", "評価理由": item.get("reason", "")
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
                            "馬連ペア": combo, "的中確率": f"{rate*100:.1f}%", "想定オッズ": f"{odds}倍" if odds > 0 else "取得不可",
                            "期待値(EV)": kelly["ev"] if odds > 0 else "-", "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "-"
                        })
                    st.dataframe(pd.DataFrame(table_umaren), use_container_width=True)
            else:
                st.error("解析エラーが発生しました。")

elif page == "🔄 指定日一括 AI予想＆抽出":
    st.title("🔄 名古屋 1日全レース 一括AI予想 ＆ 結果照合")
    base_url = st.text_input("1Rの出馬表 または 結果URLを入力 (名古屋)", value=default_nagoya_url)
    max_race = st.slider("処理する最終レース番号", min_value=1, max_value=12, value=12)
    
    col1, col2 = st.columns(2)
    if col1.button("🚀 一括AI予想 ＆ 買い目リスト作成"):
        progress_bar = st.progress(0)
        target_races = []
        all_umaren_preds = []
        for race_no in range(1, max_race + 1):
            current_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', base_url.replace("RaceMarkTable", "DebaTable"))
            text = fetch_text_from_url(current_url)
            if text:
                res = get_gemini_prediction(text, st.session_state["feedback_rules"])
                st.session_state["last_prediction"][f"{race_no}R"] = res
                if "error" not in res:
                    score = res.get("volatility_score", 2)
                    if is_target_race(score, target_volatility):
                        target_races.append(f"{race_no}R ({res.get('volatility_level')})")
                    for u in res.get("recommended_umaren", []):
                        all_umaren_preds.append({
                            "レース": f"{race_no}R", "馬連ペア": u.get("combination", "-"), "的中確率": f"{u.get('predicted_rate', 0)*100:.1f}%", "推奨理由": u.get("reason", "")
                        })
            time.sleep(1.5)
            progress_bar.progress(race_no / max_race)
            
        st.success("✅ 一括予想完了！")
        st.header(f"💰 買うべき勝負レース一覧 (設定: {target_volatility})")
        if target_races:
            for tr in target_races: st.markdown(f"- **{tr}**")
        else:
            st.warning("設定条件に合致するレースがありませんでした。資金温存を推奨します。")
        st.header("🎟️ 本日のAI推奨 馬連一覧")
        if all_umaren_preds:
            st.dataframe(pd.DataFrame(all_umaren_preds), use_container_width=True)

    if col2.button("🏁 レース終了後：一括結果照合＆学習"):
        progress_bar = st.progress(0)
        all_suggested_rules = []
        comparison_results = []
        for race_no in range(1, max_race + 1):
            res_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', base_url.replace("DebaTable", "RaceMarkTable"))
            text = fetch_text_from_url(res_url)
            prev_pred = st.session_state["last_prediction"].get(f"{race_no}R", None)
            
            if text:
                analysis = analyze_race_result(text, previous_prediction=prev_pred)
                if "error" not in analysis:
                    if analysis.get('suggested_rule'):
                        all_suggested_rules.append(analysis['suggested_rule'])
                    
                    actual_umaren = analysis.get("actual_umaren", "-")
                    actual_payout = analysis.get("actual_umaren_payout", "-")
                    ai_umaren_list = [u.get("combination") for u in prev_pred.get("recommended_umaren", [])] if prev_pred else []
                    ai_umaren_str = ", ".join([str(a) for a in ai_umaren_list]) if ai_umaren_list else "予想なし"
                    is_hit = check_umaren_hit(ai_umaren_list, actual_umaren)
                    hit_mark = "🎯 的中" if is_hit else "❌ 不的中"
                    
                    comparison_results.append({
                        "レース": f"{race_no}R", "AI予想馬連": ai_umaren_str, "実際の馬連結果": actual_umaren, "配当": actual_payout, "結果": hit_mark
                    })
            time.sleep(1.5)
            progress_bar.progress(race_no / max_race)
            
        if comparison_results:
            st.header("📊 馬連 予想 vs 結果 比較レポート")
            st.dataframe(pd.DataFrame(comparison_results), use_container_width=True)
            
        if all_suggested_rules:
            new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], "\n".join(all_suggested_rules))
            st.session_state["feedback_rules"] = new_rules
            save_rules(new_rules)
            st.success("✅ 全レースの答え合わせと学習が完了し、`rules.json` に保存しました！")

elif page == "📈 過去レース 予想＆結果シミュレーター":
    st.title("📈 過去レース 予想＆結果シミュレーター")
    st.write("過去のレースURLを入力すると、「①AIが競走成績から事前予想」→「②実際の結果・オッズと照合」→「③能力評価とオッズの比較検証」を行います。")
    
    past_base_url = st.text_input("過去レース 1Rの出馬表URLを入力 (名古屋)", placeholder="例: .../DebaTable?k_raceDate=2026%2F09%2F01&k_raceNo=1&k_babaCode=24")
    past_max_race = st.slider("シミュレーションするレース数", min_value=1, max_value=12, value=12)
    
    if st.button("🧠 シミュレーション実行 ＆ ルール学習"):
        if "k_raceNo=" not in past_base_url:
            st.error("URLに `k_raceNo=` が含まれていません。")
        else:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            all_suggested_rules = []
            simulation_results = []
            
            for race_no in range(1, past_max_race + 1):
                progress_text.text(f"【{race_no}R】 出馬表から予想を生成中...")
                deba_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', past_base_url.replace("RaceMarkTable", "DebaTable"))
                deba_text = fetch_text_from_url(deba_url)
                
                mock_pred = None
                if deba_text:
                    mock_pred = get_gemini_prediction(deba_text, st.session_state["feedback_rules"])
                time.sleep(1.5)
                
                progress_text.text(f"【{race_no}R】 レース結果を取得・照合中...")
                res_url = deba_url.replace("DebaTable", "RaceMarkTable")
                res_text = fetch_text_from_url(res_url)
                
                if res_text:
                    analysis = analyze_race_result(res_text, previous_prediction=mock_pred)
                    if "error" not in analysis:
                        if analysis.get('suggested_rule'):
                            all_suggested_rules.append(analysis['suggested_rule'])
                        
                        actual_umaren = analysis.get("actual_umaren", "-")
                        actual_payout = analysis.get("actual_umaren_payout", "-")
                        ai_umaren_list = [u.get("combination") for u in mock_pred.get("recommended_umaren", [])] if mock_pred else []
                        ai_umaren_str = ", ".join([str(a) for a in ai_umaren_list]) if ai_umaren_list else "予想なし"
                        
                        is_hit = check_umaren_hit(ai_umaren_list, actual_umaren)
                        hit_mark = "🎯 的中" if is_hit else "❌ 不的中"
                        
                        simulation_results.append({
                            "レース": f"{race_no}R",
                            "AI推奨の馬連": ai_umaren_str,
                            "実際の馬連": actual_umaren,
                            "配当": actual_payout,
                            "結果判定": hit_mark
                        })
                        
                        with st.expander(f"📝 {race_no}R シミュレーション詳細 ({hit_mark})", expanded=False):
                            st.write(f"**【AIの事前展開予想】:** {mock_pred.get('pace_prediction', '') if mock_pred else '-'}")
                            st.write(f"**【実際の結果】:** {analysis.get('race_summary', '')}")
                            st.success(f"**【🔍 能力 vs オッズ 比較分析】:**\n{analysis.get('ability_vs_odds_insight', '分析なし')}")
                            st.warning(f"**【AIの反省・照合】:** {analysis.get('ai_reflection', '')}")
                            st.info(f"**【抽出されたルール】:** {analysis.get('suggested_rule', '')}")
                
                time.sleep(1.5)
                progress_bar.progress(race_no / past_max_race)
            
            progress_text.text("✅ シミュレーションが完了しました！")
            
            if simulation_results:
                st.markdown("---")
                st.header("📊 シミュレーション 馬連成績表")
                st.dataframe(pd.DataFrame(simulation_results), use_container_width=True)
            
            if all_suggested_rules:
                with st.spinner("今回のシミュレーションから得たルールを統合・保存中..."):
                    new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], "\n".join(all_suggested_rules))
                    st.session_state["feedback_rules"] = new_rules
                    save_rules(new_rules)
                st.success("🎉 シミュレーションで得た教訓を `rules.json` に保存しました！")
