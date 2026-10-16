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
st.set_page_config(page_title="高知ファイナル特化 AIオプティマイザ", page_icon="🏇", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 永続化（JSONファイル保存）ロジック
# ==========================================
RULES_FILE = "rules.json"
DEFAULT_RULES = [
    "【一発逆転ファイナル専用・近走無視】ファイナルレースは近走不振馬の集まりであるため、直近の着順や単純な調子トレンドは完全に無視し、「過去に同距離・同条件で出した持ち時計（ベストタイム）」の比較を最優先すること。",
    "【高知の馬場バイアス】高知競馬場は内側の砂が深く重たいため、砂を被らずにスムーズに外を回せる「外枠の先行・差し馬」の評価を大幅に引き上げ、最内枠の差し馬は割り引くこと。",
    "【展開利・ハイペース想定】ファイナルは先行争いが激化してハイペースによる「前崩れ」が起きやすいため、過去に上がり3F上位の脚を使えている穴馬の期待値(EV)を高く見積もること。"
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
    あなたは高知競馬・ファイナルレース特化AIのナレッジベース管理モジュールです。
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
    あなたは高知競馬の「一発逆転ファイナルレース（記者選抜）」のオッズの歪み・展開読み・時計解析に精通したプロデータサイエンティストです。
    提供された【出馬表データ】を精査し、各馬の勝率、レース展開、荒れ度、印、推奨買い目を分析してください。
    
    【🚨絶対厳守】
    1. オッズの捏造禁止。見つからない場合は `0.0` とすること。
    2. 馬名は実在するもののみ使用。印(◎◯▲△)は最低4〜5頭出力し、人気薄（穴馬）を積極的に組み込むこと。
    3. volatility_score はファイナルのため基本的に高め(2または3)を想定すること。
    
    【学習済みルール（最優先）】\n{rules_text}
    
    JSONフォーマットのみを出力:
    {{
      "race_name": "レース名", 
      "volatility_score": 3,
      "volatility_level": "🔥🔥🔥 超高波乱(ファイナル特有)", 
      "volatility_reason": "理由",
      "pace_prediction": "展開予測", 
      "pace_reason": "理由",
      "predictions": [ {{"mark": "◎", "horse_number": 1, "horse_name": "馬名", "predicted_win_rate": 0.15, "current_odds": 0.0, "reason": "理由"}} ],
      "recommended_umaren": [ {{"combination": "1-2", "predicted_rate": 0.08, "current_odds": 0.0, "reason": "理由"}} ],
      "recommended_sanrenpuku": [ {{"combination": "1-2-3", "predicted_rate": 0.03, "current_odds": 0.0, "reason": "理由"}} ]
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
    あなたは高知競馬・ファイナルレース予想AIの自己改善プログラムです。実際のレース結果を分析してください。
    {compare_prompt}
    今後のファイナルレース特化ロジックの精度を上げるための追加ルール文を提案してください。
    また、実際の「馬連の的中組み合わせ（馬番）」と「その配当金額」を正確に抽出してください。
    
    JSONのみ出力: 
    {{
      "race_summary": "結果", 
      "actual_umaren": "実際の馬連組み合わせ (例: 1-2)",
      "actual_umaren_payout": "実際の馬連配当 (例: 1,500円)",
      "winning_factors": "勝因(ファイナル特有のバイアスや展開など)", 
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

def find_final_race_url(base_url, is_result=False):
    """12Rから順番に遡って「ファイナル」が含まれるレースのURLを探す"""
    search_base = base_url.replace("DebaTable", "RaceMarkTable") if is_result else base_url.replace("RaceMarkTable", "DebaTable")
    
    # 12Rから9Rまで遡って検索（通常は11Rか12R）
    for race_no in [12, 11, 10, 9]:
        current_url = re.sub(r'k_raceNo=\d+', f'k_raceNo={race_no}', search_base)
        text = fetch_text_from_url(current_url)
        if text and "ファイナル" in text:
            return current_url, text, race_no
        time.sleep(0.5)
    return None, None, None

# ==========================================
# 2. UI画面構成
# ==========================================
st.sidebar.title("🏇 高知ファイナル AI ROI")
page = st.sidebar.radio("メニュー", ["🛠️ ファイナル単体 分析", "🔄 当日/翌日 ファイナル予想＆抽出", "📈 過去データ バックテスト自動学習"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 適用中ルール (ファイナル特化)")
if st.session_state["feedback_rules"]:
    for i, r in enumerate(st.session_state["feedback_rules"]): st.sidebar.caption(f"{i+1}. {r}")
if st.sidebar.button("ルール初期化"):
    st.session_state["feedback_rules"] = DEFAULT_RULES
    save_rules(DEFAULT_RULES)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("💾 ルールのバックアップと復元")
rules_json_str = json.dumps(st.session_state["feedback_rules"], ensure_ascii=False, indent=2)
st.sidebar.download_button(label="📥 ルールをPCに保存", data=rules_json_str, file_name="kochi_final_rules.json", mime="application/json")
uploaded_file = st.sidebar.file_uploader("📤 保存したルールを復元", type=["json"])
if uploaded_file is not None:
    try:
        loaded_rules = json.load(uploaded_file)
        if isinstance(loaded_rules, list):
            st.session_state["feedback_rules"] = loaded_rules
            save_rules(loaded_rules)
            st.sidebar.success("復元完了！")
    except Exception:
        st.sidebar.error("読み込み失敗")

# 高知競馬の競馬場コードは k_babaCode=31
default_kochi_url = "https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=2026%2F10%2F17&k_raceNo=1&k_babaCode=31"

if page == "🛠️ ファイナル単体 分析":
    st.title("🎯 高知ファイナル 直指定分析")
    st.write("ファイナルレースのURLを直接入力して分析します。")
    target_url = st.text_input("出馬表URL (高知ファイナル)", value=default_kochi_url)
    
    if st.button("AI予想を実行"):
        with st.spinner("ファイナルレースの波乱要因を解析中..."):
            text = fetch_text_from_url(target_url)
            res = get_gemini_prediction(text, st.session_state["feedback_rules"])
            if "error" not in res:
                st.session_state["last_prediction"]["single"] = res
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
                            "印": item.get("mark", "-"), "馬番": item.get("horse_number", "-"), "馬名": item.get("horse_name", "-"),
                            "AI予測勝率": f"{rate*100:.1f}%", "単勝オッズ": f"{odds}倍" if odds > 0 else "取得不可",
                            "期待値(EV)": kelly["ev"] if odds > 0 else "-", "単勝判定": kelly["action"], "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "-",
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
                            "馬連ペア": combo, "的中確率": f"{rate*100:.1f}%", "想定オッズ": f"{odds}倍" if odds > 0 else "取得不可",
                            "期待値(EV)": kelly["ev"] if odds > 0 else "-", "馬連判定": kelly["action"], "推奨額": f"¥{kelly['bet_amount']:,}" if odds > 0 else "-"
                        })
                    st.dataframe(pd.DataFrame(table_umaren), use_container_width=True)

elif page == "🔄 当日/翌日 ファイナル予想＆抽出":
    st.title("🔄 高知ファイナル自動検索 ＆ AI予想・結果照合")
    st.write("高知競馬のURL（何Rでも可）を入力すると、AIが**自動でファイナルレースを探し出して**分析と検証を行います。")
    base_url = st.text_input("出馬表 または 結果URLを入力 (高知)", value=default_kochi_url)
    
    col1, col2 = st.columns(2)
    if col1.button("🚀 自動検索 ＆ ファイナル予想"):
        with st.spinner("ファイナルレースを自動検索中..."):
            final_url, text, race_no = find_final_race_url(base_url, is_result=False)
            
            if final_url and text:
                st.success(f"🎯 {race_no}R がファイナルレースであることを検知しました！解析を開始します。")
                res = get_gemini_prediction(text, st.session_state["feedback_rules"])
                st.session_state["last_prediction"][f"final_{base_url}"] = res
                
                if "error" not in res:
                    st.markdown(f"### 🏇 【{race_no}R】ファイナル予想結果")
                    st.write(f"**波乱度:** {res.get('volatility_level')} / **展開:** {res.get('pace_prediction')}")
                    
                    st.markdown("#### 🎟️ AI推奨 馬連")
                    all_umaren_preds = []
                    for u in res.get("recommended_umaren", []):
                        all_umaren_preds.append({
                            "馬連ペア": u.get("combination", "-"), "的中確率": f"{u.get('predicted_rate', 0)*100:.1f}%", "推奨理由": u.get("reason", "")
                        })
                    if all_umaren_preds:
                        st.dataframe(pd.DataFrame(all_umaren_preds), use_container_width=True)
            else:
                st.error("指定された日のデータに「ファイナル」レースが見つかりませんでした。")

    if col2.button("🏁 レース終了後：ファイナル結果照合＆学習"):
        with st.spinner("ファイナルレースの結果を自動検索・照合中..."):
            final_url, text, race_no = find_final_race_url(base_url, is_result=True)
            
            if final_url and text:
                prev_pred = st.session_state["last_prediction"].get(f"final_{base_url}", None)
                analysis = analyze_race_result(text, previous_prediction=prev_pred)
                
                if "error" not in analysis:
                    st.success(f"🎯 {race_no}R ファイナルレースの結果を照合しました！")
                    
                    actual_umaren = analysis.get("actual_umaren", "-")
                    actual_payout = analysis.get("actual_umaren_payout", "-")
                    ai_umaren_list = [u.get("combination") for u in prev_pred.get("recommended_umaren", [])] if prev_pred else []
                    ai_umaren_str = ", ".join([str(a) for a in ai_umaren_list]) if ai_umaren_list else "予想なし"
                    is_hit = check_umaren_hit(ai_umaren_list, actual_umaren)
                    hit_mark = "🎯 的中" if is_hit else "❌ 不的中"
                    
                    st.markdown("### 📊 馬連 予想 vs 結果 比較")
                    comp_df = pd.DataFrame([{
                        "AI予想馬連": ai_umaren_str, "実際の馬連結果": actual_umaren, "配当": actual_payout, "結果": hit_mark
                    }])
                    st.dataframe(comp_df, use_container_width=True)
                    
                    rule = analysis.get('suggested_rule')
                    if rule:
                        st.info(f"💡 **AIからの新ルール提案:** {rule}")
                        new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], rule)
                        st.session_state["feedback_rules"] = new_rules
                        save_rules(new_rules)
                        st.success("✅ ファイナル特化の学びを `rules.json` に永続保存しました！")
            else:
                st.error("指定された日のデータに「ファイナル」レースの結果が見つかりませんでした。")

elif page == "📈 過去データ バックテスト自動学習":
    st.title("📈 高知ファイナル専用 バックテスト自動学習マシーン")
    st.write("過去のレース日（何Rでも可）のURLを複数入力できます。AIが自動でファイナルレースだけを探し出し、一気に学習します。")
    
    # 複数日のURLを一気に入力できるようにテキストエリア化
    past_urls_input = st.text_area("過去レースのURLを改行で入力 (高知)", placeholder="https://...k_raceDate=2026%2F09%2F01...\nhttps://...k_raceDate=2026%2F09%2F02...")
    
    if st.button("🧠 ファイナルレース限定で過去データを全自動学習する"):
        urls = [u.strip() for u in past_urls_input.split('\n') if u.strip()]
        if not urls:
            st.error("URLを入力してください。")
        else:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            all_suggested_rules = []
            
            for i, base_url in enumerate(urls):
                progress_text.text(f"処理中 ({i+1}/{len(urls)}日目): ファイナルレースを自動検索中...")
                
                # ファイナルレースを自動検索
                final_url, deba_text, race_no = find_final_race_url(base_url, is_result=False)
                
                if final_url and deba_text:
                    progress_text.text(f"処理中 ({i+1}/{len(urls)}日目): {race_no}R ファイナルの出馬表解析 ＆ 結果照合中...")
                    mock_pred = get_gemini_prediction(deba_text, st.session_state["feedback_rules"])
                    time.sleep(1.5)
                    
                    res_url = final_url.replace("DebaTable", "RaceMarkTable")
                    res_text = fetch_text_from_url(res_url)
                    if res_text:
                        analysis = analyze_race_result(res_text, previous_prediction=mock_pred)
                        if "error" not in analysis and analysis.get('suggested_rule'):
                            all_suggested_rules.append(analysis['suggested_rule'])
                            
                        with st.expander(f"📝 {i+1}日目 ({race_no}R ファイナル) ログ", expanded=False):
                            st.write(f"**結果:** {analysis.get('race_summary', '')}")
                            st.info(f"**抽出ルール:** {analysis.get('suggested_rule', '')}")
                    time.sleep(1.5)
                else:
                    st.warning(f"{i+1}日目のURLにファイナルレースが見つかりませんでした。スキップします。")
                    
                progress_bar.progress((i + 1) / len(urls))
            
            progress_text.text("✅ 全日程のファイナル限定バックテストが完了しました！")
            
            if all_suggested_rules:
                with st.spinner("ルールを統合し、データベースに保存しています..."):
                    new_rules, _ = deduplicate_and_merge_rules(st.session_state["feedback_rules"], "\n".join(all_suggested_rules))
                    st.session_state["feedback_rules"] = new_rules
                    save_rules(new_rules)
                st.success("🎉 ファイナルレースの教訓をすべて `rules.json` に統合・保存しました！AIがファイナルマスターに近づきました。")
                st.rerun()
