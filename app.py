import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# ==========================================
# 0. 初期設定とAPI設定
# ==========================================
st.set_page_config(page_title="競馬AI ROIオプティマイザ", page_icon="🏇", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# 反省ルール（フィードバック）のセッション状態の初期化
if "feedback_rules" not in st.session_state:
    st.session_state["feedback_rules"] = [
        "【展開・位置取り】東京など直線が長いコースでは、上がり3F性能だけでなく、4角7番手以内の好位・中団に付けられる位置取りの馬を評価すること。",
        "【展開・ペース】逃げ馬（通過順1番手経験馬）が3頭以上競合する場合はハイペース必至と判断し、上がり3F最速級の差し・追込馬の勝率を引き上げ、逃げ馬は割り引くこと。"
    ]

# ==========================================
# 1. AI予測 & ルール統合ロジック
# ==========================================
def deduplicate_and_merge_rules(existing_rules, new_rule):
    """既存のルール一覧と新規ルールを比較し、同種のものがあれば統合、なければ新規追加する"""
    if not existing_rules:
        return [new_rule], "新規追加"

    system_prompt = """
    あなたは競馬AIのナレッジベース管理モジュールです。
    【既存のルール一覧】と【追加したい新しいルール】を分析し、以下を行ってください。

    1. もし【追加したい新しいルール】が【既存のルール一覧】のいずれかと「内容が重複している」「同種の事象を扱っている」場合:
       - 類似ルールをより精緻・包括的な1つのルール文に「統合（アップデート）」してください。
    2. もし完全に「新しい観点のルール」である場合:
       - 既存ルール群をそのまま維持し、末尾に【追加したい新しいルール】を追加してください。

    以下のJSONフォーマットのみを出力してください。
    {
      "status": "MERGED" (統合・更新した場合) または "ADDED" (新規追加した場合),
      "merged_rules": ["ルール1", "ルール2", ...],
      "reason": "どのような処理を行ったかの説明（例: 第1条の展開ルールと同種だったため統合更新しました）"
    }
    """
    
    prompt_input = f"【既存のルール一覧】:\n" + "\n".join([f"{i+1}. {r}" for i, r in enumerate(existing_rules)]) + f"\n\n【追加したい新しいルール】:\n{new_rule}"
    
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        response = model.generate_content(prompt_input, generation_config=generation_config)
        res_json = json.loads(response.text)
        return res_json.get("merged_rules", existing_rules + [new_rule]), res_json.get("reason", "処理完了")
    except Exception as e:
        # エラー時はシンプルに重複を完全一致だけで防ぐ安全処理
        if new_rule not in existing_rules:
            return existing_rules + [new_rule], "新規追加（フォールバック）"
        return existing_rules, "同種ルールが既に存在します"

def get_gemini_prediction(combined_race_text, feedback_rules=None):
    """Gemini 3.5 Flash による反省ルールフィードバック反映型の統合解析"""
    
    rules_text = ""
    if feedback_rules:
        rules_text = "\n".join([f"- {rule}" for rule in feedback_rules])
    else:
        rules_text = "- 特になし（デフォルトロジック適用）"

    system_prompt = f"""
    あなたは競馬の確率論・展開読み・コース適性・調教時計解析に精通したプロのデータサイエンティストです。
    提供された【競馬新聞・出馬表データ】および【調教データ】を精査し、各馬の実質勝率（1着確率）、レース展開、荒れ度、印、馬連推奨を分析してください。

    【学習済みのフィードバック・反省ルール（最優先適用）】
    以下のルールは過去のレース結果検証から得られた重要改善点です。勝率・印の評価に強く反映させてください：
    {rules_text}

    【最重要解析ポイント】
    1. 調教内容・時計重視（コメントより具体的タイム・内容優先）
    2. 上がり3F性能（過去3走平均＆前走上がり）
    3. 同距離・コース適性実績
    4. 予想展開（ペース・隊列）

    以下のJSONフォーマットのみを出力してください。
    {{
      "race_name": "レース名",
      "volatility_level": "🔥🔥🔥 超高波乱",
      "volatility_reason": "理由",
      "pace_prediction": "🔥 ハイペース想定",
      "pace_reason": "展開解説",
      "predictions": [
        {{
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "馬名",
          "predicted_win_rate": 0.28,
          "current_odds": 3.2,
          "same_distance_eval": "🏆 同距離最高実績",
          "condition_trend": "🔥 急上昇（絶好調）",
          "recent_3f_history": [35.1, 34.5, 34.0, 33.6],
          "reason": "詳細理由"
        }}
      ],
      "recommended_umaren": [
        {{
          "combination": "1 - 2",
          "predicted_rate": 0.15,
          "current_odds": 10.5,
          "reason": "理由"
        }}
      ]
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        response = model.generate_content(combined_race_text, generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def analyze_race_result(result_text):
    """レース結果(result.html)のデータをGeminiで反省・検証分析"""
    system_prompt = """
    あなたは競馬予想AIの自己改善プログラムです。
    提供された【実際のレース結果（result.htmlのスクレイピングデータ）】を分析し、以下の内容を考察・反省してください。

    1. 1〜3着馬の共通点（上がり3Fタイム、通過順、人気、調教傾向など）
    2. 上がり3F最速馬が勝ち切れたか、展開（逃げ・差し）の有利不利はどうだったか
    3. 今後のAIプロンプト（予想重み付け）で改善・追加すべき具体的なルール文（1〜2文）

    以下のJSONフォーマットのみを出力してください。
    {
      "race_summary": "1〜3着の実際の着順とタイムまとめ",
      "winning_factors": "勝因・好走要因（上がり3F、展開など）",
      "ai_reflection": "AI予想ロジックの反省点と今後の改善・修正ポイント",
      "suggested_rule": "AIに学習させる具体的な追加ルール"
    }
    """
    try:
        model = genai.GenerativeModel('gemini-3.5-flash', system_instruction=system_prompt)
        generation_config = genai.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        response = model.generate_content(result_text, generation_config=generation_config)
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def calculate_kelly_bet(predicted_win_rate, odds, bankroll, kelly_fraction=0.25):
    expected_value = predicted_win_rate * odds
    if expected_value <= 1.0 or odds <= 1.0:
        return {"action": "見送り", "bet_amount": 0, "percentage": 0.0, "ev": round(expected_value, 2)}
    b = odds - 1.0
    p = predicted_win_rate
    q = 1.0 - p
    f = (b * p - q) / b
    adjusted_fraction = f * kelly_fraction
    raw_bet = bankroll * adjusted_fraction
    bet_amount = int(raw_bet // 100 * 100)
    if bet_amount == 0 and raw_bet > 0:
        bet_amount = 100
    return {
        "action": "買い" if bet_amount > 0 else "見送り",
        "bet_amount": max(bet_amount, 0),
        "percentage": round(max(adjusted_fraction, 0) * 100, 2),
        "ev": round(expected_value, 2)
    }

# ==========================================
# 2. UI画面構成
# ==========================================
st.sidebar.title("🏇 AI競馬 ROIシステム")
page = st.sidebar.radio("メニュー", ["🛠️ レース分析＆AI予測", "🏁 結果検証＆プロンプト学習", "📈 バックテスト分析"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 適用中のフィードバックルール")
if st.session_state["feedback_rules"]:
    for i, r in enumerate(st.session_state["feedback_rules"]):
        st.sidebar.caption(f"{i+1}. {r}")
else:
    st.sidebar.caption("適用中のカスタムルールはありません")

if st.sidebar.button("全ルールを初期化"):
    st.session_state["feedback_rules"] = []
    st.rerun()

if page == "🛠️ レース分析＆AI予測":
    st.title("🎯 AI全自動分析（学習フィードバック適用中）")
    default_url = "https://race.netkeiba.com/race/newspaper.html?m=riot-shutuba-past&race_id=202605040401"
    target_url = st.text_input("出馬表 / 競馬新聞URLを入力", value=default_url)
    
    if st.button("総合AI分析を実行"):
        if target_url and GOOGLE_API_KEY:
            headers = {'User-Agent': 'Mozilla/5.0'}
            with st.spinner('学習済みルールを適用して分析中...'):
                try:
                    res_shutuba = requests.get(target_url, headers=headers, timeout=10)
                    soup_shutuba = BeautifulSoup(res_shutuba.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_shutuba(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    race_title = soup_shutuba.find('div', class_='RaceName') or soup_shutuba.find('h1', class_='RaceName')
                    race_name = race_title.get_text(strip=True) if race_title else "対象レース"
                    newspaper_table = soup_shutuba.find('div', id='RaceNewspaper') or soup_shutuba.find('table', class_='Shutuba_Table') or soup_shutuba.find('table')
                    shutuba_text = newspaper_table.get_text(separator=' ', strip=True) if newspaper_table else soup_shutuba.get_text(separator=' ', strip=True)

                    oikiri_url = target_url.replace("newspaper.html", "oikiri.html").replace("shutuba.html", "oikiri.html")
                    res_oikiri = requests.get(oikiri_url, headers=headers, timeout=10)
                    soup_oikiri = BeautifulSoup(res_oikiri.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_oikiri(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    oikiri_table = soup_oikiri.find('table', class_='Oikiri_Table') or soup_oikiri.find('div', class_='OikiriData')
                    oikiri_text = oikiri_table.get_text(separator=' ', strip=True) if oikiri_table else soup_oikiri.get_text(separator=' ', strip=True)

                    combined_race_text = f"【レース名】: {race_name}\n\n【出馬表】:\n{shutuba_text}\n\n【調教】:\n{oikiri_text}"
                    
                    res = get_gemini_prediction(combined_race_text, feedback_rules=st.session_state["feedback_rules"])
                    
                    if "error" not in res:
                        st.subheader(f"📊 【{res.get('race_name', race_name)}】 分析結果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"⚡ **レース波乱度:** {res.get('volatility_level', '')}\n\n{res.get('volatility_reason', '')}")
                        with col2:
                            st.success(f"🏁 **予想展開:** {res.get('pace_prediction', '')}\n\n{res.get('pace_reason', '')}")
                        
                        preds = res.get("predictions", [])
                        if preds:
                            table_preds = []
                            for item in preds:
                                rate = item.get("predicted_win_rate", 0)
                                odds = item.get("current_odds", 1.0)
                                kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                                table_preds.append({
                                    "印": item.get("mark", "-"),
                                    "馬番": item.get("horse_number", "-"),
                                    "馬名": item.get("horse_name", "-"),
                                    "AI予測勝率": f"{rate*100:.1f}%",
                                    "単勝オッズ": f"{odds}倍",
                                    "同距離適性": item.get("same_distance_eval", "-"),
                                    "調子トレンド": item.get("condition_trend", "-"),
                                    "期待値(EV)": kelly["ev"],
                                    "単勝判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                    "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                    "理由": item.get("reason", "")
                                })
                            st.dataframe(pd.DataFrame(table_preds), use_container_width=True)
                except Exception as e:
                    st.error(f"エラー: {e}")

elif page == "🏁 結果検証＆プロンプト学習":
    st.title("🏁 レース結果検証 ＆ AIプロンプト自動重複排除学習")
    st.write("過去のレース結果（`result.html`）を解析し、得られた反省ルールを既存のルールと重複比較しながら自動統合・追加します。")
    
    result_url = st.text_input("レース結果URLを入力", value="https://race.netkeiba.com/race/result.html?race_id=202605040311")
    
    if st.button("レース結果を検証・反省する"):
        if result_url and GOOGLE_API_KEY:
            headers = {'User-Agent': 'Mozilla/5.0'}
            with st.spinner('レース結果を取得して分析中...'):
                try:
                    res_result = requests.get(result_url, headers=headers, timeout=10)
                    soup_result = BeautifulSoup(res_result.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_result(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    result_table = soup_result.find('table', id='All_Result_Table') or soup_result.find('table')
                    result_text = result_table.get_text(separator=' ', strip=True) if result_table else soup_result.get_text(separator=' ', strip=True)
                    
                    analysis = analyze_race_result(result_text)
                    
                    if "error" not in analysis:
                        st.subheader("📝 AIのレース振り返り＆反省レポート")
                        st.write(f"**【1〜3着結果】:** {analysis.get('race_summary', '')}")
                        st.success(f"💡 **【実際の勝因・好走要因】:**\n{analysis.get('winning_factors', '')}")
                        st.warning(f"🔧 **【AIの反省点】:**\n{analysis.get('ai_reflection', '')}")
                        
                        sug_rule = analysis.get('suggested_rule', '')
                        st.session_state["current_sug_rule"] = sug_rule
                        st.markdown(f"### 🧠 AIが提案する新フィードバックルール:\n> **{sug_rule}**")
                    else:
                        st.error(f"分析エラー: {analysis['error']}")
                except Exception as e:
                    st.error(f"取得エラー: {e}")

    # ボタン押下エリア（セッションをまたいで追加できるように独立）
    if "current_sug_rule" in st.session_state and st.session_state["current_sug_rule"]:
        if st.button("➕ 重複チェックを行ってルールを保存・統合する"):
            with st.spinner('既存ルールとの重複チェック＆統合判定中...'):
                new_rules, reason = deduplicate_and_merge_rules(
                    st.session_state["feedback_rules"], 
                    st.session_state["current_sug_rule"]
                )
                st.session_state["feedback_rules"] = new_rules
                st.success(f"処理完了: {reason}")
                st.session_state["current_sug_rule"] = None
                st.rerun()

elif page == "📈 バックテスト分析":
    st.title("📈 バックテスト結果")
    st.write("過去データのシミュレーション表示エリアです。")
