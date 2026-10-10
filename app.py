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
        "東京ダート・芝の長直線コースでは、上がり3F性能だけでなく、4角で後ろすぎない(4角7番手以内)展開・位置取りも加点すること。",
        "同型（逃げ馬）が3頭以上競合する場合は、差し・追込馬の期待値を引き上げ、逃げ馬の勝率を割り引くこと。"
    ]

# ==========================================
# 1. AI予測 & 資金管理ロジック
# ==========================================
def get_gemini_prediction(combined_race_text, feedback_rules=None):
    """Gemini 3.5 Flash による反省ルールフィードバック反映型の統合解析"""
    
    # 蓄積された学習ルールの整形
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
          "reason": "詳細理由（学習ルールの適用有無も触れる）"
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
      "suggested_rule": "AIに学習させる具体的な追加ルール（例: 東京芝1600mでは上がり3F上位かつ4角5番手以内の先行馬の評価を1.2倍にすること）"
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

# サイドバーに学習済みルールの表示
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 適用中のフィードバックルール")
if st.session_state["feedback_rules"]:
    for i, r in enumerate(st.session_state["feedback_rules"]):
        st.sidebar.caption(f"{i+1}. {r}")
else:
    st.sidebar.caption("適用中のカスタムルールはありません")

if st.sidebar.button("ルールを初期化"):
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
