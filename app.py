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

# ==========================================
# 1. AI予測 & 資金管理ロジック
# ==========================================
def get_gemini_prediction(combined_race_text):
    """出馬表（過去3走成績込）＋調教データを統合解析し、勝率・期待値・印を算出"""
    system_prompt = """
    あなたは競馬の確率論・過去走分析・調教分析に精通したプロのデータサイエンティスト・トラックマンです。
    提供された【出馬表データ（各馬の過去3走成績含む）】および【調教・追い切りデータ】を精査し、各馬の実質勝率（1着確率）と印、馬連推奨を分析してください。

    【重要解析ポイント】
    1. 過去3走成績の分析:
       - 直近3走の「着順」「走破タイム」「着差」「クラス（未勝利/新馬/1勝クラス等）」「コース・距離適性」を評価。
       - 前走からの「距離短縮・延長」「ダート⇄芝の変更」「叩き2走目の上積み」を重視してください。
    2. 調教データとの掛け合わせ:
       - 最終追い切りのタイム（坂路・CW）や動きの良さを過去走データと合体させて状態面を判断。
    3. 馬名整合性:
       - 必ずテキスト内に存在する「実際の馬番」と「実際の馬名」のみを使用してください。

    印の定義:
    ◎: 本命（過去走実績＋近走の安定度＋調教全てにおいて最も勝ち切る確率が高い馬）
    ◯: 対抗（2番手に勝ち切る確率が高い馬）
    ▲: 穴馬（過去走の巻き返し要素や調教一変があり、オッズに対する期待値が大きい馬）
    △: ひも（過去3走で掲示板内などの実績があり2・3着候補の馬）

    以下のJSONフォーマットのみを出力してください。
    {
      "race_name": "レース名（例: 東京1R 2歳未勝利）",
      "predictions": [
        {
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.28,
          "current_odds": 3.2,
          "reason": "【過去3走: 前走同コース2着】 過去走の指数が高く、調教のCW伸び脚も抜群。軸として信頼"
        },
        {
          "mark": "◯",
          "horse_number": 2,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.18,
          "current_odds": 4.8,
          "reason": "【過去3走: 2走前3着】 前走は外枠で展開不向きも今回内枠で巻き返し濃厚"
        },
        {
          "mark": "▲",
          "horse_number": 5,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.12,
          "current_odds": 14.0,
          "reason": "【過去3走: 芝→ダート初挑戦】 今回初のダート変わり＋調教時計一変で一発の妙味あり"
        },
        {
          "mark": "△",
          "horse_number": 8,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.07,
          "current_odds": 18.0,
          "reason": "【過去3走: 安定して5着以内】 決め手に欠けるが連下・複勝圏なら優秀"
        }
      ],
      "recommended_umaren": [
        {
          "combination": "1 - 2",
          "predicted_rate": 0.14,
          "current_odds": 10.5,
          "reason": "過去走実績上位◎と◯の堅実組み合わせ"
        },
        {
          "combination": "1 - 5",
          "predicted_rate": 0.08,
          "current_odds": 28.0,
          "reason": "◎からダート変わり期待の穴▲への高期待値ペア"
        }
      ]
    }
    """
    try:
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=system_prompt
        )
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
        response = model.generate_content(
            combined_race_text,
            generation_config=generation_config
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def calculate_kelly_bet(predicted_win_rate, odds, bankroll, kelly_fraction=0.25):
    """ケリー基準による最適ベット額計算"""
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
page = st.sidebar.radio("メニュー", ["🛠️ レース分析＆AI予測", "📈 バックテスト分析"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数 (
