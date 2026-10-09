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
def get_gemini_prediction(race_data_text):
    """Geminiで実在の出馬表データから印（◎◯△▲）と勝率・期待値を算出"""
    system_prompt = """
    あなたは競馬の確率論に精通したプロのデータサイエンティストです。
    以下に提供された【実際の出馬表テキストデータ】のみを解析し、出走馬の勝率と印、馬連推奨を分析してください。

    【重要制約事項】
    - 必ずテキスト内に存在する「実際の馬番」と「実際の馬名」のみを使用してください。架空の馬名やデータを創作することは厳禁です。
    - 単勝オッズが取得できている場合はその数字を使用し、不明な場合は実力に応じた想定オッズで計算してください。

    印の定義:
    ◎: 本命（勝ち切る確率が最も高い馬）
    ◯: 対抗（2番手に高い馬）
    ▲: 穴馬（オッズに対して勝率が高く期待値が大きい馬）
    △: ひも（2・3着候補）

    以下のJSONフォーマットのみを出力してください。
    {
      "race_name": "レース名（例: 東京1R 2歳未勝利）",
      "predictions": [
        {
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.25,
          "current_odds": 3.5,
          "reason": "本命の評価理由"
        },
        {
          "mark": "◯",
          "horse_number": 2,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.18,
          "current_odds": 4.8,
          "reason": "対抗の評価理由"
        },
        {
          "mark": "▲",
          "horse_number": 5,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.12,
          "current_odds": 12.0,
          "reason": "穴馬の評価理由"
        },
        {
          "mark": "△",
          "horse_number": 8,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.08,
          "current_odds": 15.0,
          "reason": "ひもの評価理由"
        }
      ],
      "recommended_umaren": [
        {
          "combination": "1 - 2",
          "predicted_rate": 0.12,
          "current_odds": 10.5,
          "reason": "◎と◯の堅実な組み合わせ"
        },
        {
          "combination": "1 - 5",
          "predicted_rate": 0.07,
          "current_odds": 25.0,
          "reason": "◎と▲の高期待値組み合わせ"
        }
      ]
    }
    """
    try:
        model = genai.GenerativeModel(
            'gemini-3.5-flash',
            system_instruction=system_prompt
        )
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
        response = model.generate_content(
            race_data_text,
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
    bet_amount = int((bankroll * adjusted_fraction) // 100 * 100)
    
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
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

if page == "🛠️ レース分析＆AI予測":
    st.title("🎯 AI印別評価・勝率分析＆馬連期待値")
    st.write("netkeibaの出馬表URLを入力すると、出走馬を正確に読み込んで分析します。")
    
    target_url = st.text_input("出馬表URLを入力", value="https://race.netkeiba.com/race/shutuba.html?race_id=202605040301&rf=race_list")
    
    if st.button("AI予想・勝率分析を実行"):
        if not target_url:
            st.warning("URLを入力してください。")
        elif not GOOGLE_API_KEY:
            st.error("APIキーが設定されていません。Streamlit CloudのSecretsを確認してください。")
        else:
            with st.spinner('1/2 出馬表テーブルのみをピンポイント取得中...'):
                try:
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    response = requests.get(target_url, headers=headers, timeout=10)
                    response.encoding = 'euc-jp'
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 出馬表テーブルだけをピンポイント抽出
                    shutuba_table = soup.find('table', class_='Shutuba_Table')
                    race_title = soup.find('div', class_='RaceName')
                    race_name = race_title.get_text(strip=True) if race_title else "対象レース"
                    
                    if shutuba_table:
                        race_text = f"レース名: {race_name}\n" + shutuba_table.get_text(separator=' ', strip=True)
                    else:
                        # テーブルが特定できない場合はテキスト全体から不要部分を削除して代用
                        for script in soup(["script", "style", "header", "footer"]):
                            script.extract()
                        race_text = soup.get_text(separator=' ', strip=True)
                    
                    st.success("正確な出馬表データを取得！Geminiで分析中...")
                    
                    with st.spinner('2/2 勝率＆推奨購入額を算出中...'):
                        res = get_gemini_prediction(race_text)
                        
                        if "error" in res:
                            st.error(f"分析エラー: {res['error']}")
                        else:
                            st.subheader(f"📊 【{res.get('race_name', race_name)}】 分析結果")
                            
                            # 1. 各馬の印と勝率テーブル
                            st.markdown("### 🏇 出走馬・印別予測勝率 (◎・◯・▲・△)")
                            preds = res.get("predictions", [])
                            if preds:
                                table_preds = []
                                for item in preds:
                                    mark = item.get("mark", "-")
                                    num = item.get("horse_number", "-")
                                    name = item.get("horse_name", "-")
                                    rate = item.get("predicted_win_rate", 0)
                                    odds = item.get("current_odds", 1.0)
                                    
                                    kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                                    
                                    table_preds.append({
                                        "印": mark,
                                        "馬番": num,
                                        "馬名": name,
                                        "AI予測勝率": f"{rate*100:.1f}%",
                                        "単勝オッズ": f"{odds}倍",
                                        "期待値(EV)": kelly["ev"],
                                        "単勝判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                        "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                        "評価根拠": item.get("reason", "")
                                    })
                                df_preds = pd.DataFrame(table_preds)
                                st.dataframe(df_preds, use_container_width=True)
                            
                            # 2. 馬連のおすすめ
                            st.markdown("### 🎟️ おすすめ馬連ペア（期待値順）")
                            umaren_list = res.get("recommended_umaren", [])
                            if umaren_list:
                                table_umaren = []
                                for item in umaren_list:
                                    combo = item.get("combination")
                                    rate = item.get("predicted_rate", 0)
                                    odds = item.get("current_odds", 1.0)
                                    
                                    kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                                    
                                    table_umaren.append({
                                        "馬連ペア": combo,
                                        "的中確率": f"{rate*100:.1f}%",
                                        "想定オッズ": f"{odds}倍",
                                        "期待値(EV)": kelly["ev"],
                                        "馬連判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                        "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                        "理由": item.get("reason", "")
                                    })
                                df_umaren = pd.DataFrame(table_umaren)
                                st.dataframe(df_umaren, use_container_width=True)

                except Exception as e:
                    st.error(f"処理エラー: {e}")

elif page == "📈 バックテスト分析":
    st.title("📈 バックテスト結果")
    st.write("シミュレーション結果を表示します。")
