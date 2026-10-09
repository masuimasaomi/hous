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
    """Gemini 3.5 Flash によるレース荒れ度・上がり3F・調教・過去走の超高精度統合解析"""
    system_prompt = """
    あなたは競馬の確率論・末脚（上がり3F）データ分析・波乱度予測に精通したプロのデータサイエンティスト・トラックマンです。
    提供された【競馬新聞・出馬表データ（過去3走成績・上がり3F・通過順含む）】および【調教データ】を精査し、各馬の実質勝率（1着確率）、レースの荒れ度、印、馬連推奨を分析してください。

    【最重要解析ポイント】
    1. レース荒れ度（波乱度）の判定:
       - 上位人気の信頼性、メンバー全体の能力差、過去3走の上がり3Fタイムの接戦度を評価し、レースの荒れやすさを判定してください。
       - 「超高波乱（大荒れ）」「高波乱」「中波乱」「堅調（本命堅い）」のいずれかで判定し、その理由（「オッズは割れているが決定打に欠ける」「本命馬の上がり3Fが圧倒的で堅い」など）を出力してください。
    2. 上がり3F（スパート力）評価（最優先加点）:
       - 新聞データに含まれる「前走上がり3Fタイム」「過去3走の上がり3F平均」「上がり順位（1位・2位等）」を算出し、メンバー内で上位（速い）馬に高い勝率評価・ポイントを与えてください。
    3. 調教データとの掛け合わせ & 馬名・馬番の正確性:
       - 最終追い切りの伸び脚と合体させ、テキスト内に存在する「実際の馬番」「実際の馬名」のみを使用してください。

    印の定義:
    ◎: 本命（メンバー最速クラスの上がり3F性能＋調教良好で最も勝ち切る確率が高い馬）
    ◯: 対抗（2番手に高い上がり性能・安定度を持つ馬）
    ▲: 穴馬（過去3走で速い上がり3Fを出しているが人気薄、または調教で末脚一変の期待値が高い穴馬）
    △: ひも（掲示板級の上がり性能を持ち2・3着候補の馬）

    以下のJSONフォーマットのみを出力してください。
    {
      "race_name": "レース名（例: 東京1R 2歳未勝利）",
      "volatility_level": "🔥🔥🔥 超高波乱",
      "volatility_reason": "上位人気のオッズは割れているが軸馬に決定打がなく、上がり3F差もわずか。混戦で大穴激走の可能性が高い難解なレース。",
      "predictions": [
        {
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.28,
          "current_odds": 3.2,
          "reason": "【新聞データ解析: 前走上がり最速33.8秒 / 過去3走上がり平均1位】 圧倒的な末脚性能。"
        },
        {
          "mark": "◯",
          "horse_number": 2,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.18,
          "current_odds": 4.8,
          "reason": "【新聞データ解析: 過去3走上がり平均2位】 安定して上がり上位をマーク。"
        },
        {
          "mark": "▲",
          "horse_number": 5,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.14,
          "current_odds": 15.0,
          "reason": "【新聞データ解析: 前走上がり2位で展開不向き】 新聞データから末脚性能が高く、大穴妙味あり。"
        },
        {
          "mark": "△",
          "horse_number": 8,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.06,
          "current_odds": 18.0,
          "reason": "【新聞データ解析: 過去3走安定】 連下・3着候補。"
        }
      ],
      "recommended_umaren": [
        {
          "combination": "1 - 2",
          "predicted_rate": 0.15,
          "current_odds": 10.5,
          "reason": "上がり最速◎と上がり上位◯の末脚信頼組み合わせ"
        },
        {
          "combination": "1 - 5",
          "predicted_rate": 0.09,
          "current_odds": 32.0,
          "reason": "上がり最速◎から展開一変期待の末脚穴▲への高期待値ペア"
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
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

if page == "🛠️ レース分析＆AI予測":
    st.title("🎯 AI全自動分析（荒れ度判定 ＆ Gemini 3.5 Flash）")
    st.write("出馬表・競馬新聞URLを入力すると、レースの荒れ度（波乱度）とおすすめ買い目を同時に自動判定します。")
    
    default_url = "https://race.netkeiba.com/race/newspaper.html?m=riot-shutuba-past&race_id=202605040401"
    target_url = st.text_input("出馬表 / 競馬新聞URLを入力", value=default_url)
    
    if st.button("レース荒れ度＋上がり3F＋調教をAI分析"):
        if not target_url:
            st.warning("URLを入力してください。")
        elif not GOOGLE_API_KEY:
            st.error("APIキーが設定されていません。Streamlit CloudのSecretsを確認してください。")
        else:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            with st.spinner('1/3 競馬新聞データを全取得中...'):
                try:
                    res_shutuba = requests.get(target_url, headers=headers, timeout=10)
                    soup_shutuba = BeautifulSoup(res_shutuba.content, 'html.parser', from_encoding='euc-jp')
                    
                    for tag in soup_shutuba(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    race_title = soup_shutuba.find('div', class_='RaceName') or soup_shutuba.find('h1', class_='RaceName')
                    race_name = race_title.get_text(strip=True) if race_title else "対象レース"
                    
                    newspaper_table = soup_shutuba.find('div', id='RaceNewspaper') or soup_shutuba.find('table', class_='Shutuba_Table') or soup_shutuba.find('table')
                    
                    shutuba_text = newspaper_table.get_text(separator=' ', strip=True) if newspaper_table else soup_shutuba.get_text(separator=' ', strip=True)
                except Exception as e:
                    st.error(f"データの取得に失敗しました: {e}")
                    st.stop()

            with st.spinner('2/3 追い切り・調教データを自動連動取得中...'):
                try:
                    oikiri_url = target_url.replace("newspaper.html", "oikiri.html").replace("shutuba.html", "oikiri.html")
                    res_oikiri = requests.get(oikiri_url, headers=headers, timeout=10)
                    soup_oikiri = BeautifulSoup(res_oikiri.content, 'html.parser', from_encoding='euc-jp')
                    
                    for tag in soup_oikiri(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    oikiri_table = soup_oikiri.find('table', class_='Oikiri_Table') or soup_oikiri.find('div', class_='OikiriData')
                    oikiri_text = oikiri_table.get_text(separator=' ', strip=True) if oikiri_table else soup_oikiri.get_text(separator=' ', strip=True)
                    st.success("競馬新聞高密度データ ＆ 調教データを正常取得！")
                except Exception as e:
                    oikiri_text = "※調教データの取得スキップ（データなし）"

            combined_race_text = f"【レース名】: {race_name}\n\n【競馬新聞・過去走・上がり3Fデータ】:\n{shutuba_text}\n\n【調教・追い切りデータ】:\n{oikiri_text}"

            with st.spinner('3/3 Gemini 3.5 Flash で荒れ度判定＆上がり3F分析を実行中...'):
                res = get_gemini_prediction(combined_race_text)
                
                if "error" in res:
                    st.error(f"分析エラー: {res['error']}")
                else:
                    st.subheader(f"📊 【{res.get('race_name', race_name)}】 分析結果")
                    
                    # 💥 レース荒れ度（波乱度）表示ブロック
                    vol_level = res.get("volatility_level", "判定不能")
                    vol_reason = res.get("volatility_reason", "")
                    
                    st.info(f"⚡ **レース波乱度:** {vol_level}\n\n**【混戦・波乱理由】:** {vol_reason}")
                    
                    st.markdown("### 🏇 出走馬・印別予測勝率 (新聞上がり3F＆調教反映)")
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
                                "新聞データ評価・上がり3F解析": item.get("reason", "")
                            })
                        df_preds = pd.DataFrame(table_preds)
                        st.dataframe(df_preds, use_container_width=True)
                    
                    st.markdown("### 🎟️ おすすめ馬連ペア（総合期待値順）")
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

elif page == "📈 バックテスト分析":
    st.title("📈 バックテスト結果")
    st.write("過去データのシミュレーション表示エリアです。")
