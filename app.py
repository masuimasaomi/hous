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
    """Gemini 3.5 Flash による同距離成績比較＋調子視覚化＋上がり3F＋調教の統合解析"""
    system_prompt = """
    あなたは競馬の確率論・コース適性・同距離データ分析・調教時計解析に精通したプロのデータサイエンティストです。
    提供された【競馬新聞・出馬表データ（過去走成績・同距離実績含む）】および【調教データ】を精査し、各馬の実質勝率（1着確率）、同距離競走における比較評価、調子トレンド、レース荒れ度、印、馬連推奨を分析してください。

    【最重要解析ポイント】
    1. 同距離競走（コース・距離適性）の比較解析（重要視）:
       - 今回のレース距離と同距離（または類似距離）における過去走の「走破タイム」「上がり3F」「着順実績」を抽出し、比較評価してください。
       - 同距離での実績が豊富な馬（「同距離得意」「同距離1位タイム保有」）には勝率ポイントを高く与え、距離延長・短縮による不安要素も考慮してください。
    2. 調子視覚化（過去走上がり3Fタイム推移）:
       - 印（◎, ◯, ▲）をつけた主要馬について、過去走（最大10走分）の「上がり3Fタイム（秒）」を数値配列として抽出してください。
    3. 調教データの評価（内容・数字重視）:
       - 具体的調教内容（コース・強さ・タイム・ラスト1F・併せ馬結果）を最優先で評価してください。
    4. 馬名・馬番の正確性:
       - 必ずテキスト内に存在する「実際の馬番」「実際の馬名」のみを使用してください。

    印の定義:
    ◎: 本命（同距離実績＋メンバー最速クラスの上がり3F性能＋調教最優秀で最も勝ち切る確率が高い馬）
    ◯: 対抗（2番手に同距離適性・上がり性能・調教内容が良い馬）
    ▲: 穴馬（同距離実績はあるが人気薄、または調教一変で激走期待の穴馬）
    △: ひも（掲示板級の同距離実績を持ち2・3着候補の馬）

    以下のJSONフォーマットのみを出力してください。
    {
      "race_name": "レース名（例: 東京1R 2歳未勝利）",
      "volatility_level": "🔥🔥🔥 超高波乱",
      "volatility_reason": "上位人気と同距離実績馬の差が小さく、調教時計も僅差の混戦。",
      "predictions": [
        {
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.28,
          "current_odds": 3.2,
          "same_distance_eval": "🏆 同距離最高実績",
          "condition_trend": "🔥 急上昇（絶好調）",
          "recent_3f_history": [35.1, 34.5, 34.0, 33.6],
          "reason": "【同距離実績: 1600mで勝率50% / 上がり最速33.6秒】 同距離での走破タイム・上がり性能がメンバー随一。【調教: CW 6F 81.2-11.2 (馬ナリ)】"
        },
        {
          "mark": "◯",
          "horse_number": 2,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.18,
          "current_odds": 4.8,
          "same_distance_eval": "◎ 同距離得意",
          "condition_trend": "安定ピーク",
          "recent_3f_history": [34.2, 34.0, 34.1, 33.9],
          "reason": "【同距離実績: 過去2勝がすべて同距離】 距離適性は文句なし。【調教: 坂路 52.4-11.8 (強め)】"
        },
        {
          "mark": "▲",
          "horse_number": 5,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.14,
          "current_odds": 15.0,
          "same_distance_eval": "▲ 同距離初挑戦",
          "condition_trend": "一発の妙味あり",
          "recent_3f_history": [36.2, 35.8, 34.2],
          "reason": "【同距離実績: 今回初の距離短縮】 上がり3F短縮傾向にあり、距離短縮で一発激走の妙味大"
        },
        {
          "mark": "△",
          "horse_number": 8,
          "horse_name": "実際の馬名",
          "predicted_win_rate": 0.06,
          "current_odds": 18.0,
          "same_distance_eval": "△ 同距離掲示板級",
          "condition_trend": "平行線",
          "recent_3f_history": [35.0, 34.8, 35.2],
          "reason": "【同距離実績: 同距離で3着2回】 連下・3着候補"
        }
      ],
      "recommended_umaren": [
        {
          "combination": "1 - 2",
          "predicted_rate": 0.15,
          "current_odds": 10.5,
          "reason": "同距離実績上位◎と◯の堅実軸太ペア"
        },
        {
          "combination": "1 - 5",
          "predicted_rate": 0.09,
          "current_odds": 32.0,
          "reason": "同距離実績◎から距離変更一変期待の穴▲への高期待値ペア"
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
    st.title("🎯 AI全自動分析（同距離実績比較 ＆ 調子バイタル視覚化）")
    st.write("出馬表・競馬新聞URLを入力すると、同距離レースでの過去実績やタイム・調子グラフを統合解析します。")
    
    default_url = "https://race.netkeiba.com/race/newspaper.html?m=riot-shutuba-past&race_id=202605040401"
    target_url = st.text_input("出馬表 / 競馬新聞URLを入力", value=default_url)
    
    if st.button("同距離比較＋調子バイタル＋調教＋オッズを総合AI分析"):
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

            with st.spinner('2/3 追い切り・調教タイム＆内容データを連動取得中...'):
                try:
                    oikiri_url = target_url.replace("newspaper.html", "oikiri.html").replace("shutuba.html", "oikiri.html")
                    res_oikiri = requests.get(oikiri_url, headers=headers, timeout=10)
                    soup_oikiri = BeautifulSoup(res_oikiri.content, 'html.parser', from_encoding='euc-jp')
                    
                    for tag in soup_oikiri(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    oikiri_table = soup_oikiri.find('table', class_='Oikiri_Table') or soup_oikiri.find('div', class_='OikiriData')
                    oikiri_text = oikiri_table.get_text(separator=' ', strip=True) if oikiri_table else soup_oikiri.get_text(separator=' ', strip=True)
                    st.success("競馬新聞高密度データ ＆ 調教タイムデータを正常取得！")
                except Exception as e:
                    oikiri_text = "※調教データの取得スキップ（データなし）"

            combined_race_text = f"【レース名】: {race_name}\n\n【競馬新聞・過去走・同距離成績・上がり3Fデータ】:\n{shutuba_text}\n\n【調教・追い切りタイム・内容データ】:\n{oikiri_text}"

            with st.spinner('3/3 Gemini 3.5 Flash で同距離適性＆上がり3F・調子を分析中...'):
                res = get_gemini_prediction(combined_race_text)
                
                if "error" in res:
                    st.error(f"分析エラー: {res['error']}")
                else:
                    st.subheader(f"📊 【{res.get('race_name', race_name)}】 分析結果")
                    
                    vol_level = res.get("volatility_level", "判定不能")
                    vol_reason = res.get("volatility_reason", "")
                    
                    st.info(f"⚡ **レース波乱度:** {vol_level}\n\n**【混戦・波乱理由】:** {vol_reason}")
                    
                    st.markdown("### 📈 主要馬の上がり3Fタイム推移（調子バイタル）")
                    preds = res.get("predictions", [])
                    
                    chart_data = {}
                    for item in preds:
                        horse_label = f"{item.get('mark', '')} {item.get('horse_number', '')}.{item.get('horse_name', '')}"
                        history = item.get("recent_3f_history", [])
                        if history:
                            chart_data[horse_label] = history
                    
                    if chart_data:
                        max_len = max(len(v) for v in chart_data.values())
                        formatted_chart_data = {}
                        for k, v in chart_data.items():
                            padded = [np.nan] * (max_len - len(v)) + v
                            formatted_chart_data[k] = padded
                        
                        df_chart = pd.DataFrame(formatted_chart_data)
                        df_chart.index = [f"{i}年前/前走" if i==0 else f"{i}走前" for i in range(max_len-1, -1, -1)]
                        st.line_chart(df_chart)
                        st.caption("※折れ線グラフが下に行くほど上がり3Fが速く（末脚性能が高い）、右肩下がりなら調子上昇傾向を示します。")

                    st.markdown("### 🏇 出走馬・印別予測勝率 ＆ 同距離適性評価")
                    if preds:
                        table_preds = []
                        for item in preds:
                            mark = item.get("mark", "-")
                            num = item.get("horse_number", "-")
                            name = item.get("horse_name", "-")
                            rate = item.get("predicted_win_rate", 0)
                            odds = item.get("current_odds", 1.0)
                            dist_eval = item.get("same_distance_eval", "-")
                            trend = item.get("condition_trend", "-")
                            
                            kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                            
                            table_preds.append({
                                "印": mark,
                                "馬番": num,
                                "馬名": name,
                                "AI予測勝率": f"{rate*100:.1f}%",
                                "単勝オッズ": f"{odds}倍",
                                "同距離適性": dist_eval,
                                "調子トレンド": trend,
                                "期待値(EV)": kelly["ev"],
                                "単勝判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                "同距離成績・調教・過去走解析": item.get("reason", "")
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
