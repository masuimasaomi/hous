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
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

if page == "🛠️ レース分析＆AI予測":
    st.title("🎯 AI全自動分析（出馬表＋過去3走成績＋調教データ統合解析）")
    st.write("出馬表URLを入力すると、Pythonが自動で「過去3走データ」と「調教データ」を読み込んで分析します。")
    
    target_url = st.text_input("出馬表URLを入力", value="https://race.netkeiba.com/race/shutuba.html?race_id=202605040301&rf=race_list")
    
    if st.button("過去走＋調教＋オッズを総合AI分析"):
        if not target_url:
            st.warning("URLを入力してください。")
        elif not GOOGLE_API_KEY:
            st.error("APIキーが設定されていません。Streamlit CloudのSecretsを確認してください。")
        else:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            with st.spinner('1/3 過去3走データを含む出馬表を取得中...'):
                try:
                    res_shutuba = requests.get(target_url, headers=headers, timeout=10)
                    soup_shutuba = BeautifulSoup(res_shutuba.content, 'html.parser', from_encoding='euc-jp')
                    
                    for tag in soup_shutuba(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    shutuba_table = soup_shutuba.find('table', class_='Shutuba_Table')
                    race_title = soup_shutuba.find('div', class_='RaceName') or soup_shutuba.find('h1', class_='RaceName')
                    race_name = race_title.get_text(strip=True) if race_title else "対象レース"
                    
                    shutuba_text = shutuba_table.get_text(separator=' ', strip=True) if shutuba_table else soup_shutuba.get_text(separator=' ', strip=True)
                except Exception as e:
                    st.error(f"出馬表の取得に失敗しました: {e}")
                    st.stop()

            with st.spinner('2/3 追い切り・調教データを自動取得中...'):
                try:
                    oikiri_url = target_url.replace("shutuba.html", "oikiri.html")
                    res_oikiri = requests.get(oikiri_url, headers=headers, timeout=10)
                    soup_oikiri = BeautifulSoup(res_oikiri.content, 'html.parser', from_encoding='euc-jp')
                    
                    for tag in soup_oikiri(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    oikiri_table = soup_oikiri.find('table', class_='Oikiri_Table') or soup_oikiri.find('div', class_='OikiriData')
                    oikiri_text = oikiri_table.get_text(separator=' ', strip=True) if oikiri_table else soup_oikiri.get_text(separator=' ', strip=True)
                    st.success("過去3走データ ＆ 調教データを正常取得！")
                except Exception as e:
                    oikiri_text = "※調教データの取得スキップ（データなし）"

            combined_race_text = f"【レース名】: {race_name}\n\n【出馬表＆過去3走データ】:\n{shutuba_text}\n\n【調教・追い切りデータ】:\n{oikiri_text}"

            with st.spinner('3/3 Geminiで（過去3走×調教×オッズ）を分析中...'):
                res = get_gemini_prediction(combined_race_text)
                
                if "error" in res:
                    st.error(f"分析エラー: {res['error']}")
                else:
                    st.subheader(f"📊 【{res.get('race_name', race_name)}】 分析結果")
                    
                    st.markdown("### 🏇 出走馬・印別予測勝率 (過去3走＆調教反映)")
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
                                "評価・過去走＆調教コメント": item.get("reason", "")
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
