import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import requests  # ←これが抜けていたため追加
from bs4 import BeautifulSoup  # ←こちらもセットで追加
import google.generativeai as genai

# ==========================================
# 0. 初期設定とAPIクライアント
# ==========================================
st.set_page_config(page_title="競馬AI ROIオプティマイザ", page_icon="🏇", layout="wide")

# Google AI Studioで取得したAPIキーを設定
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "ここにAI StudioのAPIキーを入力")
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. コア・アルゴリズム (AI予測 & 資金管理)
# ==========================================
def get_gemini_prediction(race_data_text):
    """Gemini-3.5-flashを使ってレースデータのテキストから勝率を予測する"""
    system_prompt = """
    あなたは競馬の確率論に精通したプロのデータサイエンティストです。
    提供されたデータのみから「その馬が1着になる実質勝率」を算出し、以下のJSONフォーマットのみを出力してください。
    {
      "horse_number": 1,
      "predicted_win_rate": 0.15,
      "confidence_score": 8,
      "reasoning": "根拠"
    }
    """
    try:
        # モデルの指定 (gemini-3.5-flash) とシステムプロンプトの設定
        model = genai.GenerativeModel(
            'gemini-3.5-flash',
            system_instruction=system_prompt
        )
        # JSON出力を強制するコンフィグ
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        )
        
        response = model.generate_content(
            race_data_text,
            generation_config=generation_config
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def calculate_kelly_bet(predicted_win_rate, odds, bankroll, kelly_fraction=0.25):
    """ケリー基準を用いて最適なベット額を計算する"""
    expected_value = predicted_win_rate * odds
    
    if expected_value <= 1.0:
        return {"action": "見送り", "bet_amount": 0, "percentage": 0.0, "ev": expected_value}
        
    b = odds - 1.0
    p = predicted_win_rate
    q = 1.0 - p
    
    f = (b * p - q) / b
    adjusted_fraction = f * kelly_fraction
    
    bet_amount = int((bankroll * adjusted_fraction) // 100 * 100)
    
    return {
        "action": "買い" if bet_amount > 0 else "見送り",
        "bet_amount": bet_amount,
        "percentage": round(adjusted_fraction * 100, 2),
        "ev": round(expected_value, 2)
    }

# ==========================================
# 2. データ準備 (※UI表示用のダミーデータ)
# ==========================================
@st.cache_data
def load_mock_backtest_data(initial_bankroll, kelly_fraction):
    dates = pd.date_range(start="2025-01-01", periods=100, freq="W-SUN")
    bankroll = initial_bankroll
    history = []
    
    for d in dates:
        is_win = np.random.rand() < 0.22 
        bet_amount = int((bankroll * kelly_fraction * 0.1) // 100 * 100)
        bankroll -= bet_amount
        return_amount = int(bet_amount * np.random.uniform(5.0, 12.0)) if is_win else 0
        bankroll += return_amount
            
        history.append({"日付": d, "購入額": bet_amount, "払戻額": return_amount, "資金残高": bankroll})
        
    df = pd.DataFrame(history)
    summary = {
        "初期資金": initial_bankroll,
        "最終資金": int(bankroll),
        "回収率 (ROI)": f"{((df['払戻額'].sum() / df['購入額'].sum()) * 100):.1f}%" if df['購入額'].sum() > 0 else "0%",
        "最大ドローダウン": "-28.4%",
    }
    return summary, df

def load_weekend_predictions(current_bankroll, kelly_fraction):
    raw_data = [
        {"レース": "東京11R", "馬番": 7, "馬名": "ジェミニフラッシュ", "AI勝率": 0.185, "オッズ": 8.5},
        {"レース": "東京12R", "馬番": 3, "馬名": "データストーム", "AI勝率": 0.100, "オッズ": 5.2},
        {"レース": "京都11R", "馬番": 12, "馬名": "ケリーインパクト", "AI勝率": 0.080, "オッズ": 15.0},
    ]
    
    results = []
    for d in raw_data:
        kelly = calculate_kelly_bet(d["AI勝率"], d["オッズ"], current_bankroll, kelly_fraction)
        results.append({
            "レース": d["レース"],
            "馬番": d["馬番"],
            "馬名": d["馬名"],
            "AI予測勝率": f"{d['AI勝率']*100:.1f}%",
            "オッズ": d["オッズ"],
            "期待値 (EV)": kelly["ev"],
            "指示": kelly["action"],
            "推奨投資割合": f"{kelly['percentage']}%",
            "推奨購入額": f"¥{kelly['bet_amount']:,}"
        })
    return pd.DataFrame(results)

# ==========================================
# 3. UIレイアウト
# ==========================================
st.sidebar.title("🏇 AI競馬 ROIシステム")
page = st.sidebar.radio("メニュー", ["📈 バックテスト分析", "🔮 今週末の予測・投票", "🛠️ APIテスト"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("初期資金 / 現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

if page == "📈 バックテスト分析":
    st.title("📈 バックテスト結果")
    summary, history_df = load_mock_backtest_data(initial_bankroll, kelly_fraction)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("初期資金", f"¥{summary['初期資金']:,}")
    col2.metric("最終資金", f"¥{summary['最終資金']:,}", f"{summary['最終資金'] - summary['初期資金']:,} 円")
    col3.metric("回収率 (ROI)", summary['回収率 (ROI)'])
    col4.metric("最大ドローダウン", summary['最大ドローダウン'], delta_color="inverse")
    
    st.line_chart(history_df.set_index("日付")["資金残高"])
    st.dataframe(history_df, use_container_width=True)

elif page == "🔮 今週末の予測・投票":
    st.title("🔮 今週末の最適ベット額")
    
    df = load_weekend_predictions(initial_bankroll, kelly_fraction)
    
    def highlight_action(row):
        return ['background-color: #d4edda; color: #155724'] * len(row) if row['指示'] == '買い' else ['background-color: #f8d7da; color: #721c24'] * len(row)

    st.dataframe(df.style.apply(highlight_action, axis=1), use_container_width=True)

elif page == "🛠️ APIテスト":
    st.title("🛠️ 全自動スクレイピング＆予測テスト")
    st.write("netkeibaなどの出馬表URLを入力すると、Pythonが自動でWebページを取得し、AIが解析します。")
    
    target_url = st.text_input("出馬表URLを入力してください", value="https://race.netkeiba.com/race/shutuba.html?race_id=202605040301&rf=race_list")
    
    if st.button("データ取得＆Gemini分析を実行"):
        if not target_url:
            st.warning("URLを入力してください。")
        else:
            with st.spinner('1/2 ウェブサイトから馬柱データを取得中...'):
                try:
                    # Python側でWebページを取得
                    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    response = requests.get(target_url, headers=headers, timeout=10)
                    response.encoding = 'euc-jp' # netkeibaの文字コード
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 不要なタグ（JavascriptやCSS）を排除
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # 抽出したテキストを取得
                    race_text = soup.get_text(separator=' ', strip=True)
                    
                    st.success("データ取得成功！Geminiで勝率を分析中...")
                    
                    # テキスト化したデータをGeminiに渡す
                    with st.spinner('2/2 Geminiで勝率と期待値を算出中...'):
                        result = get_gemini_prediction(race_text)
                        st.json(result)
                        
                except Exception as e:
                    st.error(f"データ取得エラー: {e}")
