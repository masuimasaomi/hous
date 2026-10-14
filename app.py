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
st.set_page_config(page_title="名古屋競馬特化 AI ROIオプティマイザ", page_icon="🏇", layout="wide")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# 名古屋競馬場特化のフィードバックルールの初期化
if "feedback_rules" not in st.session_state:
    st.session_state["feedback_rules"] = [
        "【名古屋・新コース形態】2022年移転後の新コースは直線が300mと西日本最長クラスのため、旧コースのような前残り一辺倒ではなく、過去3走平均の上がり3F上位の差し・追込馬の勝率・期待値を高く評価すること。",
        "【名古屋・1500m戦の展開】主力距離である1500m戦は最初のコーナーまで距離がありペースが速くなりやすいため、逃げ・先行馬が競り合う場合は中団待機の馬を大幅に加点すること。",
        "【名古屋・920m短距離】920mの超短距離戦では圧倒的にテンのスピード（スタートからのダッシュ力・前走の通過順1〜2番手）を最優先で評価すること。",
        "【馬場状態・時計勝負】重・不良馬場では極端に時計が速くなる傾向があるため、良馬場の上がり性能よりも「持ち時計（走破タイム実績）」の優秀な馬を強力に加点すること。"
    ]

# ==========================================
# 1. AI予測 & 名古屋特化統合ロジック
# ==========================================
def deduplicate_and_merge_rules(existing_rules, new_rule):
    if not existing_rules:
        return [new_rule], "新規追加"

    system_prompt = """
    あなたは地方・名古屋競馬AIのナレッジベース管理モジュールです。
    【既存の名古屋ルール一覧】と【追加したい新しいルール】を分析し、同種のものは統合し、新しいものは末尾に追加してください。

    JSONフォーマットのみを出力してください:
    {
      "status": "MERGED" または "ADDED",
      "merged_rules": ["ルール1", "ルール2", ...],
      "reason": "説明"
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
        if new_rule not in existing_rules:
            return existing_rules + [new_rule], "新規追加（フォールバック）"
        return existing_rules, "同種ルールが既に存在します"

def get_gemini_prediction(combined_race_text, feedback_rules=None):
    rules_text = ""
    if feedback_rules:
        rules_text = "\n".join([f"- {rule}" for rule in feedback_rules])
    else:
        rules_text = "- 特になし（デフォルト名古屋ロジック適用）"

    system_prompt = f"""
    あなたは地方競馬・名古屋競馬場の確率論・波乱度解析・展開読み・調教時計解析に精通したプロのデータサイエンティストです。
    提供された【名古屋競馬の出馬表・新聞データ】および【調教データ】を精査し、各馬の実質勝率（1着確率）、レース展開、荒れ度、印、馬連推奨、三連複推奨を分析してください。

    【名古屋競馬特化の最重要解析ポイント】
    1. 新コース形態（直線300m）に応じた末脚（差し・追込）の再評価
    2. 距離別の展開有利不利（920mの先行力 vs 1500m/1700mの差し有利など）
    3. 上がり3F性能（過去3走平均＆前走上がり3F順位）
    4. トラックバイアスと持ち時計の比較

    【学習済みの名古屋フィードバックルール（最優先適用）】
    {rules_text}

    以下のJSONフォーマットのみを出力してください。
    {{
      "race_name": "レース名",
      "volatility_level": "🔥🔥🔥 超高波乱（大荒れ注意）",
      "volatility_reason": "単勝オッズは割れているが上位人気の上がり3F差が小さく、波乱要素大。",
      "pace_prediction": "🔥 ハイペース想定（差し有利）",
      "pace_reason": "同型先行馬が競合。直線300mの新名古屋コースで末脚炸裂の展開利。",
      "predictions": [
        {{
          "mark": "◎",
          "horse_number": 1,
          "horse_name": "馬名",
          "predicted_win_rate": 0.28,
          "current_odds": 3.2,
          "same_distance_eval": "🏆 名古屋同距離最高実績",
          "condition_trend": "🔥 急上昇（絶好調）",
          "recent_3f_history": [37.1, 36.8, 36.5],
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
      ],
      "recommended_sanrenpuku": [
        {{
          "combination": "1 - 2 - 5",
          "predicted_rate": 0.08,
          "current_odds": 28.5,
          "reason": "名古屋の長い直線を活かす差し脚信頼の三連複ペア"
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
    system_prompt = """
    あなたは地方名古屋競馬予想AIの自己改善プログラムです。
    提供された【名古屋競馬の実際のレース結果】を分析し、1〜3着馬の共通点（上がり3F、通過順、トラックバイアスなど）から、今後の名古屋予想ロジックに組み込むべき追加ルール文を1〜2文で提案してください。

    以下のJSONフォーマットのみを出力してください。
    {
      "race_summary": "1〜3着の実際の着順とタイムまとめ",
      "winning_factors": "勝因・好走要因（名古屋新コース形態、上がり3F、展開など）",
      "ai_reflection": "AI予想ロジックの反省点と今後の改善・修正ポイント",
      "suggested_rule": "名古屋競馬用に学習させる具体的な追加ルール"
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
st.sidebar.title("🏇 名古屋競馬特化 AI ROI")
page = st.sidebar.radio("メニュー", ["🛠️ 名古屋レース分析＆AI予測", "🏁 結果検証＆名古屋プロンプト学習", "📈 バックテスト分析"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 資金管理設定")
initial_bankroll = st.sidebar.number_input("現在資金 (円)", min_value=10000, value=100000, step=10000)
kelly_fraction = st.sidebar.slider("ケリー係数 (安全率)", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 適用中の【名古屋専用】ルール")
if st.session_state["feedback_rules"]:
    for i, r in enumerate(st.session_state["feedback_rules"]):
        st.sidebar.caption(f"{i+1}. {r}")
else:
    st.sidebar.caption("適用中のカスタムルールはありません")

if st.sidebar.button("全ルールを初期化"):
    st.session_state["feedback_rules"] = []
    st.rerun()

if page == "🛠️ 名古屋レース分析＆AI予測":
    st.title("🎯 名古屋競馬 AI全自動分析（馬連・三連複対応）")
    st.write("netkeibaの地方競馬版（nar.netkeiba.com）の出馬表・新聞URLを入力してください。")
    
    # 今日の名古屋競馬（例: 2026年10月14日）のデフォルトURLをセット
    default_url = "https://nar.netkeiba.com/race/newspaper.html?race_id=202642101411"
    target_url = st.text_input("名古屋競馬の出馬表 / 競馬新聞URLを入力", value=default_url)
    
    if st.button("名古屋特化AI分析を実行"):
        if target_url and GOOGLE_API_KEY:
            headers = {'User-Agent': 'Mozilla/5.0'}
            with st.spinner('名古屋競馬データを取得＆AI解析中...'):
                try:
                    res_shutuba = requests.get(target_url, headers=headers, timeout=10)
                    soup_shutuba = BeautifulSoup(res_shutuba.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_shutuba(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    race_title = soup_shutuba.find('div', class_='RaceName') or soup_shutuba.find('h1', class_='RaceName')
                    race_name = race_title.get_text(strip=True) if race_title else "名古屋対象レース"
                    newspaper_table = soup_shutuba.find('div', id='RaceNewspaper') or soup_shutuba.find('table', class_='Shutuba_Table') or soup_shutuba.find('table')
                    shutuba_text = newspaper_table.get_text(separator=' ', strip=True) if newspaper_table else soup_shutuba.get_text(separator=' ', strip=True)

                    oikiri_url = target_url.replace("newspaper.html", "oikiri.html").replace("shutuba.html", "oikiri.html")
                    res_oikiri = requests.get(oikiri_url, headers=headers, timeout=10)
                    soup_oikiri = BeautifulSoup(res_oikiri.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_oikiri(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    oikiri_table = soup_oikiri.find('table', class_='Oikiri_Table') or soup_oikiri.find('div', class_='OikiriData')
                    oikiri_text = oikiri_table.get_text(separator=' ', strip=True) if oikiri_table else soup_oikiri.get_text(separator=' ', strip=True)

                    combined_race_text = f"【レース名】: {race_name}\n\n【名古屋出馬表・新聞データ】:\n{shutuba_text}\n\n【調教データ】:\n{oikiri_text}"
                    
                    res = get_gemini_prediction(combined_race_text, feedback_rules=st.session_state["feedback_rules"])
                    
                    if "error" not in res:
                        st.subheader(f"📊 【{res.get('race_name', race_name)}】 名古屋AI分析結果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"⚡ **レース波乱度:** {res.get('volatility_level', '')}\n\n{res.get('volatility_reason', '')}")
                        with col2:
                            st.success(f"🏁 **名古屋予想展開:** {res.get('pace_prediction', '')}\n\n{res.get('pace_reason', '')}")
                        
                        st.markdown("### 🏇 出走馬・印別予測勝率 (名古屋新コース形態反映)")
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
                                    "名古屋適性": item.get("same_distance_eval", "-"),
                                    "調子トレンド": item.get("condition_trend", "-"),
                                    "期待値(EV)": kelly["ev"],
                                    "単勝判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                    "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                    "評価理由": item.get("reason", "")
                                })
                            st.dataframe(pd.DataFrame(table_preds), use_container_width=True)

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
                            st.dataframe(pd.DataFrame(table_umaren), use_container_width=True)

                        st.markdown("### 🥉 おすすめ三連複組み合わせ（期待値順）")
                        sanrenpuku_list = res.get("recommended_sanrenpuku", [])
                        if sanrenpuku_list:
                            table_sanrenpuku = []
                            for item in sanrenpuku_list:
                                combo = item.get("combination")
                                rate = item.get("predicted_rate", 0)
                                odds = item.get("current_odds", 1.0)
                                kelly = calculate_kelly_bet(rate, odds, initial_bankroll, kelly_fraction)
                                table_sanrenpuku.append({
                                    "三連複組合せ": combo,
                                    "的中確率": f"{rate*100:.1f}%",
                                    "想定オッズ": f"{odds}倍",
                                    "期待値(EV)": kelly["ev"],
                                    "三連複判定": "🔥 買い" if kelly["ev"] > 1.0 else "⏸️ 見送り",
                                    "推奨購入額": f"¥{kelly['bet_amount']:,}",
                                    "理由": item.get("reason", "")
                                })
                            st.dataframe(pd.DataFrame(table_sanrenpuku), use_container_width=True)
                except Exception as e:
                    st.error(f"エラー: {e}")

elif page == "🏁 結果検証＆名古屋プロンプト学習":
    st.title("🏁 名古屋レース結果検証 ＆ AI学習")
    st.write("名古屋競馬のレース結果（`result.html`）をスクレイピングして、名古屋特有の波乱要因・トラックバイアスを学習させます。")
    
    result_url = st.text_input("名古屋レース結果URLを入力", value="https://nar.netkeiba.com/race/result.html?race_id=202642101411")
    
    if st.button("名古屋レース結果を検証・反省する"):
        if result_url and GOOGLE_API_KEY:
            headers = {'User-Agent': 'Mozilla/5.0'}
            with st.spinner('名古屋レース結果を取得中...'):
                try:
                    res_result = requests.get(result_url, headers=headers, timeout=10)
                    soup_result = BeautifulSoup(res_result.content, 'html.parser', from_encoding='euc-jp')
                    for tag in soup_result(["script", "style", "noscript", "iframe"]):
                        tag.extract()
                    
                    result_table = soup_result.find('table', id='All_Result_Table') or soup_result.find('table')
                    result_text = result_table.get_text(separator=' ', strip=True) if result_table else soup_result.get_text(separator=' ', strip=True)
                    
                    analysis = analyze_race_result(result_text)
                    
                    if "error" not in analysis:
                        st.subheader("📝 名古屋AIのレース振り返り＆反省")
                        st.write(f"**【1〜3着結果】:** {analysis.get('race_summary', '')}")
                        st.success(f"💡 **【名古屋での勝因・好走要因】:**\n{analysis.get('winning_factors', '')}")
                        st.warning(f"🔧 **【AIの反省点】:**\n{analysis.get('ai_reflection', '')}")
                        
                        sug_rule = analysis.get('suggested_rule', '')
                        st.session_state["current_sug_rule"] = sug_rule
                        st.markdown(f"### 🧠 AIが提案する新・名古屋ルール:\n> **{sug_rule}**")
                    else:
                        st.error(f"分析エラー: {analysis['error']}")
                except Exception as e:
                    st.error(f"取得エラー: {e}")

    if "current_sug_rule" in st.session_state and st.session_state["current_sug_rule"]:
        if st.button("➕ 重複チェックを行って名古屋ルールに統合・保存"):
            with st.spinner('名古屋既存ルールとの重複統合処理中...'):
                new_rules, reason = deduplicate_and_merge_rules(
                    st.session_state["feedback_rules"], 
                    st.session_state["current_sug_rule"]
                )
                st.session_state["feedback_rules"] = new_rules
                st.success(f"処理完了: {reason}")
                st.session_state["current_sug_rule"] = None
                st.rerun()

elif page == "📈 バックテスト分析":
    st.title("📈 名古屋競馬 バックテスト結果")
    st.write("名古屋競馬の過去シミュレーションエリアです。")
