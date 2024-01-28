from openai import OpenAI
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
import plotly.graph_objects as go
import json

# OpenAI clientインスタンス
client = OpenAI(api_key = st.secrets["api_key"])

# =================
#      Model
# =================

# 音声　→　文字　【 Whisper 】
def audio_to_text(audio_bytes):
    with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()
        try:
            with open(temp_file.name, "rb") as audio_file:
                response = client.audio.transcriptions.create(
                                model="whisper-1", 
                                file=audio_file,
                                prompt="",
                                language="ja",
                                response_format="text",
                            )
        except Exception:
            return False
        # 無音のときは、Falseを返す
        for ng_word in ["視聴", "字幕", "by H", "見てくれて"]:
            if ng_word in response:
                return False
    return response

# 文字　→　データ　【 ChatGPT 】
def text_to_data(transcript):
    sys_prompt = '''出力例に従いJSONで出力して下さい。関係ないことは無視して下さい。
    # 出力例: {"shapes": [{"type": "rectangle", "width": *, "height": *}, {"type": "circle", "radius": *}]}
    # 出力がない場合: {"shapes": []}
    # 正方形は四角形で処理すること
    # 直径は半径に変換すること
    '''
    response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": transcript}
                    ],
                    temperature=0,
                )
    order_json = response.choices[0].message.content
    order_list = json.loads(order_json)['shapes']
    return order_json, order_list 

# データ　→　出力
def data_to_output(order_lst):
    x_lst, y_lst = [0], [0] # 表示領域特定のための座標（x, y）リスト
    fig = go.Figure()
    for order in order_lst:
        # 四角形
        if order['type'] == 'rectangle':
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=0, y0=0, x1=order["width"], y1=order["height"],
                          opacity=0.2, fillcolor="orange", line_color="red",
                          )
            x_lst.append(order["width"])
            y_lst.append(order["height"])
        # 円
        elif order['type'] == 'circle':
            fig.add_shape(type="circle", xref="x", yref="y",
                          x0=0, y0=0, x1=order["radius"] * 2, y1=order["radius"] * 2,
                          opacity=0.2, fillcolor="LightSeaGreen", line_color="blue",
                          )
            x_lst.append(order["radius"] * 2)
            y_lst.append(order["radius"] * 2)
    # レイアウト・表示領域の設定
    fig.update_layout(
        plot_bgcolor='lightgray',
        width = 500,
        height = 500,
        showlegend = False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.update_xaxes(
        range=[-max(x_lst) * 0.1, max(x_lst) * 1.1],
        showgrid = True,
    )
    fig.update_yaxes(
        range=[-max(y_lst) * 0.1, max(y_lst) * 1.1],
        scaleanchor = "x",
        scaleratio = 1,
    )
    return fig

# utility
def sub_title(main_lst, sub_lst):
    sub_title = f"**{main_lst[0]}**"
    for main, sub in zip(main_lst[1:], sub_lst):
        sub_title += f"&ensp;:arrow_right:&nbsp;{sub}&nbsp;:arrow_right:&ensp;**{main}**"
    return sub_title


# =========================
#      View + Control
# =========================

# ページタイトル
st.set_page_config(
    page_title = "音声認識デモ",
)

# タイトル
st.title("音声認識デモ")
st.success(sub_title(["音声", "文字", "データ", "出力"], ["Whisper", "ChatGPT", "PC"]))

# 音声（入力）
order_audio = audio_recorder(text="録音　開始／終了", pause_threshold=10, neutral_color="#6aa36f")

if order_audio:
    # 音声　→　文字　【 Whisper 】
    st.info(sub_title(["音声", "文字"], ["Whisper"]))
    order_text = audio_to_text(order_audio)
    if order_text:
        st.write(order_text)

        # 文字　→　データ　【 ChatGPT 】
        st.info(sub_title(["文字", "データ"], ["ChatGPT"]))
        order_json, order_list = text_to_data(order_text)
        st.json(order_json)

        # データ　→　出力
        st.info(sub_title(["データ", "出力"], ["PC"]))
        order_figure = data_to_output(order_list)
        st.plotly_chart(order_figure, use_container_width=False)
    else:
        st.error("もう一度、録音して下さい。")
