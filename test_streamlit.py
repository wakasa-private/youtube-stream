import streamlit as st
import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import func_ikawidget2 as spla


filepath = "ikaWidgetJSON_20220219190342/"
df_games = spla.read_spla_jsons(filepath)
df_games = spla.result_means(df_games)

'# Splatoon 2 Result Analyze'
'## game result'
st.dataframe(df_games)
# st.dataframe(df_games)
# data add



st.title('streamlit 超入門')
st.write('data frame')

st.sidebar.write("## データ設定")
my_weapon = st.sidebar.selectbox("使用武器: ", df_games["my main"].unique())
select_rule = st.sidebar.selectbox("ルール選択: ", df_games["rule"].unique())
select_stage = st.sidebar.selectbox("ステージ: ", df_games["stage"].unique())
text = st.text_input("あなたの趣味を教えてください")

left_column, right_column = st.beta_columns(2)
button = left_column.button("右カラムに文字")
expander = st.beta_expander('問い合わせ')
expander.write("いえーーー")
if button:
    right_column.write("ここは左カラムです")

option = st.selectbox(
    'あなたが好きな数字を教えてください',
    list(range(1,11))
    )
condition = st.slider('あなたの今の調子は?', 0, 100, 5)

'あなたの好きな数字は', option, 'です'
'あなたの趣味は', text, 'です'
'あんたの今の調子は', condition, 'でっせ'


