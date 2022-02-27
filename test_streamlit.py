import weakref
import streamlit as st
import time
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import func_ikawidget2 as spla


filepath = "ikaWidgetJSON_20220219190342/"
df_games = spla.read_spla_jsons(filepath)
df_games = spla.result_means(df_games)

'# Splatoon 2 Result Analyze'
'## game result'
target_columns = st.multiselect("select view data species",
    list(df_games.columns.values),
    ['stage', 'mode', 'rule', 'result', 'my kill', 'my death', 'my paint_point', 'my main'])
st.dataframe(df_games.loc[:, target_columns])

# サイドバーの設定
st.sidebar.write("## データ設定")
my_weapon = st.sidebar.selectbox("使用武器: ", df_games["my main"].unique())
select_mode = st.sidebar.selectbox("モード選択: ", df_games["mode"].unique())
select_rule = st.sidebar.selectbox("ルール選択: ", df_games["rule"].unique())
select_stage = st.sidebar.selectbox("ステージ: ", df_games["stage"].unique())

#勝率計算
'\n# 指定したルールでの勝率確認'
st.sidebar.write("### 指定したルールでの勝率確認")
if_weapon = st.sidebar.checkbox("武器を考慮する?")
if_mode = st.sidebar.checkbox("モードを考慮する?", value=True)
if_rule = st.sidebar.checkbox("ルールを考慮する?")
if_stage = st.sidebar.checkbox("ステージを考慮する?")


condition_rule = df_games['rule']==select_rule if if_rule else pd.Series([True for x in range(len(df_games))])
condition_weapon = df_games['my main']==my_weapon if if_weapon else pd.Series([True for x in range(len(df_games))])
condition_stage = df_games['stage']==select_stage if if_stage else pd.Series([True for x in range(len(df_games))])
condition_mode = df_games['mode']==select_mode if if_mode else pd.Series([True for x in range(len(df_games))])
condition_1 = ((condition_rule) & (condition_stage)) & ((condition_weapon) & (condition_mode))
target_df_1 = df_games[condition_1]
# st.dataframe(target_df_1.loc[:, target_columns])
st.write('#### 試合数: ', len(target_df_1), "games")
st.write('#### 勝率: ', round(target_df_1['result'].mean() * 100, 2), "%")
st.write('#### k/d mean: ', round(target_df_1['my k/d'].mean(), 2))

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
# ax.plot(target_df_1['my kill'] ,target_df_1['my death'], 'b.')
sns.distplot(target_df_1['my k/d'], bins=30)
sns.distplot(target_df_1['team k/d'], bins=30)
sns.distplot(target_df_1['enemy k/d'], bins=30)
ax.legend(['my k/d', 'team k/d mean', 'enemy k/d mean'])
ax.set_xlabel('kill & death num')
ax.set_xlim([0, 6])
# ax.set_ylabel('death num')
ax.grid()
st.pyplot(fig)

# view_target_fig2 = st.multiselect("rader view", 
#                 list(target_df_1.columns.unique()),
#                 default=["my kill", "team kills", "enemy kills", "my death", "team deaths", "enemy deaths"]
# )
fig2 = spla.view_result_mean(target_df_1)
st.pyplot(fig2)

'\n# Streamlit 超入門'

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


