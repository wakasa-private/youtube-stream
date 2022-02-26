import streamlit as st
import time
# import numpy as np
# import pandas as pd
# from PIL import Image


st.title('streamlit 超入門')
st.write('data frame')

st.sidebar.write("## これがあなたのプロフィールだ！")
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

# latest_iteration = st.empty()
# bar = st.progress(0)
# for i in range(100):
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i+1)
#     time.sleep(0.01)
# '''Done !'''


# if st.checkbox('show image'):
#     # 画像の表示
#     img = Image.open('Image/MILAN.jpg')
#     st.image(img, caption='MILAN', use_column_width=True)




# # 新宿付近のプロット
# df = pd.DataFrame(
#     np.random.rand(100, 2)/[50, 50] + [35.69, 139.70],
#     columns=['lat', 'lon']
# )

# st.dataframe(df.style.highlight_max(axis=1), width=400, height=400)
# st.map(df)


# """
# # area chart
# """
# st.area_chart(df)
# """
# # bar chart
# """
# st.bar_chart(df)
# st.latex(r'''
#      a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#      \sum_{k=0}^{n-1} ar^k =
#      a \left(\frac{1-r^{n}}{1-r}\right)
#      ''')

# """
# # 章
# ## 説
# ### こう

# ```python
# import numpy as np
# import pandas as pd
# import streamlit as st
# ```

# """