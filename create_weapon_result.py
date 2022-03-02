import numpy as np
import pandas as pd
import csv

# 自作関数
import func_ikawidget2 as spla

filepath = "ikaWidgetJSON_20220219190342/"
df_games = spla.read_spla_jsons(filepath)
df_games = spla.result_means(df_games)

# マッチングした武器のリザルト
df_weapon = spla.get_weapon_result(df_games)
weapon_list = df_weapon['weapon'].unique()
print(weapon_list)

csv_name = "weapon_information_2.csv"
with open(csv_name, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(weapon_list)
    # for item in weapon_list:
    #     writer.writerow(item)
f.close()