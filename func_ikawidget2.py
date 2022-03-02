import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import streamlit as st
import json
import os 


@st.cache(allow_output_mutation=True)
def read_spla_jsons(filepath):
    files = os.listdir(filepath)
    game_result = []
    own_player_result = []
    my_player_result = []
    ene_player_result = []

    for file in files:
        with open(filepath+file,"r",encoding="utf-8") as f:
            result_json=json.load(f)

            # game result
            result = result_json['my_team_result']['key']
            stage = result_json['stage']['name']
            rule = result_json['rule']['name']
            date = result_json['start_time']
            tmp_mode = result_json["game_mode"]["key"]
            if tmp_mode.find("gachi") != -1:
                mode = "ガチマ"
            elif tmp_mode.find("league") != -1:
                mode = "リグマ"
            elif tmp_mode.find("private") != -1:
                mode = "プライベート"
            elif tmp_mode.find("fes") != -1:
                mode = "フェス"
            else:
                mode = "ナワバリ"
            
            if mode == "ガチマ" or mode == "リグマ":
                tmp_suffix = result_json["udemae"]["number"]
                if tmp_suffix == 0:
                    suffix = "-"
                elif tmp_suffix == 2:
                    suffix = "+"
                else:
                    suffix = ""
                    udemae = result_json["udemae"]["name"]
                    estimate_power = result_json["estimate_gachi_power"]
            else:
                udemae = "R"
                estimate_power = 0

            game_result.append([date, stage, mode, rule, result, udemae, estimate_power])

            my_result = result_json['my_team_members']
            ene_result = result_json['other_team_members']

            # Result of myself
            own_result = []
            kill_count = result_json['player_result']['kill_count']
            assist_count = result_json['player_result']['assist_count']
            death_count = result_json['player_result']['death_count']
            special_count = result_json['player_result']['special_count']
            paint_point = result_json['player_result']['game_paint_point']
            k_d = kill_count/death_count if death_count != 0 else kill_count

            main_weapon = result_json['player_result']['player']['weapon']['name']
            sub_weapon = result_json['player_result']['player']['weapon']['sub']['name']
            special_weapon = result_json['player_result']['player']['weapon']['special']['name']
            own_player_result.append([kill_count, assist_count, death_count, special_count, paint_point, k_d, main_weapon, sub_weapon, special_weapon])

            i_enemy = len(ene_result)
            i_my = len(my_result)
            # print('i_my', i_my, 'i_enemy', i_enemy)

            my_teams = []
            for i in range(i_my):
                paint_point = my_result[i]['game_paint_point']
                kill_count = my_result[i]['kill_count'] if paint_point != 0 else 0
                assist_count = my_result[i]['assist_count'] if paint_point != 0 else 0
                death_count = my_result[i]['death_count'] if paint_point != 0 else 0
                special_count = my_result[i]['special_count'] if paint_point != 0 else 0
                k_d = kill_count/death_count if death_count != 0 else kill_count

                main_weapon = my_result[i]['player']['weapon']['name'] if paint_point != 0 else 'バカ'
                sub_weapon = my_result[i]['player']['weapon']['sub']['name'] if paint_point != 0 else 'アホ'
                special_weapon = my_result[i]['player']['weapon']['special']['name'] if paint_point != 0 else '逃げ虫'
                my_teams.append([kill_count, assist_count, death_count, special_count, paint_point, k_d, main_weapon, sub_weapon, special_weapon])

            ene_teams = []
            for i in range(i_enemy):
                paint_point = ene_result[i]['game_paint_point']
                kill_count = ene_result[i]['kill_count'] if paint_point != 0 else 0
                assist_count = ene_result[i]['assist_count'] if paint_point != 0 else 0
                death_count = ene_result[i]['death_count'] if paint_point != 0 else 0
                special_count = ene_result[i]['special_count'] if paint_point != 0 else 0
                k_d = kill_count/death_count if death_count != 0 else kill_count

                main_weapon = ene_result[i]['player']['weapon']['name'] if paint_point != 0 else 'バカ'
                sub_weapon = ene_result[i]['player']['weapon']['sub']['name'] if paint_point != 0 else 'アホ'
                special_weapon = ene_result[i]['player']['weapon']['special']['name'] if paint_point != 0 else '逃げ虫'
                ene_teams.append([kill_count, assist_count, death_count, special_count, paint_point, k_d, main_weapon, sub_weapon, special_weapon])

            sky_data = [0, 0, 0, 0, 0, 0, 'バカ', 'アホ', '逃げ虫']
            for i in range(3-i_my):
                my_teams.append(sky_data)
            for i in range(4-i_enemy):
                ene_teams.append(sky_data)
            my_player_result.append(my_teams)
            ene_player_result.append(ene_teams)
            
    trans_my_result = list(zip(*my_player_result))
    trans_ene_result = list(zip(*ene_player_result))
    df_games_tmp = pd.DataFrame(game_result, columns=['time', 'stage', 'mode', 'rule', 'result', 'udemae', 'estimate_power'])
    df_own_result = pd.DataFrame(own_player_result, columns=['my kill', 'my assist', 'my death', 'my sp_count', 'my paint_point', 'my k/d', 'my main', 'my sub', 'my special'])
    df_my_1 = pd.DataFrame(trans_my_result[0], columns=['ally 1 kill', 'ally 1 assist', 'ally 1 death', 'ally 1 sp_count', 'ally 1 paint_point', 'ally 1 k/d', 'ally 1 main', 'ally 1 sub', 'ally 1 special'])
    df_my_2 = pd.DataFrame(trans_my_result[1], columns=['ally 2 kill', 'ally 2 assist', 'ally 2 death', 'ally 2 sp_count', 'ally 2 paint_point', 'ally 2 k/d', 'ally 2 main', 'ally 2 sub', 'ally 2 special'])
    df_my_3 = pd.DataFrame(trans_my_result[2], columns=['ally 3 kill', 'ally 3 assist', 'ally 3 death', 'ally 3 sp_count', 'ally 3 paint_point', 'ally 3 k/d', 'ally 3 main', 'ally 3 sub', 'ally 3 special'])
    df_ene_1 = pd.DataFrame(trans_ene_result[0], columns=['enemy 1 kill', 'enemy 1 assist', 'enemy 1 death', 'enemy 1 sp_count', 'enemy 1 paint_point', 'enemy 1 k/d', 'enemy 1 main', 'enemy 1 sub', 'enemy 1 special'])
    df_ene_2 = pd.DataFrame(trans_ene_result[1], columns=['enemy 2 kill', 'enemy 2 assist', 'enemy 2 death', 'enemy 2 sp_count', 'enemy 2 paint_point', 'enemy 2 k/d', 'enemy 2 main', 'enemy 2 sub', 'enemy 2 special'])
    df_ene_3 = pd.DataFrame(trans_ene_result[2], columns=['enemy 3 kill', 'enemy 3 assist', 'enemy 3 death', 'enemy 3 sp_count', 'enemy 3 paint_point', 'enemy 3 k/d', 'enemy 3 main', 'enemy 3 sub', 'enemy 3 special'])
    df_ene_4 = pd.DataFrame(trans_ene_result[3], columns=['enemy 4 kill', 'enemy 4 assist', 'enemy 4 death', 'enemy 4 sp_count', 'enemy 4 paint_point', 'enemy 4 k/d', 'enemy 4 main', 'enemy 4 sub', 'enemy 4 special'])
    
    df_games = pd.concat([df_games_tmp,
                  df_own_result, df_my_1, df_my_2, df_my_3,
                  df_ene_1, df_ene_2, df_ene_3, df_ene_4],
                  axis=1, sort=False)

    return df_games


@st.cache(allow_output_mutation=True)
def result_means(df_games):
    # modify rsult
    df_games['result'][df_games['result']=="victory"] = 1
    df_games['result'][df_games['result']=="defeat"] = 0

    # exist data for ally
    condition = [df_games['ally 1 paint_point']==0]
    df_games['ally exist 1'] = np.select(condition, [0], default=1)
    condition = [df_games['ally 2 paint_point']==0]
    df_games['ally exist 2'] = np.select(condition, [0], default=1)
    condition = [df_games['ally 3 paint_point']==0]
    df_games['ally exist 3'] = np.select(condition, [0], default=1)

    # exist data for enemy
    condition = [df_games['enemy 1 paint_point']==0]
    df_games['enemy exist 1'] = np.select(condition, [0], default=1)
    condition = [df_games['enemy 2 paint_point']==0]
    df_games['enemy exist 2'] = np.select(condition, [0], default=1)
    condition = [df_games['enemy 3 paint_point']==0]
    df_games['enemy exist 3'] = np.select(condition, [0], default=1)
    condition = [df_games['enemy 4 paint_point']==0]
    df_games['enemy exist 4'] = np.select(condition, [0], default=1)

    # data add
    df_games['result'][df_games['result']=="victory"] = 1
    df_games['result'][df_games['result']=="defeat"] = 0
    df_games['team members'] = df_games['ally exist 1'] + df_games['ally exist 2'] + df_games['ally exist 3'] + 1
    df_games['enemy members'] = df_games['enemy exist 1'] + df_games['enemy exist 2'] + df_games['enemy exist 3'] + df_games['enemy exist 4']

    # average result ally
    df_games['team kills'] = (df_games['ally 1 kill'] + df_games['ally 2 kill'] + df_games['ally 3 kill']) / (df_games['team members']-1)
    df_games['team assists'] = (df_games['ally 1 assist'] + df_games['ally 2 assist'] + df_games['ally 3 assist']) / (df_games['team members']-1)
    df_games['team deaths'] = (df_games['ally 1 death'] + df_games['ally 2 death'] + df_games['ally 3 death']) / (df_games['team members']-1)
    df_games['team sp_counts'] = (df_games['ally 1 sp_count'] + df_games['ally 2 sp_count'] + df_games['ally 3 sp_count']) / (df_games['team members']-1)
    df_games['team paint_points'] = (df_games['ally 1 paint_point'] + df_games['ally 2 paint_point'] + df_games['ally 3 paint_point']) / (df_games['team members']-1)

    # average result enemy
    df_games['enemy kills'] = (df_games['enemy 1 kill'] + df_games['enemy 2 kill'] + df_games['enemy 3 kill'] + df_games['enemy 4 kill']) / df_games['enemy members']
    df_games['enemy assists'] = (df_games['enemy 1 assist'] + df_games['enemy 2 assist'] + df_games['enemy 3 assist'] + df_games['enemy 4 assist']) / df_games['enemy members']
    df_games['enemy deaths'] = (df_games['enemy 1 death'] + df_games['enemy 2 death'] + df_games['enemy 3 death'] + df_games['enemy 4 death']) / df_games['enemy members']
    df_games['enemy sp_counts'] = (df_games['enemy 1 sp_count'] + df_games['enemy 2 sp_count'] + df_games['enemy 3 sp_count'] + df_games['enemy 4 sp_count']) / (df_games['team members']-1)
    df_games['enemy paint_points'] = (df_games['enemy 1 paint_point'] + df_games['enemy 2 paint_point'] + df_games['enemy 3 paint_point'] + df_games['enemy 4 paint_point']) / (df_games['team members']-1)
    
    # kill / death & occupation of kill & death
    df_games['team k/d'] = df_games['team kills'] / df_games['team deaths']
    df_games['enemy k/d'] = df_games['enemy kills'] / df_games['enemy deaths']
    df_games['kill occupy'] = df_games['my kill'] / (df_games['my kill'] + df_games['team kills']*(df_games['team members']-1))
    df_games['death occupy'] = df_games['my death'] / (df_games['my death'] + df_games['team deaths']*(df_games['team members']-1))
    df_games['paint occupy'] = df_games['my paint_point'] / (df_games['my paint_point'] + df_games['team paint_points']*(df_games['team members']-1))
    df_games['team k/d'].loc[df_games['team deaths'] == 0] = df_games['team kills']
    df_games['enemy k/d'].loc[df_games['enemy deaths'] == 0] = df_games['enemy kills']

    return df_games


# レーダーチャートを作成するようの関数
# https://analytics-note.xyz/programming/matplotlib-radar-chart/
def make_rader_chart(rader_value, rgrids, title, labels, legend_names):
    # print("rader_value(@defmake_rader_cahrt): ", rader_value.shape)
    angles = np.linspace(0, 2*np.pi, len(labels)+1, endpoint=True)
    fig = plt.figure(figsize=(7, 7), facecolor='w')
    ax = fig.add_subplot(1, 1, 1, polar=True)

    if len(legend_names) == 1:
        ax.plot(angles, rader_value)
        # レーダーチャート内を塗りつぶす
        ax.fill(angles, rader_value, alpha=0.2)
    else:
        # レーダーチャートの線を引く
        for i in range(rader_value.shape[0]):
            ax.plot(angles, rader_value[i])
            # レーダーチャート内を塗りつぶす
            ax.fill(angles, rader_value[i], alpha=0.2)

    # 項目ラベルの表示
    ax.set_thetagrids(angles[:-1] * 180/np.pi, labels)
    # 円形のメモリ線を消す
    ax.set_rgrids([])
    # 一番外側の円を消す
    ax.spines['polar'].set_visible(False)
    # 視点を上(来た)に変更
    ax.set_theta_zero_location("N")
    # 時計周りに変更
    ax.set_theta_direction(-1)
    # legendをつける
    ax.legend(legend_names, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax.legend(legend_names, prop = {"family" : "Ms Gothic"})

    # 多角形の目盛線を引く
    for grid_value in rgrids:
        grid_values = [grid_value] * (len(labels)+1)
        ax.plot(angles, grid_values, color="gray",  linewidth=0.5)

    # メモリの値を表示する
    for t in rgrids:
        # xが偏角、yが絶対値でテキストの表示場所が指定される
        ax.text(x=0, y=t, s=t)
    # rの範囲を指定
    ax.set_rlim([min(rgrids), max(rgrids)])

    ax.set_title(title, pad=20)
    # plt.plot()
    return fig

# 入力したデータの自分，味方，敵のリザルト平均を取得
@st.cache(allow_output_mutation=True)
def view_result_mean(df):
    rgrids = [0, 2, 4, 6, 8, 10]
    labels = ["kill", "death", "assist", "sp_count", "paint_point"]
    my_labels = ["my kill", "my death", "my assist", "my sp_count", "my paint_point"]
    team_labels = ["team kills", "team deaths", "team assists", "team sp_counts", "team paint_points"]
    enemy_labels = ["enemy kills", "enemy deaths", "enemy assists", "enemy sp_counts", "enemy paint_points"]

    my_value = np.zeros(len(labels))
    for i in range(len(my_labels)):
        my_value[i] = df[my_labels[i]].mean()
    my_value[4] = my_value[4]/200
    my_rader_value = np.concatenate([my_value, [my_value[0]]])

    team_value = np.zeros(len(labels))
    for i in range(len(team_labels)):
        team_value[i] = df[team_labels[i]].mean()
    team_value[4] = team_value[4]/200
    team_rader_value = np.concatenate([team_value, [team_value[0]]])

    enemy_value = np.zeros(len(labels))
    for i in range(len(enemy_labels)):
        enemy_value[i] = df[enemy_labels[i]].mean()
    enemy_value[4] = enemy_value[4]/200
    enemy_rader_value = np.concatenate([enemy_value, [enemy_value[0]]])

    rader_value = np.vstack([my_rader_value, team_rader_value, enemy_rader_value])
    fig = make_rader_chart(rader_value, rgrids, title="result mean", labels=labels, legend_names=["my result", "team result", "enemy result"])
    return fig

# Sub マッチングした別の人の武器ごとのリザルト取得
def sub_get_weapon_kill_death(main_weapon, result, weapon_list, output):
    target_index = np.where(weapon_list==main_weapon)[0]
    output[target_index, 0] += result[0]
    output[target_index, 1] += result[1]
    output[target_index, 2] += result[2]
    output[target_index, 3] += result[3]
    output[target_index, 4] += result[4]
    output[target_index, 5] += result[5]
    output[target_index, 6] += 1
    return output

# マッチングした別の人の武器ごとのリザルト取得
@st.cache(allow_output_mutation=True)
def get_weapon_result(df):
    # get team and enemy member result divided by main weapon
    df_team_weapons = pd.concat([df['ally 1 main'], df['ally 2 main'], df['ally 3 main']])
    # df_enemy_weapons = pd.concat([df['enemy 1 main'], df['enemy 2 main'], df['enemy 3 main'], df['enemy 4 main']])
    weapon_list = df_team_weapons.unique()

    # ['負け数', '勝ち数', '試合数', '勝率']
    team_weapon_result = np.zeros((len(weapon_list), 4))
    enemy_weapon_result = np.zeros((len(weapon_list), 4))

    # 一緒になった武器の勝ち数・負け数を取得
    for i in range(len(df)):
        tmp = df.iloc[i]
        tmp_result = tmp['result']
        my_weapon_list = [tmp["ally 1 main"], tmp["ally 2 main"], tmp["ally 3 main"]]
        ene_weapon_list = [tmp["enemy 1 main"], tmp["enemy 2 main"], tmp["enemy 3 main"], tmp["enemy 4 main"]]
        set_my_weapon = set(my_weapon_list)
        set_ene_weapon = set(ene_weapon_list)

        for my_weapon in set_my_weapon:
            team_weapon_result[np.where(weapon_list==my_weapon)[0], tmp_result] += 1
        for ene_weapon in set_ene_weapon:
            enemy_weapon_result[np.where(weapon_list==ene_weapon)[0], tmp_result] += 1
    
    # 勝率を計算
    for i in range(team_weapon_result.shape[0]):
        team_weapon_result[i, 2] = team_weapon_result[i, 1] + team_weapon_result[i, 0]
        team_weapon_result[i, 3] = team_weapon_result[i, 1] / team_weapon_result[i, 2]
        enemy_weapon_result[i, 2] = enemy_weapon_result[i, 1] + enemy_weapon_result[i, 0]
        enemy_weapon_result[i, 3] = enemy_weapon_result[i, 1] / enemy_weapon_result[i, 2]

    df_weapon_result = pd.DataFrame(weapon_list, columns=['weapon'])
    df_weapon_result = pd.concat([df_weapon_result,
                                  pd.DataFrame(team_weapon_result[:, 2:],
                                               columns=["all games (ally)", "win rate (ally)"])],
                                 axis=1)
    df_weapon_result = pd.concat([df_weapon_result,
                                  pd.DataFrame(enemy_weapon_result[:, 2:],
                                               columns=["all games (enemy)", "win rate (enemy)"])],
                                 axis=1)

    # マッチングした武器のkill数 death数 special数 paint_point数の平均
    # svedata = [kill, death, k/d, assist, special, all_k/d, count]
    weapon_kd_result = np.zeros((len(df_team_weapons.unique()), 7))
    for i in range(len(df)):
        # 算出条件
        tmp = df.iloc[i]
        tmp_result_1 = [tmp['ally 1 kill'], tmp['ally 1 death'], tmp['ally 1 k/d'], tmp['ally 1 assist'], tmp['ally 1 paint_point'], tmp['ally 1 sp_count']]
        tmp_result_2 = [tmp['ally 2 kill'], tmp['ally 2 death'], tmp['ally 2 k/d'], tmp['ally 2 assist'], tmp['ally 2 paint_point'], tmp['ally 2 sp_count']]
        tmp_result_3 = [tmp['ally 3 kill'], tmp['ally 3 death'], tmp['ally 3 k/d'], tmp['ally 3 assist'], tmp['ally 3 paint_point'], tmp['ally 3 sp_count']]
        tmp_result_4 = [tmp['enemy 1 kill'], tmp['enemy 1 death'], tmp['enemy 1 k/d'], tmp['enemy 1 assist'], tmp['enemy 1 paint_point'], tmp['enemy 1 sp_count']]
        tmp_result_5 = [tmp['enemy 2 kill'], tmp['enemy 2 death'], tmp['enemy 2 k/d'], tmp['enemy 2 assist'], tmp['enemy 2 paint_point'], tmp['enemy 2 sp_count']]
        tmp_result_6 = [tmp['enemy 3 kill'], tmp['enemy 3 death'], tmp['enemy 3 k/d'], tmp['enemy 3 assist'], tmp['enemy 3 paint_point'], tmp['enemy 3 sp_count']]
        tmp_result_7 = [tmp['enemy 4 kill'], tmp['enemy 4 death'], tmp['enemy 4 k/d'], tmp['enemy 4 assist'], tmp['enemy 4 paint_point'], tmp['enemy 4 sp_count']]

        # get result
        weapon_kd_result = sub_get_weapon_kill_death(tmp["ally 1 main"], tmp_result_1, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["ally 2 main"], tmp_result_2, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["ally 3 main"], tmp_result_3, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["enemy 1 main"], tmp_result_4, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["enemy 2 main"], tmp_result_5, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["enemy 3 main"], tmp_result_6, weapon_list, weapon_kd_result)
        weapon_kd_result = sub_get_weapon_kill_death(tmp["enemy 3 main"], tmp_result_7, weapon_list, weapon_kd_result)

    for i in range(6): 
        weapon_kd_result[:, i] =  weapon_kd_result[:, i]/  weapon_kd_result[:, 6]

    df_weapon_result = pd.concat([df_weapon_result,
                                  pd.DataFrame(weapon_kd_result[:, :-1],
                                               columns=["kill", "death", "k/d", "assist", "paint_point", "special"])],
                                 axis=1)
    return df_weapon_result


# 強い武器・弱い武器のレーダーチャートを出力
def get_strong_and_weak_weapon(df, if_lose=True, if_ally=True, plot_num=3):
    # 回線落ちのデータは削除 (回線落ちデータは塗がゼロのはずなので)
    df = df.drop(df.index[df['weapon']=='バカ'])
    
    # target_df_weapon = df[(df['all games (ally)'] >= thresh_game) & (df['all games (enemy)'] >= thresh_game)]
    labels = ["kill", "death", "k/d", "assist", "paint_point", "special"]
    # リザルトを正規化
    for item in labels:
        if item =='death':
            # df[item] = round((df[item] - df[item].max()) * 100 / (df[item].min() - df[item].max()), 2)
            df[item] = 50 - 20 * (df[item] - df[item].mean())/df[item].std()
        else:
            # df[item] = round((df[item] - df[item].min()) * 100 / (df[item].max() - df[item].min()), 2)
            df[item] = 50 + 20 * (df[item] - df[item].mean())/df[item].std()
    
    if if_ally:
        tmp_df = df.sort_values(by='win rate (ally)', ascending=if_lose).head(plot_num)
        title = '味方に来ると負ける武器' if if_lose else '味方に来ると勝てる武器'
    else:
        tmp_df = df.sort_values(by='win rate (enemy)', ascending=if_lose).head(plot_num)
        title = '敵に来ると負ける武器' if if_lose else '敵に来ると勝てる武器'
    
    legend_names = tmp_df['weapon'].unique()
    tmp_df = tmp_df.loc[:, labels]
    rgrids = [0, 20, 40, 60, 80, 100]
    value = tmp_df.iloc[0].values
    rader_value =np.concatenate([value, [value[0]]])
    if plot_num > 1:
        for i in range(plot_num-1):
            value = tmp_df.iloc[i+1].values
            tmp_rader_value =np.concatenate([value, [value[0]]])
            rader_value = np.vstack([rader_value, tmp_rader_value])
    
    # 0-100の間の値に収める
    rader_value[np.where(rader_value < 0)] = 0
    rader_value[np.where(rader_value > 100)] = 100
            
    return make_rader_chart(rader_value, rgrids, title, labels, legend_names)
        
@st.cache(allow_output_mutation=True)
def get_kill_death_result_by_stage(df_games):
    stage_names = df_games['stage'].unique()
    plot_data = np.zeros((len(stage_names), 4))
    
    i = 0
    for stage in stage_names:
        tmp_df = df_games[df_games['stage']==stage]
        plot_data[i, 0] = tmp_df['my kill'].mean()
        plot_data[i, 1] = tmp_df['my death'].mean()
        plot_data[i, 2] = tmp_df['result'].mean()
        plot_data[i, 3] = tmp_df['result'].std() * 1/2
        i += 1
        
    good_at_index = np.where(plot_data[:, 2]>0.5)[0]
    
    x = np.arange(plot_data.shape[0])
    fig = plt.figure(figsize=(15, 5))
    # kill数 death数のプロット
    ax1 = fig.add_subplot(111)
    ax1.bar(x, plot_data[:, 0], align="edge", width=-0.3, color='indianred')
    ax1.bar(x, plot_data[:, 1], align="edge", width=0.3, color='paleturquoise')
    ax1.grid()
    ax1.legend(['kill', 'death'])
    ax1.set_ylabel("kill & death num")
    plt.xticks(x, stage_names, rotation=45)
    
    # 勝率50%を超えているステージだけ分かりやすく
    labels = ax1.get_xticklabels()
    if len(good_at_index) > 0:
        for stage_index in good_at_index:
            labels[stage_index].set_color('tab:red')
    
    # 勝率のプロット
    ax2 = ax1.twinx()
    ax2.plot(x, plot_data[:, 2]*100, color='indigo', linestyle='dashed')
    ax2.errorbar(x, plot_data[:, 2]*100, plot_data[:, 3]*100, capsize=5, fmt='D', markersize=7,
                 ecolor='indigo', markeredgecolor ='indigo', color='w', linewidth=3)
    ax2.plot([x[0], x[-1]], [50, 50], "k-")
    ax2.set_ylabel("win rate (%)")
    ax2.spines['right'].set_color('red')
    ax2.grid(color='r', linestyle='dotted')
    ax2.tick_params(axis = 'y', colors ='indigo')
    ax2.set_ylim([0, 100])
    
    return fig
    

