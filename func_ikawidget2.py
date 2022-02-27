import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    # df_games['team specials'] = (df_my_player_1_result['special'] + df_my_player_2_result['special'] + df_my_player_3_result['special']) / (df_games['team members']-1)
    df_games['team paint points'] = (df_games['ally 1 paint_point'] + df_games['ally 2 paint_point'] + df_games['ally 3 paint_point']) / (df_games['team members']-1)

    # average result enemy
    df_games['enemy kills'] = (df_games['enemy 1 kill'] + df_games['enemy 2 kill'] + df_games['enemy 3 kill'] + df_games['enemy 4 kill']) / df_games['enemy members']
    df_games['enemy assists'] = (df_games['enemy 1 assist'] + df_games['enemy 2 assist'] + df_games['enemy 3 assist'] + df_games['enemy 4 assist']) / df_games['enemy members']
    df_games['enemy deaths'] = (df_games['enemy 1 death'] + df_games['enemy 2 death'] + df_games['enemy 3 death'] + df_games['enemy 4 death']) / df_games['enemy members']
    # df_games['team specials'] = (df_my_player_1_result['special'] + df_my_player_2_result['special'] + df_my_player_3_result['special']) / (df_games['team members']-1)
    df_games['enemy paint points'] = (df_games['enemy 1 paint_point'] + df_games['enemy 2 paint_point'] + df_games['enemy 3 paint_point'] + df_games['enemy 4 paint_point']) / (df_games['team members']-1)
    
    # kill / death & occupation of kill & death
    df_games['team k/d'] = df_games['team kills'] / df_games['team deaths']
    df_games['enemy k/d'] = df_games['enemy kills'] / df_games['enemy deaths']
    df_games['kill occupy'] = df_games['my kill'] / (df_games['my kill'] + df_games['team kills']*(df_games['team members']-1))
    df_games['death occupy'] = df_games['my death'] / (df_games['my death'] + df_games['team deaths']*(df_games['team members']-1))
    df_games['paint occupy'] = df_games['my paint_point'] / (df_games['my paint_point'] + df_games['team paint points']*(df_games['team members']-1))

    return df_games