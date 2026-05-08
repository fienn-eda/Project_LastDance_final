import streamlit as st
import pandas as pd
import plotly.express as px
import os
import random

# ==========================================
# 0. 페이지 기본 설정 및 유틸리티 함수
# ==========================================
st.set_page_config(page_title="LoL 오토필 생존 가이드", page_icon="🎮", layout="wide")

# 포지션 한글화 매핑 딕셔너리
POS_KOR_MAP = {
    'TOP': '탑',
    'JUNGLE': '정글',
    'MIDDLE': '미드',
    'BOTTOM': '원딜',
    'UTILITY': '서폿'
}
# 역매핑 (한글 -> 영문)
KOR_POS_MAP = {v: k for k, v in POS_KOR_MAP.items()}

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    match_path = os.path.join(current_dir, 'match_data_v4_light.parquet')
    champ_path = os.path.join(current_dir, 'champ_data_v4.parquet')
    user_path = os.path.join(current_dir, 'user_profile_v4_light.parquet')
    
    return pd.read_parquet(match_path), pd.read_parquet(champ_path), pd.read_parquet(user_path)

try:
    df_match, df_champ, user_profile_df = load_data()
except FileNotFoundError as e:
    st.error(f"데이터 파일을 찾을 수 없습니다: {e}")
    st.stop()

# PUUID -> 랜덤닉네임 변환 함수
@st.cache_data
def generate_nicknames(puuid_list):
    random.seed(42) # 항상 동일한 닉네임이 부여되도록 시드 고정
    adjs = ["GEN", "T1", "KT", "NS", "DNS", "BRO", "HLE", "DK", "BFX", "DRX", "SSG"]
    nouns = ["Kiin", "Canyon", "Chovy", "Ruler", "Duro", "Doran", "Oner", "Faker", "Peyz", "Keria", "Zeus", "Kanavi", "Zeka", "Gumayusi", "Delight",
            "Cuzz", "Bdd", "Ghost", "Smash", "Lucid", "Diable", "Kingen", "DuDu", "Pyosik", "Clozer", "deokdam", "Peter"]
    
    nick_map = {}
    for i, p in enumerate(puuid_list):
        nick_map[p] = f"{random.choice(adjs)} {random.choice(nouns)} #{i+1}"
    return nick_map

puuid_to_nick = generate_nicknames(user_profile_df['puuid'].unique().tolist())

# ==========================================
# 1. 추천 로직 V4 (사거리 3단계 분류 적용)
# ==========================================
def recommend_autofill_v4(target_puuid, target_position, df_match, df_champ, user_profile_df, 
                          banned_champs=[], team_needs='balanced', is_ranked=True, top_n=5):
    
    user_data = user_profile_df[user_profile_df['puuid'] == target_puuid].iloc[0]
    primary_cluster = user_data['primary_cluster_id']
    secondary_cluster = user_data['secondary_cluster_id']
    user_ad_pref = user_data['avg_preferred_attack']
    user_ap_pref = user_data['avg_preferred_magic']
    user_range_pref = user_data['avg_preferred_range']
    
    # [요청 1] 사거리 선호도 3단계 분류
    if user_range_pref < 250:
        user_range_type = 'melee'   # 근거리 선호
    elif user_range_pref > 400:
        user_range_type = 'ranged'  # 원거리 선호
    else:
        user_range_type = 'neutral' # 중립 (올라운더)
        
    played_champs_keys = df_match[df_match['puuid'] == target_puuid]['champ_match_key'].unique().tolist()
    
    position_meta = df_match[df_match['team_position'] == target_position]['champ_match_key'].value_counts()
    banned_keys = df_champ[df_champ['champion_name'].isin(banned_champs)]['champ_match_key'].tolist()
    
    valid_candidate_keys = [c for c in position_meta.head(30).index if c not in banned_keys]
    candidates_df = df_champ[df_champ['champ_match_key'].isin(valid_candidate_keys)].copy()
    
    candidates_df['score_prof'] = 0.0
    candidates_df['score_team'] = 0.0
    candidates_df['score_style'] = 0.0
    candidates_df['score_diff'] = 0.0
    candidates_df['total_score'] = 0.0
    candidates_df['recommend_reason'] = ""
    candidates_df['is_played'] = candidates_df['champ_match_key'].isin(played_champs_keys)
    
    for idx, row in candidates_df.iterrows():
        s_prof, s_team, s_style, s_diff, s_stat = 0, 0, 0, 0, 0
        reason = []
        
        # A. 숙련도
        if row['is_played']:
            s_prof += 30 if is_ranked else 10
        else:
            if is_ranked: s_prof -= 20
                
        # B. 팀 조합
        if team_needs == 'AP_needed' and row['info_magic'] >= 6:
            s_team += 20
            reason.append("아군 AP 보완")
        elif team_needs == 'AD_needed' and row['info_attack'] >= 6:
            s_team += 20
            reason.append("아군 AD 보완")
        else:
            s_stat += max(0, 10 - (abs(row['info_attack'] - user_ad_pref) + abs(row['info_magic'] - user_ap_pref)))
            
        # C. [요청 1 반영] 사거리 성향 보정 (중립 추가)
        is_champ_ranged = row['attackrange'] > 300
        champ_range_str = "원거리" if is_champ_ranged else "근거리"
        
        if user_range_type == 'neutral':
            s_stat += 5 # 올라운더에게는 약한 기본 가산점
            reason.append("잡식성")
        elif (user_range_type == 'melee' and not is_champ_ranged) or (user_range_type == 'ranged' and is_champ_ranged):
            s_stat += 10
            reason.append(f"{champ_range_str} 챔피언")
            
        # D. 스타일
        if row['cluster_id'] == primary_cluster:
            s_style += 20 if is_ranked else 40
            reason.append(f"주력 스타일 일치 ({row['cluster_name']})")
        elif row['cluster_id'] == secondary_cluster:
            s_style += 15 if is_ranked else 30
            reason.append(f"서브 스타일 일치 ({row['cluster_name']})")
            
        # E. 난이도
        if row['info_difficulty'] <= 4:
            s_diff += 20 if is_ranked else 40
            reason.append("쉬운 조작 난이도")
            
        candidates_df.at[idx, 'score_prof'] = s_prof
        candidates_df.at[idx, 'score_team'] = s_team
        candidates_df.at[idx, 'score_style'] = s_style
        candidates_df.at[idx, 'score_diff'] = s_diff
        candidates_df.at[idx, 'total_score'] = s_prof + s_team + s_style + s_diff + s_stat
        candidates_df.at[idx, 'recommend_reason'] = " / ".join(reason) if reason else "무난한 픽"
        
    # 정렬 및 탐험 픽 할당 로직
    if is_ranked:
        sorted_df = candidates_df.sort_values(by=['total_score', 'score_prof', 'score_diff', 'score_style', 'score_team', 'champion_name'], ascending=[False, False, False, False, False, True])
        final_recommendation = sorted_df.head(top_n)
    else:
        sorted_df = candidates_df.sort_values(by=['total_score', 'score_style', 'score_diff', 'champion_name'], ascending=[False, False, False, True])
        exploration_pool = sorted_df[(~sorted_df['is_played']) & (sorted_df['info_difficulty'] <= 5)]
        
        num_wildcards = min(2, len(exploration_pool))
        # [0307 수정] '점수 상위 10개' 안에서 랜덤 추출
        if num_wildcards > 0:
            # 풀이 10개보다 적으면 전체 풀 크기로 제한
            pool_size = min(10, len(exploration_pool)) 
            top_exploration_pool = exploration_pool.head(pool_size)
            
            # 상위 10개 중에서 랜덤으로 num_wildcards(2개) 추출
            wildcards = top_exploration_pool.sample(n=num_wildcards).copy()
            wildcards['recommend_reason'] = "💡 새로운 도전 / " + wildcards['recommend_reason']
        else:
            # 탐험 풀이 아예 없을 경우를 대비한 빈 데이터프레임
            wildcards = pd.DataFrame()
        
        # 나머지 자리는 랭크게임 기준 추천 픽으로 채움
        num_regulars = top_n - num_wildcards
        if not wildcards.empty:
            remaining_pool = sorted_df[~sorted_df['champ_match_key'].isin(wildcards['champ_match_key'])]
        else:
            remaining_pool = sorted_df
            
        regulars = remaining_pool.head(num_regulars).copy()
        
        # 합친 후 최종 정렬
        final_recommendation = pd.concat([regulars, wildcards]).sort_values(by=['total_score', 'score_style', 'score_diff', 'champion_name'], ascending=[False, False, False, True])
            
    # 특성 표시를 위해 전체 컬럼을 다 반환하도록 수정
    return final_recommendation

# ==========================================
# 2. 사이드바 (사용자 입력)
# ==========================================
st.sidebar.header("시뮬레이션 설정")

# [요청 2] 랜덤 닉네임으로 유저 선택
user_options = list(puuid_to_nick.values())
selected_nick = st.sidebar.selectbox("대상 유저 선택", user_options)
# 선택한 닉네임을 다시 실제 puuid로 역추적
selected_puuid = [k for k, v in puuid_to_nick.items() if v == selected_nick][0]

# [요청 4] 포지션 한글화
selected_pos_kor = st.sidebar.selectbox("배정받은 포지션", list(POS_KOR_MAP.values()))
selected_pos_eng = KOR_POS_MAP[selected_pos_kor]

# 밴 카드 (챔피언 이름도 ko_KR을 불러왔다면 한글로 표시됨)
all_champs = sorted(df_champ['champion_name'].unique())
bans = st.sidebar.multiselect("🚫 밴 챔피언 (최대 10개)", all_champs, default=[], max_selections=10)

needs = st.sidebar.radio("아군 조합 상황", options=['balanced', 'AP_needed', 'AD_needed'],
                         format_func=lambda x: '무난함' if x == 'balanced' else ('AP 부족' if x == 'AP_needed' else 'AD 부족'))

is_ranked = st.sidebar.toggle("랭크 게임인가요?", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("추천 챔피언 분석하기"):
    st.session_state['run'] = True

# ==========================================
# 3. 메인 화면 - 탭 구성
# ==========================================
st.title("당신만을 위한 챔피언 추천 시스템")

# tab_recom, tab_cluster = st.tabs(["맞춤형 챔피언 추천", "챔피언 특징 기반 클러스터"])
# [0307 수정] 탭을 3개로 분리 (유저 프로필 분석 탭 추가)
tab_recom, tab_user, tab_cluster = st.tabs(["맞춤형 챔피언 추천", "유저 프로필 분석", "챔피언 특징 기반 클러스터"])

# ----------------- 1. 추천 탭 -----------------
with tab_recom:
    if st.session_state.get('run'):
        st.subheader(f"[{selected_nick}] 님을 위한 '{selected_pos_kor}' 포지션 Top 5")
        
        with st.spinner('유저 성향과 메타를 분석하고 있습니다...'):
            result_df = recommend_autofill_v4(selected_puuid, selected_pos_eng, df_match, df_champ, user_profile_df, bans, needs, is_ranked)
            
            if isinstance(result_df, str):
                st.error(result_df)
            else:
                cols = st.columns(5)
                for rank, (idx, row) in enumerate(result_df.iterrows()):
                    with cols[rank]:
                        # [요청 6] 챔피언 공식 이미지 삽입 (데이터에 champion_id가 있어야 함. 없으면 영문명 우회 필요)
                        try:
                            champ_img_id = row['champion_id']
                        except KeyError:
                            # 만약 champion_id를 안 만드셨다면 임시로 작동하게 하는 방어 코드
                            champ_img_id = row['champ_match_key'].capitalize()
                            
                        img_url = f"https://ddragon.leagueoflegends.com/cdn/16.2.1/img/champion/{champ_img_id}.png"
                        st.image(img_url, width='stretch')
                        
                        st.markdown(f"**{rank+1}위: {row['champion_name']}**")
                        st.write(f"**총점:** {int(row['total_score'])}점")
                        
                        # [요청 5] 챔피언 특성 배지 생성
                        dmg_type = "🔮 AP" if row['info_magic'] > row['info_attack'] else "🗡️ AD"
                        range_type = "🏹 원거리" if row['attackrange'] > 300 else "🛡️ 근거리"
                        style_match = "✅ 스타일 일치" if row['score_style'] > 0 else "➖ 스타일 무관"
                        
                        diff_val = row['info_difficulty']
                        if diff_val <= 4: diff_str = "⭐ 쉬움"
                        elif diff_val <= 7: diff_str = "⭐⭐ 보통"
                        else: diff_str = "⭐⭐⭐ 어려움"
                        
                        # 깔끔한 캡션 형태로 특성 나열
                        st.caption(f"{dmg_type} | {range_type}")
                        st.caption(f"{style_match} | {diff_str}")
                        
                        # 기존 추천 사유
                        st.info(row['recommend_reason'])
    else:
        st.info("👈 좌측 사이드바에서 시뮬레이션 설정을 맞춘 후 '추천 챔피언 분석하기' 버튼을 눌러주세요.")

# ----------------- 2. [0307 추가] 유저 프로필 분석 탭 -----------------
with tab_user:
    st.subheader(f"🔎 [{selected_nick}] 님의 플레이 성향 분석")
    
    # 해당 유저의 프로필 데이터 추출
    user_row = user_profile_df[user_profile_df['puuid'] == selected_puuid].iloc[0]
    
    # 1. 메인 포지션 / 서브 포지션 (한글화)
    # POS_KOR_MAP에 없는 값(예: 'None')일 경우 그대로 출력하도록 get() 활용
    main_pos_kor = POS_KOR_MAP.get(user_row['main_position'], user_row['main_position'])
    sub_pos_kor = POS_KOR_MAP.get(user_row['sub_position'], user_row['sub_position'])
    
    col1, col2 = st.columns(2)
    col1.metric("🥇 주 포지션", main_pos_kor)
    col2.metric("🥈 부 포지션", sub_pos_kor)
    st.markdown("---")
    
    # 유저의 매치 전적에서 챔피언별 플레이 횟수 집계
    user_history = df_match[df_match['puuid'] == selected_puuid]
    play_counts = user_history['champ_match_key'].value_counts().reset_index()
    play_counts.columns = ['champ_match_key', 'play_count']
    
    if play_counts.empty:
        st.warning("이 유저의 챔피언 플레이 기록이 부족합니다.")
    else:
        # 플레이 횟수 데이터와 챔피언 세부 정보(df_champ) 결합
        merged_history = pd.merge(play_counts, df_champ, on='champ_match_key', how='inner')
        
        # 만약 cluster_name이 없다면 cluster_id로 임시 대체 (에러 방지용)
        if 'cluster_name' not in merged_history.columns:
            merged_history['cluster_name'] = "클러스터 " + merged_history['cluster_id'].astype(str)
            
        # 2. 클러스터별 플레이 비율 파이 차트
        cluster_agg = merged_history.groupby('cluster_name')['play_count'].sum().reset_index()
        
        fig_pie = px.pie(
            cluster_agg, 
            names='cluster_name', 
            values='play_count', 
            title="선호하는 챔피언 스타일 (클러스터별 플레이 비율)",
            hole=0.4, # 도넛 차트 형태로 디자인 (0이면 일반 파이차트)
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(title_font=dict(size=24), margin=dict(t=80))
        st.plotly_chart(fig_pie, width='stretch')
        
        # 3. 특정 클러스터 클릭(선택) 시 상세 챔피언 리스트 표시
        st.markdown("#### 📝 클러스터별 세부 플레이 챔피언 정보")
        st.caption("파이 차트에 있는 클러스터를 아래에서 선택하면, 해당 스타일의 챔피언 플레이 기록과 세부 스탯을 볼 수 있습니다.")
        
        # 셀렉트박스로 파이 차트의 조각을 '클릭'하는 효과 구현
        selected_cluster_for_details = st.selectbox(
            "상세 정보를 확인할 클러스터를 선택하세요:", 
            cluster_agg['cluster_name'].tolist()
        )
        
        # 선택한 클러스터의 챔피언만 필터링
        details_df = merged_history[merged_history['cluster_name'] == selected_cluster_for_details].copy()
        
        # 표시할 데이터 가공 (원거리/근거리 판별 등)
        details_df['range_type'] = details_df['attackrange'].apply(lambda x: '원거리' if x > 300 else '근거리')
        
        # 화면에 띄울 컬럼만 추출 및 이름 한글화
        display_df = details_df[[
            'champion_name', 'play_count', 'range_type', 'info_difficulty', 
            'info_attack', 'info_magic', 'info_defense'
        ]].copy()
        
        display_df.columns = [
            '챔피언', '플레이 횟수(숙련도)', '전투 사거리', '난이도 (1~10)', 
            '물리 데미지', '마법 데미지', '탱킹력'
        ]
        
        # 플레이 횟수 기준 내림차순 정렬
        display_df = display_df.sort_values(by='플레이 횟수(숙련도)', ascending=False).reset_index(drop=True)
        
        # 데이터프레임 시각화 (인덱스 숨김)
        st.dataframe(display_df, width='stretch', hide_index=True)
        
# ----------------- 3. 클러스터 데이터 탭 -----------------
with tab_cluster:
    st.subheader("챔피언 성향 클러스터링 매핑 (PCA)")
    df_champ['cluster_id'] = df_champ['cluster_id'].astype(str)
    
    color_col = 'cluster_name' if 'cluster_name' in df_champ.columns else 'cluster_id'
    
    fig_scatter = px.scatter(
        df_champ, 
        x='pca_x', 
        y='pca_y', 
        color=color_col, 
        hover_name='champion_name',
        # [0307 수정] 툴팁(Hover)에 표시할 데이터와 숨길 데이터 명시 (False = 숨김)
        hover_data={
            'pca_x': False,
            'pca_y': False,
            'info_attack': True,
            'info_magic': True,
            'info_difficulty': True,
            'attackrange': True,
            color_col: True  # 소속된 클러스터 이름도 표시
        },
        # [0307 수정] 영어 컬럼명을 한글로 매핑
        labels={
            'info_attack': '물리 데미지',
            'info_magic': '마법 데미지',
            'info_difficulty': '난이도',
            'attackrange': '사거리',
            'cluster_name': '플레이 성향',
            'cluster_id': '클러스터 번호'
        },
        title="<b>챔피언별 스탯 기반 전투 스타일 맵</b>",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # 그래프 높이와 제목 크기 조정
    fig_scatter.update_layout(height=600, title_font=dict(size=24), margin=dict(t=80))
    st.plotly_chart(fig_scatter, width='stretch')
    st.caption("※ 머신러닝이 챔피언의 기본 스탯과 성장스탯을 분석하여 유사한 전투 스타일끼리 묶어낸 지도입니다.")