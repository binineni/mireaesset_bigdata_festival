#라이브러리 import
#필요한 경우 install

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import seaborn as sns
import time
from streamlit_option_menu import option_menu

#########################################중요###########################################
# cd C:/Users/sook7/미래에솦_주피터/본선코드정리/streamlit_mockup
# 터미널에서 명령어(streamlit run 추천공모주.py)를 실행 시켜주어야 스트림릿이 작동함
#######################################################################################

#페이지를 위한 코드
#layout = wide : 화면 설정 디폴트값을 와이드로
st.set_page_config(page_title="추천공모주")

#메뉴 탭 하단 사이드바에 이미지 넣기
with st.sidebar:
    choose = option_menu("메뉴", ["진행 예정 청약", "최근 상장한 기업 목록"],
                         icons=['메뉴 아이콘1', '메뉴 아이콘2'],
                         menu_icon="bi bi-question-circle", default_index=0,
                         styles={
                         # default_index = 처음에 보여줄 페이지 인덱스 번호
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    } # css 설정
    )

image = Image.open('streamlit_mockup/img/미래에셋로고.png')
image2 = Image.open('streamlit_mockup/img/네이버클라우드.png')
image3 = Image.open('streamlit_mockup/img/미래에솦.png')

st.sidebar.image(image, use_column_width=True)
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image3, use_column_width=True)

###########################################################################################
#필요한 데이터셋 불러오기
# 화면이 업데이트될 때 마다 변수 할당이 된다면 시간이 오래 걸려서 @st.cache_data 사용(캐싱)
@st.cache_data
def load_streamlit_data():
    df = pd.read_csv('streamlit_mockup/data/data_streamlit_df.csv')
    df['시초/공모(%)'] = df['시초/공모(%)'].str.rstrip('%')
    df['시초/공모(%)'] = pd.to_numeric(df['시초/공모(%)'])
    df['예측일'] = pd.to_datetime(df['예측일']).dt.date
    df['신규상장일'] = pd.to_datetime(df['신규상장일']).dt.date
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df = df.sort_values(by=['예측일', '신규상장일'], ascending=[False, False])

    return df

df = load_streamlit_data()

@st.cache_data
def load_company_analysis_data():
    comp_analysis = pd.read_csv('streamlit_mockup/data/data_요약서비스.csv')
    comp_analysis['종목코드'] = comp_analysis['종목코드'].astype(str).str.zfill(6)

    return comp_analysis

comp_analysis = load_company_analysis_data()
#################################################################################


#별점을 매기는 기준
f_quantiles = [df["f_score"].quantile(q=0.3), df["f_score"].quantile(q=0.5),
               df["f_score"].quantile(q=0.7), df["f_score"].quantile(q=0.9)]

t_quantiles = [df["t_score"].quantile(q=0.3), df["t_score"].quantile(q=0.5),
               df["t_score"].quantile(q=0.7), df["t_score"].quantile(q=0.9)]

a_quantiles = [df["a_score"].quantile(q=0.3), df["a_score"].quantile(q=0.5),
               df["a_score"].quantile(q=0.7), df["a_score"].quantile(q=0.9)]


# 별점은 전체 df의 F, T, A quantile을 기준으로 함
def count_star(score, quantiles):
    if score <= quantiles[0]:
        star = 1
    elif score <= quantiles[1]:
        star = 2
    elif score <= quantiles[2]:
        star = 3
    elif score <= quantiles[3]:
        star = 4
    else:
        star = 5

    return star

# 주간사 정보를 처리하는 함수
def process_agency(row):
    agencies = row.split(',')
    if '미래에셋증권' in agencies:
        return '미래에셋증권'
    elif len(agencies) > 1:
        return ', '.join(agencies)  # 최대 두 개까지 표시
    elif len(agencies) == 1:
        return agencies[0]  # 하나만 있는 경우
    else:
        return None



#제목
st.title('추천공모주')
#info
st.info('"청약권장도"와 "공모주 청약판단 종합 점수"가 궁금하시면 좌측 메뉴 탭의 **FAQ** 를 이용해 주세요.')
st.info('다크모드를 사용중이시라면, 다크모드를 해제해주세요!')

st.divider()

#진행 예정 청약 탭과 최근 상장기업 목록을 나누기
# today = datetime.datetime.now().date()
today = datetime.date(2023, 8, 1)

#df_pred : 예측일 ~ 상장일 사이에 있는 추천할 기업
df_pred = df[(df['예측일'] <= today) & (df['신규상장일'] >= today)]
df_pred.reset_index(drop=True,inplace=True)

#df : 예측 완료되었고 실제 결과가 나온 기업
df_done = df[~df.index.isin(df_pred.index)]
df_done.reset_index(drop=True,inplace=True)

# 진행 예정 청약 탭
if choose == "진행 예정 청약":

    st.text('')

    st.header('진행 예정 청약')
    st.text('')

    # 기업 정보를 담은 박스를 만듭니다.

    for i in range(0, len(df_pred)):

        # 청약권장도 박스 색
        color_list = ['#FF5733', '#FFD700', '#01DF01 ', '#009933']

        # 청약 권장도
        if df_pred['model_score'][i] < 25:
            comment = '위험'
            color = color_list[0]
        elif df_pred['model_score'][i] < 45:
            comment = '중립'
            color = color_list[1]
        elif df_pred['model_score'][i] < 70:
            comment = '권장'
            color = color_list[2]
        else:
            comment = '추천'
            color = color_list[3]


        # 데이터프레임에 주간사 정보 처리 결과 적용
        df_pred['주간사'] = df_pred['주간사'].apply(process_agency)

        st.markdown(
            f"""
            <div style="border: 2px solid #e5e5e5; padding: 10px; text-align: left;">
                <h2 style='color: #043B72;'>{df_pred['기업명'].values[i]}</h2>
                <p style='font-size: 20px;'><strong>청약 예정일 </strong> {df_pred['공모주일정'].values[i]}</p>
                <div style="display: flex; justify-content: space-between;">
                    <div style="border: 2px solid #FFFFFF; padding: 3px; background-color: {color};">
                    <p style='font-size: 20px; font-weight : bold; color : #FFFFFF; margin-bottom: 0;'>청약{comment}</p>
                    </div>
                </div>
                <div style="border: 2px solid #FFFFFF; padding: 10px; background-color: #FAFAFA;">    
                    <p style='font-size: 20px;'><strong>공모 예정가 </strong> {df_pred['공모희망가(원)'].values[i]}원</p>
                    <p style='font-size: 20px;'><strong>상장 예정일 </strong> {df_pred['신규상장일'].values[i]}</p>
                    <p style='font-size: 20px;'><strong>주간사 </strong> {df_pred['주간사'].values[i]}</p> 
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 예상 수익률
        if df_pred['범주'][i] == 0:
            percent = '~20%'
        elif df_pred['범주'][i] == 1:
            percent = '20%~60%'
        elif df_pred['범주'][i] == 2:
            percent = '60~100%'
        else:
            percent = '100%~'

        #F, T, A의 별점 계산
        f_star = count_star(df_pred['f_score'][i], f_quantiles)
        t_star = count_star(df_pred['t_score'][i], t_quantiles)
        a_star = count_star(df_pred['a_score'][i], a_quantiles)

        # st.tab 함수를 통해 하단에 tab 메뉴 생성
        tab1, tab2, tab3, tab4 = st.tabs(['❌','추천 자세히 보기', '과거 유사종목 비교', '기업분석 요약'])

        with tab2:
            # 청약 판단 여부 탭
            st.markdown('<span style="color: #043B72;font-size: 28px;">청약 판단 여부 </span>', unsafe_allow_html=True)

            # 점수 데이터
            total_score = 100
            score = df_pred['model_score'][i]
            remaining_score = total_score - score

            # 원 그래프 데이터
            labels = ['', '']
            sizes = [score, remaining_score]
            colors = ['#F58220', '#F6E3CE']
            wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

            # 원 그래프 생성
            fig, ax = plt.subplots()
            ax.pie(sizes, labels = labels, colors=colors, startangle=90,wedgeprops=wedgeprops)
            ax.axis('equal')  # 원형 파이차트로 설정

            # 그리드 레이아웃 생성
            cols = st.columns((4, 1, 3))

            # 첫 번째 컬럼에 원 그래프 배치
            with cols[0]:
                st.pyplot(fig)
                #st.subheader(f"{int(score_df['score'][i])}점")
                #st.markdown(f"<span style='font-size: 24px; text-align: right;'>{int(score_df['score'][i])}점</span>", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center;'>{int(df_pred['model_score'][i])}점", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: center; font-size: 18px;'>공모주 종합 평가 점수", unsafe_allow_html=True)

            # 두 번째 칼럼에 F, T, A 점수 배치
            with cols[1]:
                st.text('')
                st.header("F")
                st.caption("재무")
                st.text('')
                st.header("T")
                st.caption("시장 동향")
                st.text('')
                st.header("A")
                st.caption("기관")
                st.text('')

            with cols[2]:
                st.text('')
                st.text('')
                st.subheader('⭐'* f_star)
                st.text('')
                st.text('')
                st.text('')
                st.text('')
                st.text('')
                st.subheader('⭐' * t_star)
                st.text('')
                st.text('')
                st.text('')
                st.text('')
                st.subheader('⭐' * a_star)

            st.markdown(f"<h1 style='text-align: center; font-size: 20px;'>공모주 <span style='color: #F58220;'>{df_pred['기업명'][i]}</span>의 <u>확정 공모가 대비 수익률</u>은 약 <span style='color: #043B72;'>{percent}</span> 로 추정됩니다.</h1>", unsafe_allow_html=True)

            st.text('')
            st.text('')
            st.text('')

        # 과거 유사종목 비교 탭
        with tab3:
            st.caption('해당 기업과 F, T, A, 공모주 청약 판단 종합 점수가 가장 비슷한 기업을 확인하실 수 있습니다.')
            st.markdown('<span style="color: #043B72;font-size: 28px;">해당 점수대 수익률 </span>', unsafe_allow_html=True)
            st.markdown(
                f"<h1 style='text-align: left;'><span style='font-size: 25px;'>종합점수 : </span> <span style='color: #043B72 ; font-size: 30px;'>{int(df_pred['model_score'][i])}점</span></h1>",
                unsafe_allow_html=True)
            st.caption('최근 3년간 기업들의 점수 분포')

            ############### 산점도 그래프 생성 #######################
            # 데이터 프레임에서 필요한 부분만 선택 (최근 5년 : 400개, 최근 3년 : 250개)
            selected_df = df.iloc[:250, :]

            fig = px.scatter(
                selected_df, x='model_score', y='시초/공모(%)',
                trendline = 'ols',
                trendline_color_override="#043B72",
                color_discrete_sequence=['#F8ECE0']  # 점의 색상 설정
            )

            # x 축과 y 축의 이름 변경
            fig.update_xaxes(title_text='종합점수')
            fig.update_yaxes(title_text='시가 수익률')

            #수직선의 범위 : 점수 +- 3
            min_val = int(df_pred['model_score'][i]) - 3
            max_val = int(df_pred['model_score'][i]) + 3

            # 수직 선을 추가
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=min_val,  # 선의 시작점
                    x1=min_val,  # 선의 끝점
                    y0=min(df.iloc[:400,:]['시초/공모(%)']),  # y 축의 최소값
                    y1=max(df.iloc[:400,:]['시초/공모(%)']),  # y 축의 최대값
                    line=dict(color="#F58220", dash="dash")  # 선의 색상 및 스타일 설정
                )
            )

            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=max_val,  # 선의 시작점
                    x1=max_val,  # 선의 끝점
                    y0=min(selected_df['시초/공모(%)']),  # y 축의 최소값
                    y1=max(selected_df['시초/공모(%)']),  # y 축의 최대값
                    line=dict(color="#F58220", dash="dash")  # 선의 색상 및 스타일 설정
                )
            )

            bold_indices = (selected_df['model_score'] >= min_val) & (selected_df['model_score'] <= max_val)

            # 그래프 점 색상 설정
            fig.update_traces(marker=dict(color=['#F58220' if idx else '#F8ECE0' for idx in bold_indices]),
                              selector=dict(mode='markers'))

            # Plot
            st.plotly_chart(fig, use_container_width=True)

            #평균수익률 계산
            df_ranged = selected_df[(selected_df['model_score'] >= min_val) & (selected_df['model_score'] <= max_val)]
            mean_earn_rate = round(df_ranged['시초/공모(%)'].mean(),1)

            #중간수익률 계산
            median_earn_rate = round(df_ranged['시초/공모(%)'].median(),1)

            # 그리드 레이아웃 생성
            cols = st.columns((2, 2))
            with cols[0]:
                st.markdown(f"<h1 style='text-align: center;'><span style='font-size: 25px;'>평균 수익률</span> <span style='color: orange; font-size: 30px;'>{mean_earn_rate}%</span></h1>", unsafe_allow_html=True)

            with cols[1]:
                st.markdown(f"<h1 style='text-align: center;'><span style='font-size: 25px;'>중간 수익률</span> <span style='color: orange; font-size: 30px;'>{median_earn_rate}%</span></h1>", unsafe_allow_html=True)

            st.text('')
            st.text('')
            st.text('')
            st.text('')

            ##과거 유사종목 비교
            st.markdown('<span style="color: #043B72;font-size: 28px;">과거 유사종목 비교 </span>', unsafe_allow_html=True)


            # 유클리드 거리 계산 함수
            def compute_euclidean_similarity(target, comparison_group):
                distances = euclidean_distances(target, comparison_group)
                # 거리를 기준으로 오름차순으로 정렬하고 인덱스를 반환
                sorted_indices = np.argsort(distances[0])
                return sorted_indices

            comparison_group = df_done[["f_score", "t_score", "a_score", "model_score"]]
            target = df_pred.iloc[i, :][["f_score", "t_score", "a_score", "model_score"]].values.reshape(1, -1)

            # 각 학습 데이터와의 유클리드 거리 유사도 계산
            similarities = compute_euclidean_similarity(target, comparison_group)

            # 상위 K개 유사한 종목 선택 (예: 상위 2개 종목 선택)
            top_k = 2
            selected_indices = similarities[:top_k]

            # 최종 선택된 가장 유사한 기업
            selected_group = df_done.iloc[selected_indices, :]
            #selected_group.reset_index(drop=True, inplace=True)

            # 그리드 레이아웃 생성
            cols = st.columns((7, 1.5, 1.4, 3))

            #기업에 대한 정보
            with cols[0]:
                st.markdown(
                    f"""
                    <div style="border: 2px solid #e5e5e5; padding: 10px; text-align: left;">
                    <h2 style="font-size: 24px; color : #043B72">{selected_group['기업명'].iloc[0]}</h2>
                        <div style="border: 2px solid #FFFFFF; padding: 10px; background-color: #FAFAFA;">    
                            <p><strong>공모가 </strong> {selected_group['공모가(원)'].iloc[0]}원</p>
                            <p><strong>상장일 </strong> {selected_group['신규상장일'].iloc[0]}</p>
                            <p><strong>종합 점수 </strong> {int(selected_group['model_score'].iloc[0])}</p>
                            <p><strong style='font-weight: bold;'>시가 수익률 </strong><span style='color: #F58220;font-size : 22px'>{selected_group['시초/공모(%)'].iloc[0]}%</span></p> 
                        </div>
                    </div>
                    """,
            unsafe_allow_html=True
            )

            # 기업의 F, T, A 등급
            # 두 번째 칼럼에 F, T, A 점수 배치
            with cols[2]:
                st.subheader("F")
                st.caption("재무")
                st.subheader("T")
                st.caption("시장 동향")
                st.subheader("A")
                st.caption("기관")

            with cols[3]:
                # F, T, A의 별점 계산
                f_star = count_star(selected_group.iloc[0, selected_group.columns.get_loc("f_score")], f_quantiles)
                t_star = count_star(selected_group.iloc[0, selected_group.columns.get_loc("t_score")], t_quantiles)
                a_star = count_star(selected_group.iloc[0, selected_group.columns.get_loc("a_score")], a_quantiles)

                st.subheader('⭐' * f_star)
                st.text('')
                st.text('')
                st.text('')
                st.subheader('⭐' * t_star)
                st.text('')
                st.text('')
                st.subheader('⭐' * a_star)

            #두번째 유사 기업과 구분
            st.divider()

            # 그리드 레이아웃 생성
            cols = st.columns((7, 1.5, 1.4, 3))
            with cols[0]:
                st.markdown(
                f"""
                <div style="border: 2px solid #e5e5e5; padding: 10px; text-align: left;">
                <h2 style="font-size: 24px; color : #043B72">{selected_group['기업명'].iloc[1]}</h2>
                    <div style="border: 2px solid #FFFFFF; padding: 10px; background-color: #FAFAFA;">    
                        <p><strong>공모가 </strong> {selected_group['공모가(원)'].iloc[1]}원</p>
                        <p><strong>상장일 </strong> {selected_group['신규상장일'].iloc[1]}</p>
                        <p><strong>종합 점수 </strong> {int(selected_group['model_score'].iloc[1])}</p>
                        <p><strong style='font-weight: bold;'>시가 수익률 </strong><span style='color: #F58220;font-size : 22px'>{selected_group['시초/공모(%)'].iloc[1]}%</span></p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
                )

            # 두 번째 칼럼에 F, T, A 점수 배치
            with cols[2]:
                st.subheader("F")
                st.caption("재무")
                st.subheader("T")
                st.caption("시장 동향")
                st.subheader("A")
                st.caption("기관")

            with cols[3]:
                # F, T, A의 별점 계산
                f_star = count_star(selected_group.iloc[1, selected_group.columns.get_loc("f_score")], f_quantiles)
                t_star = count_star(selected_group.iloc[1, selected_group.columns.get_loc("t_score")], t_quantiles)
                a_star = count_star(selected_group.iloc[1, selected_group.columns.get_loc("a_score")], a_quantiles)

                st.subheader('⭐' * f_star)
                st.text('')
                st.text('')
                st.text('')
                st.subheader('⭐' * t_star)
                st.text('')
                st.text('')
                st.subheader('⭐' * a_star)


            st.text('')
            st.text('')
            st.text('')
            st.text('')

        ############################################################################
        # 기업 분석 요약 탭
        with tab4:
            st.markdown('<span style="color: #043B72;font-size: 28px;">기업분석 요약 </span>', unsafe_allow_html=True)
            st.caption('기업별 투자설명서를 요약한 정보입니다.')
            st.text('')

            #df의 i번째 행과 종목코드를 비교하여 comp에서 인덱싱
            comp_analysis_row = comp_analysis[comp_analysis['종목코드'] == df_pred['종목코드'][i]]

            #내용 펼치기 radio
            # 체크 박스 위젯에 고유한 키 부여
            pros_box = st.checkbox('강점 펼치기', key=f'pros_box_2{i}', help='투자설명서의 요약정보내 인수인의 의견의 내용을 분석')
            nega_box = st.checkbox('약점 펼치기', key=f'nega_box_2{i}', help='투자설명서의 요약정보내 투자위험요소의 내용을 분석')

            # 장점 제목(홀수 인덱스 : 19, 21, 23)
            st.markdown('<span style="color: #F58220;font-size: 25px;">강점 분석</span>', unsafe_allow_html=True)

            if pros_box:
                # 본문 인덱스 : 1, 3, 5/ 7, 9, 11 / 13, 15, 17
                start = 1
                for column in range(19, 24, 2):
                    title = comp_analysis_row.iloc[0, column]
                    if pd.isna(title) or title == "error":
                        st.markdown('<span style="color: #043B72;font-size: 18px;">아직 요약 정보가 업로드되지 않았어요 😥</span>',
                                    unsafe_allow_html=True)
                        break
                    else:
                        # 제목
                        st.markdown(f'<span style="color: #043B72; font-size: 20px;"><strong>{title}</strong></span>',
                                    unsafe_allow_html=True)
                        # 내용
                        for j in range(start, start + 5, 2):
                            content = comp_analysis_row.iloc[0, j]
                            if pd.notna(content):
                                st.markdown(f'<span style="color: #000000;font-size: 18px;">- {content}</span>',
                                            unsafe_allow_html=True)

                        st.divider()
                    start += 6

            st.divider()

            # 단점 제목(짝수 인덱스 : 20, 22, 24)
            st.markdown('<span style="color: #F58220;font-size: 25px;">약점 분석</span>', unsafe_allow_html=True)

            if nega_box:
                # 본문 인덱스 : 2, 4, 6/ 8, 10, 12 / 14, 16, 18
                start = 2
                for column in range(20, 25, 2):
                    title = comp_analysis_row.iloc[0, column]
                    if pd.isna(title) or title == "error":
                        st.markdown('<span style="color: #043B72;font-size: 18px;">아직 요약 정보가 업로드되지 않았어요 😥</span>',
                                    unsafe_allow_html=True)
                    else:
                        # 제목
                        st.markdown(f'<span style="color: #043B72; font-size: 20px;"><strong>{title}</strong></span>',
                                    unsafe_allow_html=True)
                        # 내용
                        for j in range(start, start + 5, 2):
                            content = comp_analysis_row.iloc[0, j]
                            if pd.notna(content):
                                st.markdown(f'<span style="color: #000000;font-size: 18px;">- {content}</span>',
                                            unsafe_allow_html=True)

                        st.divider()
                    start += 6

            st.divider()

if choose == "최근 상장한 기업 목록":
    st.text('')

    st.header('최근 상장한 기업 목록')
    st.caption('AI가 예측한 점수와 실제 수익률을 비교해보세요!')
    st.divider()
    st.text('')

    # 기업 정보를 담은 박스를 만듭니다.

    for i in range(0, 5):

        # 청약 권장도
        if df_done['model_score'][i] < 30:
            comment = '위험'
        elif df_done['model_score'][i] < 50:
            comment = '중립'
        elif df_done['model_score'][i] < 70:
            comment = '권장'
        else:
            comment = '추천'

        # 예상 수익률
        if df_done['범주'][i] == 0:
            percent = '~20%'
        elif df_done['범주'][i] == 1:
            percent = '20%~60%'
        elif df_done['범주'][i] == 2:
            percent = '60~100%'
        else:
            percent = '100%~'

        # F, T, A의 별점 계산
        f_star = count_star(df_done['f_score'][i], f_quantiles)
        t_star = count_star(df_done['t_score'][i], t_quantiles)
        a_star = count_star(df_done['a_score'][i], a_quantiles)

        # 그리드 레이아웃 생성
        cols = st.columns((3, 3))

        with cols[0]:

            st.markdown(
                f"""
                <div style="padding: 10px; text-align: left;">
                    <h2>{df_done['기업명'].values[i]}</h2>
                    <p><strong>공모가 </strong> {df_done['공모가(원)'].values[i]}원</p>
                    <p><strong>상장일 </strong> {df_done['신규상장일'].values[i]}</p>
                    <p><strong>주간사 </strong> {df_done['주간사'].values[i]}</p>
                    <p><strong>청약 권장도 </strong> {comment}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            #시가 수익률
            st.markdown(
                f"<h1 style='text-align: center;'><span style='font-size: 25px;'>시가 수익률</span> <span style='color: orange;'>{df_done['시초/공모(%)'][i]}%</span></h1>",
                unsafe_allow_html=True)
            #예측했던 수익률
            st.markdown(
                f"<h1 style='text-align: center;'><span style='font-size: 25px;'>예측 수익률</span> <span style='color: #043B72;'>{percent}</span></h1>",
                unsafe_allow_html=True)

        # 두번째 컬럼에 원 그래프 배치
        with cols[1]:
            # 점수 데이터
            total_score = 100
            score = df_done['model_score'][i]
            remaining_score = total_score - score

            # 원 그래프 데이터
            labels = ['', '']
            sizes = [score, remaining_score]
            colors = ['#F58220', '#F6E3CE']
            wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

            # 원 그래프 생성
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops=wedgeprops)
            ax.axis('equal')  # 원형 파이차트로 설정
            st.pyplot(fig)

            st.markdown(f"<h1 style='text-align: center;'>{int(df_done['model_score'][i])}점",
                        unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center; font-size: 18px;'>공모주 종합 평가 점수", unsafe_allow_html=True)

        with st.expander('기업분석 요약'):
            # df_done의 i번째 행과 종목코드를 비교하여 comp에서 인덱싱
            comp_analysis_row = comp_analysis[comp_analysis['종목코드'] == df_done['종목코드'][i]]

            # 체크 박스 위젯에 고유한 키 부여
            pros_box = st.checkbox('강점 펼치기', key=f'pros_box_2{i}')
            nega_box = st.checkbox('약점 펼치기', key=f'nega_box_2{i}')

            # 장점 제목(홀수 인덱스 : 19, 21, 23)
            st.markdown('<span style="color: #F58220;font-size: 25px;">강점 분석</span>', unsafe_allow_html=True)

            if pros_box:
                # 본문 인덱스 : 1, 3, 5/ 7, 9, 11 / 13, 15, 17
                start = 1
                for column in range(19, 24, 2):
                    title = comp_analysis_row.iloc[0, column]
                    if pd.isna(title) or title == "error":
                        st.markdown('<span style="color: #043B72;font-size: 18px;">아직 요약 정보가 업로드되지 않았어요 😥</span>',
                                    unsafe_allow_html=True)
                        break
                    else:
                        # 제목
                        st.markdown(f'<span style="color: #043B72; font-size: 20px;"><strong>{title}</strong></span>',
                                    unsafe_allow_html=True)
                        # 내용
                        for j in range(start, start + 5, 2):
                            content = comp_analysis_row.iloc[0, j]
                            if pd.notna(content):
                                st.markdown(f'<span style="color: #043B72;font-size: 18px;">{content}</span>',
                                            unsafe_allow_html=True)

                        st.divider()
                    start += 6

            st.divider()

            # 단점 제목(짝수 인덱스 : 20, 22, 24)
            st.markdown('<span style="color: #F58220;font-size: 25px;">약점 분석</span>', unsafe_allow_html=True)

            if nega_box:
                # 본문 인덱스 : 2, 4, 6/ 8, 10, 12 / 14, 16, 18
                start = 2
                for column in range(20, 25, 2):
                    title = comp_analysis_row.iloc[0, column]
                    if pd.isna(title) or title == "error":
                        st.markdown('<span style="color: #043B72;font-size: 18px;">아직 요약 정보가 업로드되지 않았어요 😥</span>',
                                    unsafe_allow_html=True)
                    else:
                        # 제목
                        st.markdown(f'<span style="color: #043B72; font-size: 20px;"><strong>{title}</strong></span>',
                                    unsafe_allow_html=True)
                        # 내용
                        for j in range(start, start + 5, 2):
                            content = comp_analysis_row.iloc[0, j]
                            if pd.notna(content):
                                st.markdown(f'<span style="color: #043B72;font-size: 18px;">{content}</span>',
                                            unsafe_allow_html=True)

                        st.divider()
                    start += 6

            st.divider()

        st.divider()








