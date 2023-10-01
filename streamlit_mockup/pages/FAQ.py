#라이브러리 import
#필요한 경우 install
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os


# plotly 시각화 오류시 실행시킬 코드
#import plotly.offline as pyo
#import plotly.graph_objs as go
# 오프라인 모드로 변경하기
#pyo.init_notebook_mode()

current_file = os.path.abspath(__file__)  # 현재 스크립트 파일의 절대 경로를 얻습니다.
current_directory = os.path.dirname(current_file)  # 현재 스크립트 파일이 위치한 디렉토리 경로를 얻습니다.
#private 페이지를 위한 코드
st.set_page_config(page_title="FAQ")

#메뉴 탭 하단 사이드바에 이미지 넣기
with st.sidebar:
    choose = option_menu("FAQ", ["FTA등급과 종합점수가 궁금해요","F, T, A에 사용한 데이터가 궁금해요", "청약권장도란?", ],
                         icons=['메뉴 아이콘1', '메뉴 아이콘2', '메뉴 아이콘3'],
                         menu_icon="bi bi-question-circle", default_index=0,
                         styles={
                         # default_index = 처음에 보여줄 페이지 인덱스 번호
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    } # css 설정
    )

##업로드 하고자 하는 데이터 버젼
version = 2

##################################################################################################
@st.cache_data
def load_FTA_gain_data():

    FTA_gain = pd.read_csv("streamlit_mockup/data/data_FTA_gain.csv")
    return FTA_gain
FTA_gain = load_FTA_gain_data()

@st.cache_data
def load_streamlit_data():
    df = pd.read_csv(f'streamlit_mockup/data/data_streamlit_df_ver{version}.csv')
    df['시초/공모(%)'] = df['시초/공모(%)'].str.rstrip('%')
    df['시초/공모(%)'] = pd.to_numeric(df['시초/공모(%)'])
    df['예측일'] = pd.to_datetime(df['예측일']).dt.date
    df['신규상장일'] = pd.to_datetime(df['신규상장일']).dt.date
    df = df.sort_values(by=['예측일', '신규상장일'], ascending=[False, False])

    return df
df = load_streamlit_data()
###################################################################################################



# 선택된 메뉴에 따라 문구 출력
if choose == "FTA등급과 종합점수가 궁금해요":
    st.title('F, T, A 등급과 점수가 궁금해요!')
    st.divider()

    # 그래프에 대한 설명
    st.caption('당사 모델이 산정한 F, T, A의 중요도')

    # 원 그래프 데이터
    labels = ['F', 'T', 'A']
    score = [FTA_gain['상대가중치'][0], FTA_gain['상대가중치'][1], FTA_gain['상대가중치'][2]]
    colors = ['#FE9A2E', '#FE642E', '#FFFF00']
    wedgeprops = {'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}

    # 원 그래프 생성
    fig, ax = plt.subplots()
    ax.pie(score, labels=labels, colors=colors, autopct='%.1f%%',startangle=90, wedgeprops=wedgeprops)
    ax.axis('equal')  # 원형 파이차트로 설정
    st.pyplot(fig)


    # 그래프 아래 내용
    st.header('어떤 공모주를 청약하면 좋을까?')

    # stx.scrollableTextbox('보통, 공모주를 청약하기 전 기관경쟁률, 의무보유확약과 같은 정보들을 이곳 저곳에서 확인하며 공모주를 처약하게 되는데요.'
    #                       '그래서 어떤 지표가 얼마나 높으면 좋은거야! 하는 고민들을 하신 경험, 한번쯤은 있지 않으셨나요?\n\n'
    #                       '저희 미래에셋은 공모주의 상장당일 수익률과 관련이 있는 여러 변수들을 AI알고리즘을 이용해 분석한 뒤 수익과 관련된 점수를 제공해 드려요.\n\n'
    #                       'F(finance)는 상장 대상 기업의 재무와 관련된 정보로, 기업의 성장성과 안전성, 수익성 등을 종합적으로 고려한 등급이에요.\n\n'
    #                       'T(trend)는 공모주 시장의 동향, 해당 업종 및 섹터의 최근 수익성, 상장 당시 투자자들의 심리 등을 종합적으로 고려한 등급이에요.\n\n'
    #                       'A(agent)는 수요예측결과를 바탕으로 기관경쟁률, 의무보유확약 등 공모주의 상태와 기관의 관심도 등을 종합적으로 고려한 등급이에요.\n\n'
    #                       'F,T,A 등급을 종합적으로 고려하여 산정한 0~100 사이의 점수가 공모주 종합 평가 지표 점수에요.\n'
    #                       '점수가 100에 가까워 질 수록 수익과 관련된 지표가 좋다는 의미로 해석됩니다.'
    #                       , height=300, border=True)

    st.markdown(f'<span style="color: #000000;font-size: 18px;">보통, 공모주를 청약하기 전 기관경쟁률, 의무보유확약과 같은 정보들을 이곳 저곳에서 확인하며 공모주를 처약하게 되는데요, 그래서 어떤 지표가 얼마나 높으면 좋은거야! 하는 고민들을 하신 경험, 한번쯤은 있지 않으셨나요?'
                ,unsafe_allow_html=True)
    st.markdown(f'<span style="color: #000000;font-size: 18px;">저희 미래에셋은 공모주의 상장당일 수익률과 관련이 있는 여러 변수들을 AI알고리즘을 이용해 분석한 뒤 수익과 관련된 점수를 제공해 드려요.'
              , unsafe_allow_html=True)
    st.divider()
    st.markdown(
        '<span style="font-size: 18px;"><strong><span style="color: #043B72;">F</span>(finance)</strong></span>는 상장 대상 기업의 재무와 관련된 정보로, 기업의 성장성과 안전성, 수익성 등을 종합적으로 고려한 등급이에요.',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span style="font-size: 18px;"><strong><span style="color: #043B72;">T</span>(trend)</strong></span>는 공모주 시장의 동향, 해당 업종 및 섹터의 최근 수익성, 상장 당시 투자자들의 심리 등을 종합적으로 고려한 등급이에요.',
        unsafe_allow_html=True
    )
    st.markdown(
        '<span style="font-size: 18px;"><strong><span style="color: #043B72;">A</span>(agent)</strong></span>는 수요예측결과를 바탕으로 기관경쟁률, 의무보유확약 등 공모주의 상태와 기관의 관심도 등을 종합적으로 고려한 등급이에요.',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown(f'<span style="color: #000000;font-size: 18px;">F,T,A 등급을 종합적으로 고려하여 산정한 0~100 사이의 점수가 공모주 종합 평가 지표 점수에요.'
                ,unsafe_allow_html=True)
    st.markdown(f'<span style="color: #000000;font-size: 18px;">점수가 100에 가까워 질 수록 수익과 관련된 지표가 좋다는 의미로 해석됩니다.'
                ,unsafe_allow_html=True)

    st.caption('제공되는 점수는 참고용이므로, 투자의 책임은 본인에게 있습니다.')

elif choose == "F, T, A에 사용한 데이터가 궁금해요":
    st.title("F, T, A에 사용한 데이터가 궁금해요")
    st.divider()

    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">저희 모델은 재무정보를 나타내는 F, 시장의 동향 정보를 나타내는 T, 수요예측 결과 바탕의 A 에 대한 데이터를 이용했습니다.'
        , unsafe_allow_html=True)
    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">각 부문별로 세부적인 데이터는 아래와 같습니다.'
        , unsafe_allow_html=True)

    st.text('')

    cols = st.columns((2,2,2))
    with cols[0]:
        #F
        st.header('F(finance)', divider = 'grey')
        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- EPS증감율, 매출액증감율 등'
            , unsafe_allow_html=True)
        st.caption('기업의 성장성을 나타내는 지표들')

        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 자산비율, 유동비율, PSR, 예상 시가총액 등'
            , unsafe_allow_html=True)
        st.caption('기업의 안정성을 나타내는 지표들')

        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- ROE, ROA 등'
            , unsafe_allow_html=True)
        st.caption('기업의 수익성을 나타내는 지표들')

    with cols[1]:
        #T
        st.header('T(trend)', divider='grey')
        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 업황'
            , unsafe_allow_html=True)
        st.caption('KOSPI, KOSDAQ 업종별 지수의 증감률을 이용해 섹터의 관심도를 파악하는 데이터')

        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 공모주 시장동향'
            , unsafe_allow_html=True)
        st.caption('이전 분기 상장한 공모주들의 수익 확률을 기반으로, 최근 공모주 시장의 동향을 파악하는 데이터')

        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 뉴스심리지수'
            , unsafe_allow_html=True)

    with cols[2]:
        #A
        st.header('A(agent)', divider='grey')
        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 기관 수요 예측 경쟁률'
            , unsafe_allow_html=True)
        st.caption('기관투자자들의 전반적인 관심도를 나타내는 지표')
        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">- 의무보유확약'
            , unsafe_allow_html=True)
        st.caption('주식을 팔지않고 일정 기간동안 가지고 있는 비율')
        st.markdown(
            f'<span style="color: #000000;font-size: 18px;">-유통가능물량(비율)'
            , unsafe_allow_html=True)
        st.caption('주식이 상장한 날 바로 매도할 수 있는 물량')

elif choose == "청약권장도란?":
    st.title('청약권장도란?')
    st.divider()

    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">청약권장도는 모델이 예측한 종합점수를 바탕으로 네개의 구간으로 나눈 지표에요.'
        , unsafe_allow_html=True)
    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">권장도를 나누는 기준은 다음과 같습니다.'
        , unsafe_allow_html=True)

    # 그리드 레이아웃 생성
    cols = st.columns((2, 2, 2, 2))

    # Define score ranges and corresponding colors
    score_ranges = [(0, 25), (25, 45), (45, 70), (70, 100)]
    color_list = ['#FF5733', '#FFD700', '#01DF01', '#009933']
    comment_list = ["청약위험", "청약중립", "청약권장", "청약추천"]

    # 청약권장도별 점수
    for i, (min_val, max_val) in enumerate(score_ranges):
        with cols[i]:
            color = color_list[i]
            comment = comment_list[i]

            # 점수 범위
            st.markdown(f"<h1 style='text-align: center; font-size: 18px;'>{min_val} - {max_val}점", unsafe_allow_html=True)

            # 네모박스
            st.markdown(
                f"<div style='border: 2px solid #FFFFFF; padding: 3px; background-color: {color};'>"
                f"<p style='font-size: 20px; font-weight : bold;text-align : center; color : #FFFFFF; margin-bottom: 0;'>{comment}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
    st.text('')
    st.text('')
    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">종합점수가 높을 수록, 수익이 날 확률과 수익률이 높아지는 것을 아래 그래프를 통해 확인할 수 있습니다. 회귀선의 설명력은 약 80%로, 이 값이 높을수록 회귀선이 수익률을 잘 예측한다는 것을 나타냅니다.'
        , unsafe_allow_html=True)

    st.markdown(
        f'<span style="color: #000000;font-size: 18px;">이는 저희가 개발한 종합점수와 시가 수익률이 양의 상관관계가 존재한다는 걸 잘 나타내주는 지표입니다. 구간별 수익률에 대한 정보를 아래에서 확인해보세요.'
        , unsafe_allow_html=True)

    st.divider()

    #그래프 표시
    selected_df = df.iloc[:400, :]

    # Create a scatter plot with Plotly Express
    fig = px.scatter(
        df, x='model_score', y='시초/공모(%)',
        trendline='ols',
        trendline_color_override="#043B72",
        color_discrete_sequence=['#F8ECE0'],  # Default color for all points
    )

    # Customize the OLS trendline appearance
    fig.data[0].update(
        line=dict(color='black', width=2),  # Change trendline color and width
        marker=dict(size=10),  # Adjust marker size for trendline points
    )

    for i, (min_val, max_val) in enumerate(score_ranges):
        bold_indices = (df['model_score'] >= min_val) & (df['model_score'] < max_val)

        # Add scatter trace with different colors for each score range
        fig.add_trace(go.Scatter(
            x=df[bold_indices]['model_score'],
            y=df[bold_indices]['시초/공모(%)'],
            mode='markers',
            name=comment_list[i],
            marker=dict(
                color=color_list[i],
                size=6  # Adjust marker size for points within the range
            )
        ))

    # Customize the figure layout
    fig.update_xaxes(title_text='종합점수')
    fig.update_yaxes(title_text='시가 수익률')
    fig.update_layout(
        title="청약권장도별 수익률 그래프",
    )

    # Add vertical lines to demarcate score ranges
    for min_val, max_val in score_ranges[1:]:
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=min_val,
                x1=min_val,
                y0=min(df['시초/공모(%)']),
                y1=max(df['시초/공모(%)']),
                line=dict(color="#000000", dash="dash")
            )
        )

    # R^2 값을 텍스트로 표시
    fig.add_annotation(
        go.layout.Annotation(
            x=0.05,  # 표시 위치 조정
            y=245,  # 표시 위치 조정
            text=f'R2 = 0.8',
            showarrow=False,
            font=dict(size=22, color="#043B72")
        )
    )

    # Plot
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('구간별 수익률 정보')
    cols = st.columns((2, 2, 2, 2, 2))
    # 청약권장도별 점수

    with cols[0]:
        st.header('')
        st.header('')
        st.header('')
        st.markdown(
            f"<h1 style='text-align: center;'><span style='font-size: 20px;'>수익률 평균</span></h1>",
            unsafe_allow_html=True)
        st.markdown(
            f"<h1 style='text-align: center;'><span style='font-size: 20px;'>수익률 중앙값</span></h1>",
            unsafe_allow_html=True)
        st.markdown(
            f"<h1 style='text-align: center;'><span style='font-size: 24px;'>20% ⬆</span></h1>",
            unsafe_allow_html=True)
    st.caption('20%⬆ : 수익률이 20%이상일 확률')


    for i, (min_val, max_val) in enumerate(score_ranges):
        with cols[i+1]:
            color = color_list[i]
            comment = comment_list[i]

            # 점수 범위
            st.markdown(f"<h1 style='text-align: center; font-size: 18px;'>{min_val} - {max_val}점",
                        unsafe_allow_html=True)

            # 네모박스
            st.markdown(
                f"<div style='border: 2px solid #FFFFFF; padding: 3px; background-color: {color};'>"
                f"<p style='font-size: 20px; font-weight : bold;text-align : center; color : #FFFFFF; margin-bottom: 0;'>{comment}</p>"
                f"</div>",
                unsafe_allow_html=True
            )

            # 평균수익률
            df_ranged = selected_df[(selected_df['model_score'] >= min_val) & (selected_df['model_score'] <= max_val)]
            mean_earn_rate = round(df_ranged['시초/공모(%)'].mean(), 1)
            st.markdown(
                f"<h1 style='text-align: center;'><span style='color: #043B72; font-size: 30px;'>{mean_earn_rate}%</span></h1>",
                unsafe_allow_html=True)

            # 중간수익률 계산
            median_earn_rate = round(df_ranged['시초/공모(%)'].median(), 1)
            st.markdown(
                f"<h1 style='text-align: center;'><span style='color: #043B72; font-size: 30px;'>{median_earn_rate}%</span></h1>",
                unsafe_allow_html=True)

            # 최소 수익확률 계산
            earning_rate = round( len(df_ranged[df_ranged['시초/공모(%)'] > 20]) / len(df_ranged) * 100)
            st.markdown(
                f"<h1 style='text-align: center;'><span style='color: #043B72; font-size: 30px;'>{earning_rate}%</span></h1>",
                unsafe_allow_html=True)











