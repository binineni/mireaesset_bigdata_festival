#라이브러리 import
#필요한 경우 install
import streamlit as st
import streamlit_scrollable_textbox as stx
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as px
import plotly.graph_objs as go
from streamlit_option_menu import option_menu

# plotly 시각화 오류시 실행시킬 코드
#import plotly.offline as pyo
#import plotly.graph_objs as go
# 오프라인 모드로 변경하기
#pyo.init_notebook_mode()

#private 페이지를 위한 코드
st.set_page_config(page_title="FAQ")

##################################################################################################
@st.cache_data
def load_FTA_gain_data():

    FTA_gain = pd.read_csv("data/data_FTA_gain.csv")
    return FTA_gain
FTA_gain = load_FTA_gain_data()

@st.cache_data
def load_streamlit_data():
    df = pd.read_csv('data/data_streamlit_df.csv')
    df['시초/공모(%)'] = df['시초/공모(%)'].str.rstrip('%')
    df['시초/공모(%)'] = pd.to_numeric(df['시초/공모(%)'])
    df['예측일'] = pd.to_datetime(df['예측일']).dt.date
    df['신규상장일'] = pd.to_datetime(df['신규상장일']).dt.date
    df = df.sort_values(by=['예측일', '신규상장일'], ascending=[False, False])

    return df
df = load_streamlit_data()
###################################################################################################

#메뉴 탭 하단 사이드바에 이미지 넣기
with st.sidebar:
    choose = option_menu("FAQ", ["FTA등급과 종합점수가 궁금해요", "청약권장도란?", "점수대별 과거 종목의 수익률"],
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

    stx.scrollableTextbox('보통, 공모주를 청약하기 전 기관경쟁률, 의무보유확약과 같은 정보들을 이곳 저곳에서 확인하며 공모주를 처약하게 되는데요.'
                          '그래서 어떤 지표가 얼마나 높으면 좋은거야! 하는 고민들을 하신 경험, 한번쯤은 있지 않으셨나요?\n\n'
                          '저희 미래에셋은 공모주의 상장당일 수익률과 관련이 있는 여러 변수들을 AI알고리즘을 이용해 분석한 뒤 수익과 관련된 점수를 제공해 드려요.\n\n'
                          'F(finance)는 상장 대상 기업의 재무와 관련된 정보로, 기업의 성장성과 안전성, 수익성 등을 종합적으로 고려한 등급이에요.\n\n'
                          'T(trend)는 공모주 시장의 동향, 해당 업종 및 섹터의 최근 수익성, 상장 당시 투자자들의 심리 등을 종합적으로 고려한 등급이에요.\n\n'
                          'A(agent)는 수요예측결과를 바탕으로 기관경쟁률, 의무보유확약 등 공모주의 상태와 기관의 관심도 등을 종합적으로 고려한 등급이에요.\n\n'
                          'F,T,A 등급을 종합적으로 고려하여 산정한 0~100 사이의 점수가 공모주 종합 평가 지표 점수에요.\n'
                          '점수가 100에 가까워 질 수록 수익과 관련된 지표가 좋다는 의미로 해석됩니다.'
                          , height=300, border=True)

    st.caption('제공되는 점수는 참고용이므로, 투자의 책임은 본인에게 있습니다.')

elif choose == "청약권장도란?":
    pass

elif choose == "점수대별 과거 종목의 수익률":
    st.title('점수대별 수익률이 어땠을까?')
    st.divider()

    # 산점도 그래프 생성
    scatter = go.Scatter(
        x=df['model_score'],
        y=df['시초/공모(%)'],
        mode='markers',
        marker=dict(size=10, opacity=0.7),
        name='산점도'
    )

    # 회귀선 그래프 생성
    coefficients = np.polyfit(df['model_score'], df['시초/공모(%)'], 1)
    line = go.Scatter(
        x=df['model_score'],
        y=df['model_score'] * coefficients[0] + coefficients[1],
        mode='lines',
        line=dict(color='red'),
        name='회귀선'
    )

    # 그래프 레이아웃 설정
    layout = go.Layout(
        title='산점도와 회귀선',
        xaxis=dict(title='model_score'),
        yaxis=dict(title='시초/공모(%)')
    )

    # 그래프 출력
    fig = go.Figure(data=[scatter, line], layout=layout)
    st.plotly_chart(fig)


elif choose == "메뉴 이름3... 등등":
    st.write("메뉴 이름3... 등등에 대한 내용을 여기에 표시합니다.")




