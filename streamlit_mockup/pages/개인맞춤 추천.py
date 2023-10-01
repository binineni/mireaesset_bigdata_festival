#라이브러리 import
#필요한 경우 install

import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#plotly 시각화 오류시 실행시킬 코드
#import plotly.offline as pyo
#import plotly.graph_objs as go
# 오프라인 모드로 변경하기
#pyo.init_notebook_mode()

#public 페이지를 위한 코드
st.set_page_config(page_title="개인맞춤 추천")

image = Image.open('streamlit_mockup/img/미래에셋로고.png')
image2 = Image.open('streamlit_mockup/img/네이버클라우드.png')
image3 = Image.open('streamlit_mockup/img/미래에솦.png')

st.sidebar.image(image, use_column_width=True)
st.sidebar.image(image2, use_column_width=True)
st.sidebar.image(image3, use_column_width=True)

##업로드 하고자 하는 데이터 버젼
version = 2

##################################################################################################
@st.cache_data
def load_streamlit_data():
    df = pd.read_csv(f'streamlit_mockup/data/data_streamlit_df_ver{version}.csv')
    df['시초/공모(%)'] = df['시초/공모(%)'].str.rstrip('%')
    df['시초/공모(%)'] = pd.to_numeric(df['시초/공모(%)'])
    df['예측일'] = pd.to_datetime(df['예측일']).dt.date
    df['신규상장일'] = pd.to_datetime(df['신규상장일']).dt.date
    df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
    df = df.sort_values(by=['예측일', '신규상장일'], ascending=[False, False])

    return df

df = load_streamlit_data()

@st.cache_data
def load_cs_data():
    cs_data = pd.read_csv('streamlit_mockup/data/cs_data_streamlit.csv')
    cs_data['종목코드'] = cs_data['종목코드'].astype(str).str.zfill(6)

    return cs_data

cs = load_cs_data()

@st.cache_data
def load_sector_data():
    sector_data = pd.read_csv('streamlit_mockup/data/data_전처리_model_final.csv')
    sector_data['종목코드'] = sector_data['종목코드'].astype(str).str.zfill(6)
    sector_data = sector_data[['종목코드','종목명','지수명','종목구분']]
    return sector_data

sector = load_sector_data()

#df에 종목구분, 지수명 추가
df = pd.concat([df, sector[['지수명','종목구분']]], axis=1)

######################################################################################################

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

#청약예정 데이터프레임
today = datetime.date(2023, 9, 15)
#df_pred : 예측일 ~ 상장일 사이에 있는 추천할 기업
df_pred = df[(df['예측일'] <= today) & (df['신규상장일'] >= today)]
df_pred.reset_index(drop=True,inplace=True)

#제목
st.title('개인맞춤 공모주 추천', help='개인이 직접 로그인하여 들어오는 서비스가 아니기 때문에, 직접 고객 ID를 입력하는 형식의 테스트 버전입니다.')
st.caption('개인이 직접 로그인하여 들어오는 서비스가 아니기 때문에, 직접 고객 ID를 입력하는 형식의 테스트 버전입니다.')
st.divider()

#고객 ID에 해당하는 정보 보여주기
ID = st.text_input(label = '고객 ID를 입력하세요', help='1 ~ 800,000 사이의 숫자')

#추천할 기업이 없으면 fail, 아니면 success로 구분하여 페이지 구분을 다르게 하기 위함
status = ''

if ID:
    if not ID.isdigit():
        st.warning('고객 ID는 숫자여야 합니다.')
    else:
        st.info(f"{ID}번 고객님을 위한 청약 예정 추천 공모주입니다.")

        # 해당 고객의 '지수명1'과 '지수명2' 값을 가져옵니다.
        customer_row = cs[cs['고객ID'] == int(ID)]

        if not customer_row.empty:
            index1_value = customer_row['지수명1'].values[0]
            index2_value = customer_row['지수명2'].values[0]

            # '지수명1'과 '지수명2'의 값을 사용하여 매칭된 행을 찾습니다.
            matching_row = df_pred[(df_pred['지수명'].isin([index1_value, index2_value])) & (df_pred['model_score'] >= 45)]
            matching_row.reset_index(drop=True, inplace=True)

            #추천할 공모주 화면에 출력
            if not matching_row.empty:
                # 지수명2가 '-'이 아닌 경우에만 출력
                if customer_row['지수명2'].values[0] != '-':
                    st.write(
                        f"고객님의 관심 업종인 **{customer_row['지수명1'].values[0]}**와 **{customer_row['지수명2'].values[0]}**에 대한 {len(matching_row)}개의 공모주를 권장드려요.")

                else:
                    st.write(f"고객님의 관심 업종인 **{customer_row['지수명1'].values[0]}**에 대한 {len(matching_row)}개의 공모주가 청약 예정이에요.")
                #추천할 공모주가 있을 경우
                status = 'suc'

            else:
                if customer_row['지수명2'].values[0] != '-':
                    st.write(
                        f"고객님의 관심 업종인 **{customer_row['지수명1'].values[0]}**와 **{customer_row['지수명2'].values[0]}**에 대한 공모주가 아직 청약 전이에요.")
                else:
                    st.write(f"고객님의 관심 업종인 **{customer_row['지수명1'].values[0]}**에 대한 공모주가 아직 청약 전이에요.")
                #추천할 공모주가 없을 경우
                status = 'fail'
        else:
            st.warning(f"{ID}번 고객의 정보를 찾을 수 없습니다.")
else:
    st.warning('고객 ID를 입력하세요.')

# #별점 계산 함수 ver2
# def count_star(df, score, i):
#     quantile_30 = df[score].quantile(q=0.3)
#     quantile_50 = df[score].quantile(q=0.5)
#     quantile_70 = df[score].quantile(q=0.7)
#     quantile_90 = df[score].quantile(q=0.9)
#
#     star = np.select(
#         [df[score][i] <= quantile_30, df[score][i] <= quantile_50, df[score][i] <= quantile_70, df[score][i] <= quantile_90],
#         [1, 2, 3, 4],
#         default=5
#     )
#
#     return int(star)


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

# 기업 정보를 담은 박스를 만듭니다.
if status == 'suc':
    st.caption('조금 더 자세한 정보는 **추천공모주** 페이지를 참고하세요!')
    for i in range(0, len(matching_row)):
        # 청약권장도 박스 색
        color_list = ['#FF5733', '#FFD700', '#00FF00 ', '#009933']
        # 청약 권장도
        if matching_row.loc[i, 'model_score'] < 25:
            comment = '위험'
            color = color_list[0]
        elif matching_row.loc[i, 'model_score'] < 45:
            comment = '중립'
            color = color_list[1]
        elif matching_row.loc[i, 'model_score'] < 70:
            comment = '권장'
            color = color_list[2]
        else:
            comment = '추천'
            color = color_list[3]

        # 데이터프레임에 주간사 정보 처리 결과 적용
        matching_row['주간사'] = matching_row['주간사'].apply(process_agency)

        st.markdown(
            f"""
            <div style="border: 2px solid #e5e5e5; padding: 10px; text-align: left;">
                <h2 style='color: #043B72;'>{matching_row['기업명'].values[i]}</h2>
                <p style='font-size: 20px;'><strong>청약 예정일 </strong> {matching_row['공모주일정'].values[i]}</p>
                <div style="display: flex; justify-content: space-between;">
                    <div style="border: 2px solid #FFFFFF; padding: 3px; background-color: {color};">
                    <p style='font-size: 20px; font-weight : bold; color : #FFFFFF; margin-bottom: 0;'>청약{comment}</p>
                    </div>
                </div>
                <div style="border: 2px solid #FFFFFF; padding: 10px; background-color: #FAFAFA;">    
                    <p style='font-size: 20px;'><strong>공모 예정가 </strong> {matching_row['공모희망가(원)'].values[i]}원</p>
                    <p style='font-size: 20px;'><strong>상장 예정일 </strong> {matching_row['신규상장일'].values[i]}</p>
                    <p style='font-size: 20px;'><strong>주간사 </strong> {matching_row['주간사'].values[i]}</p> 
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 예상 수익률
        if matching_row['범주'][i] == 0:
            percent = '~20%'
        elif matching_row['범주'][i] == 1:
            percent = '20%~60%'
        elif matching_row['범주'][i] == 2:
            percent = '60~100%'
        else:
            percent = '100%~'

        # F, T, A의 별점 계산, df에서 matching_row 기업들의 인덱스를 찾아야함
        f_star = count_star(matching_row['f_score'][i], f_quantiles)
        t_star = count_star(matching_row['t_score'][i], t_quantiles)
        a_star = count_star(matching_row['a_score'][i], a_quantiles)

        # 점수 데이터
        total_score = 100
        score = matching_row['model_score'][i]
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

        # 그리드 레이아웃 생성
        cols = st.columns((4, 1, 3))

        # 첫 번째 컬럼에 원 그래프 배치
        with cols[0]:
            st.pyplot(fig)
            # st.subheader(f"{int(score_df['score'][i])}점")
            # st.markdown(f"<span style='font-size: 24px; text-align: right;'>{int(score_df['score'][i])}점</span>", unsafe_allow_html=True)
            st.markdown(f"<h1 style='text-align: center;'>{int(matching_row['model_score'][i])}점", unsafe_allow_html=True)
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
            st.subheader('⭐' * f_star)
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

        st.markdown(
            f"<h1 style='text-align: center; font-size: 20px;'>공모주 <span style='color: #F58220;'>{matching_row['기업명'][i]}</span>의 <u>확정 공모가 대비 수익률</u>은 약 <span style='color: #043B72;'>{percent}</span> 로 추정됩니다.</h1>",
            unsafe_allow_html=True)

        st.text('')
        st.text('')
        st.text('')
        st.divider()

if status == 'fail':
    st.markdown(
        f'<span style="color: #043B72;font-size: 18px;">이런 종목은 어떠신가요?', unsafe_allow_html=True)
    best = df_pred.sort_values(by='model_score', ascending=False).head(1)

    # best의 청약권장도 구하기
    color_list = ['#FF5733', '#FFD700', '#01DF01', '#009933']
    # 청약 권장도
    if float(best['model_score']) < 25:
        comment = '위험'
        color = color_list[0]
    elif float(best['model_score']) < 45:
        comment = '중립'
        color = color_list[1]
    elif float(best['model_score']) < 70:
        comment = '권장'
        color = color_list[2]
    else:
        comment = '추천'
        color = color_list[3]

    #기업 정보를 담은 네모박스
    st.markdown(
        f"""
                <div style="border: 2px solid #e5e5e5; padding: 10px; text-align: left;">
                    <h2 style='color: #043B72;'>{best['기업명'].values[0]}</h2>
                    <p style='font-size: 20px;'><strong>청약 예정일 </strong> {best['공모주일정'].values[0]}</p>
                    <div style="display: flex; justify-content: space-between;">
                        <div style="border: 2px solid #FFFFFF; padding: 3px; background-color: {color};">
                        <p style='font-size: 20px; font-weight : bold; color : #FFFFFF; margin-bottom: 0;'>청약{comment}</p>
                        </div>
                    </div>
                    <div style="border: 2px solid #FFFFFF; padding: 10px; background-color: #FAFAFA;">    
                        <p style='font-size: 20px;'><strong>공모 예정가 </strong> {best['공모희망가(원)'].values[0]}원</p>
                        <p style='font-size: 20px;'><strong>상장 예정일 </strong> {best['신규상장일'].values[0]}</p>
                        <p style='font-size: 20px;'><strong>주간사 </strong> {best['주간사'].values[0]}</p> 
                    </div>
                </div>
                """,
        unsafe_allow_html=True
    )

    # 예상 수익률
    if best['범주'].values[0] == 0:
        percent = '~20%'
    elif best['범주'].values[0] == 1:
        percent = '20%~60%'
    elif best['범주'].values[0] == 2:
        percent = '60~100%'
    else:
        percent = '100%~'

    # F, T, A의 별점 계산, df에서 matching_row 기업들의 인덱스를 찾아야함
    f_star = count_star(best['f_score'].values[0], f_quantiles)
    t_star = count_star(best['t_score'].values[0], t_quantiles)
    a_star = count_star(best['a_score'].values[0], a_quantiles)

    # 점수 데이터
    total_score = 100
    score = best['model_score'].values[0]
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

    # 그리드 레이아웃 생성
    cols = st.columns((4, 1, 3))

    # 첫 번째 컬럼에 원 그래프 배치
    with cols[0]:
        st.pyplot(fig)
        # st.subheader(f"{int(score_df['score'][i])}점")
        # st.markdown(f"<span style='font-size: 24px; text-align: right;'>{int(score_df['score'][i])}점</span>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{int(best['model_score'].values[0])}점", unsafe_allow_html=True)
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
        st.subheader('⭐' * f_star)
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

    st.markdown(
        f"<h1 style='text-align: center; font-size: 20px;'>공모주 <span style='color: #F58220;'>{best['기업명'].values[0]}</span>의 <u>확정 공모가 대비 수익률</u>은 약 <span style='color: #043B72;'>{percent}</span> 로 추정됩니다.</h1>",
        unsafe_allow_html=True)

    st.text('')
    st.text('')
    st.text('')
    st.divider()

