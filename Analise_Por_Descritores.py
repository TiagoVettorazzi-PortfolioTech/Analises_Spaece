import streamlit as st
import pandas as pd
import plotly.express as px

#--------------------------------------------------------------------------------------------------------------------
# 📘 Leitura e tratamento do arquivo Excel
xls = pd.ExcelFile('Planilhao_TCT_SPAECE_EF_2023_MT_240402-2 (1).xlsx')
df = pd.DataFrame(xls.parse('ESCOLA'))

# Define a primeira linha como cabeçalho real
df.columns = df.iloc[0]
df = df[1:]

# Remove colunas irrelevantes
df = df.drop(['CD_REDE','CD_ETAPA', 'CD_REGIONAL','CD_MUNICIPIO', 'CD_ESCOLA'], axis=1, errors='ignore')

# Remove colunas completamente vazias
df = df.dropna(axis=1, how='all')

# Filtra apenas 9º ano
df_9ano = df[df['DC_ETAPA'] == 'ENSINO FUNDAMENTAL DE 9 ANOS - 9º ANO'].reset_index(drop=True)

#--------------------------------------------------------------------------------------------------------------------
# ⚙️ Configuração da página Streamlit
st.set_page_config(
    page_title="Análise por Escola",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Análise de Desempenho por Escola - SPAECE 2023")

#--------------------------------------------------------------------------------------------------------------------
# 🏫 Filtro de escola
st.sidebar.header("Filtro de Escola")

lista_escolas = sorted(df_9ano["NM_ESCOLA"].dropna().unique())

escola_escolhida = st.sidebar.selectbox(
    "Selecione a escola:",
    options=["Todas as escolas"] + lista_escolas,
    index=0
)

#--------------------------------------------------------------------------------------------------------------------
# 📊 Filtragem dinâmica
def selecionar_escola(df, nome_escola):
    filtro = df['NM_ESCOLA'].str.contains(nome_escola, case=False, na=False)
    df_filtrado = df[filtro].copy()
    return df_filtrado

if escola_escolhida == "Todas as escolas":
    df_filtrado = df_9ano.copy()
else:
    df_filtrado = selecionar_escola(df_9ano, escola_escolhida)
    st.write(f"### Escola selecionada: **{escola_escolhida}**")

#--------------------------------------------------------------------------------------------------------------------
# 📈 Geração do gráfico de desempenho por descritor (modelo Plotly)
if escola_escolhida != "Todas as escolas" and not df_filtrado.empty:
    # Seleciona colunas de descritores (as que começam com D)
    colunas_descritores = [col for col in df_filtrado.columns if str(col).startswith("D")]

    # Transforma em formato long para o gráfico
    desempenho_escola = (
        df_filtrado[colunas_descritores]
        .apply(pd.to_numeric, errors='coerce')
        .mean()
        .round(1)
    )

    desempenho_escola = pd.DataFrame({
        "Descritor": desempenho_escola.index,
        "Desempenho": desempenho_escola.values
    })

    # Adiciona cor condicional
    desempenho_escola["Cor"] = desempenho_escola["Desempenho"].apply(
        lambda x: "≤ 50%" if x <= 50 else "> 50%"
    )

    # Cria gráfico Plotly Express
    fig = px.bar(
        desempenho_escola,
        x="Descritor",
        y="Desempenho",
        color="Cor",
        color_discrete_map={"≤ 50%": "red", "> 50%": "steelblue"},
        text=desempenho_escola["Desempenho"].map(lambda v: f"{v:.1f}%"),
        title=f"Desempenho por Descritor - {escola_escolhida} (9º Ano)<br><sup>Barras vermelhas indicam ≤ 50%</sup>"
    )

    # Ajustes visuais
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=18,
        title_x=0.02,
        xaxis=dict(title="Descritor", tickangle=90),
        yaxis=dict(title="Desempenho (%)", range=[0, desempenho_escola["Desempenho"].max() + 10]),
        legend_title_text="",
        margin=dict(l=40, r=20, t=100, b=80),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Selecione uma escola específica para visualizar o gráfico.")
