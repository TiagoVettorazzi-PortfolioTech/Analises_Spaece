# Bibliotecas utilizadas
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#--------------------------------------------------------------------------------------------------------------------
# Leitura e filtros do arquivo Excel
xls = pd.ExcelFile('Planilhao_TCT_SPAECE_EF_2023_MT_240402-2 (1).xlsx')

df = pd.DataFrame(xls.parse('ESCOLA'))

df.columns = df.iloc[0]
df = df[1:]

# Remoção de Colunas 'CD_REDE','CD_ETAPA', 'CD_REGIONAL','CD_MUNICIPIO' 'CD_ESCOLA', manter apenas colunas com informações relevantes para análise de desempenho
df = df.drop(['CD_REDE','CD_ETAPA', 'CD_REGIONAL','CD_MUNICIPIO', 'CD_ESCOLA'], axis=1)

# Remover colunas completamente vazias (se existirem)
df = df.dropna(axis=1, how='all')

# Filtros para manter apenas os dados do 9º ano
df_9ano = df[df['DC_ETAPA'] == 'ENSINO FUNDAMENTAL DE 9 ANOS - 9º ANO'].reset_index(drop=True)

def selecionar_escola(df, nome_escola):
    """Filtra o DataFrame pela escola informada."""
    filtro = df_9ano['NM_ESCOLA'].str.contains(nome_escola, case=False, na=False)
    df_filtrado = df_9ano[filtro].copy()
    return df_filtrado

#--------------------------------------------------------------------------------------------------------------------
#Configuração de página
st.set_page_config(
    page_title="Análise por Escola",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📈 Painel Análises Spaece por Descritor")

# Filtro de escola
st.sidebar.header("🏫 Filtro de Escola")

lista_escolas = sorted(df_9ano["NM_ESCOLA"].dropna().unique())

# Define a escola padrão (primeira da lista)
escola_padrao = lista_escolas[0]

escola_selecionada = st.sidebar.selectbox(
    "Selecione a escola:",
    options=lista_escolas,
    index=0  # seleciona automaticamente a primeira da lista
)

df_filtrado = selecionar_escola(df_9ano, escola_selecionada)
st.write(f"### Escola selecionada: **{escola_selecionada}**")

#--------------------------------------------------------------------------------------------------------------------
# Cálculos de comparação
col1, col2 = st.columns(2)

# ---------------------------------------------------------------
# Geração do gráfico (usando o mesmo df_filtrado)
with col1:
    colunas_descritores = [col for col in df_filtrado.columns if col.startswith("D")]

    # ✅ Criação segura do DataFrame
    media_descritores = (
        df_filtrado[colunas_descritores]
        .apply(pd.to_numeric, errors='coerce')
        .mean()
    )

    desempenho_tfp = pd.DataFrame({
        "Descritor": media_descritores.index,
        "Desempenho": media_descritores.values
    })

    # Limpeza e formatação
    desempenho_tfp.dropna(subset=["Desempenho"], inplace=True)
    desempenho_tfp["Cor"] = desempenho_tfp["Desempenho"].apply(
        lambda x: "≤ 50%" if x <= 50 else "> 50%"
    )
    desempenho_tfp.sort_values(by="Descritor", ascending=True, inplace=True)

    # Criação do gráfico
    fig = px.bar(
        desempenho_tfp,
        y="Descritor",
        x="Desempenho",
        color="Cor",
        color_discrete_map={"≤ 50%": "red", "> 50%": "steelblue"},
        text=desempenho_tfp["Desempenho"].map(lambda v: f"{v:.1f}%"),
        title=f"Desempenho por Descritor - Escola Selecionada <br><sup>Barras vermelhas indicam ≤ 50%</sup>"
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=18,
        title_x=0.02,
        yaxis=dict(title="Descritor", tickangle=0),
        xaxis=dict(title="Desempenho (%)", range=[0, desempenho_tfp["Desempenho"].max() + 10]),
        legend_title_text="",
        margin=dict(l=40, r=20, t=100, b=80),
        height=1000
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=desempenho_tfp["Descritor"].tolist()
    )

    st.plotly_chart(fig,
                    use_container_width=True,
                    config={'displayModeBar': False}
                    )

#--------------------------------------------------------------------------------------------------------------------
# 🎯 Top 10 escolas (independente do filtro)
top10_geral_todos = df_9ano.sort_values(by="VL_D", ascending=False).head(10)

# Seleciona apenas colunas de descritores (as que começam com D)
colunas_descritores_top10 = [col for col in top10_geral_todos.columns if col.startswith("D")]

# Média por descritor entre as 10 melhores escolas
media_descritores_top10 = (
    top10_geral_todos[colunas_descritores_top10]
    .apply(pd.to_numeric, errors='coerce')
    .mean()
    .round(1)
)
# DataFrame para o gráfico
media_descritores_top10 = pd.DataFrame({
    "Descritor": media_descritores_top10.index,
    "Desempenho": media_descritores_top10.values
})

media_descritores_top10.sort_values(by="Descritor", ascending=True, inplace=True)
media_descritores_top10.dropna(inplace=True)
# Adiciona cores condicionais
media_descritores_top10["Cor"] = media_descritores_top10["Desempenho"].apply(
    lambda x: "≤ 50%" if x <= 50 else "> 50%"
)

with col2:
# Cria o gráfico de barras
    fig2 = px.bar(
        media_descritores_top10,
        y="Descritor",
        x="Desempenho",
        color="Cor",
        color_discrete_map={"≤ 50%": "red", "> 50%": "steelblue"},
        text=media_descritores_top10["Desempenho"].map(lambda v: f"{v:.1f}%"),
        title="Desempenho Médio por Descritor - Top 10 Escolas (9º Ano)<br><sup>"
    )

    fig2.update_traces(textposition="outside", cliponaxis=False)
    fig2.update_layout(
    showlegend=False,
    plot_bgcolor="#F0F2F6",
    paper_bgcolor="#F0F2F6",
    title_font_size=18,
    title_x=0.02,
    yaxis=dict(title="Descritor", tickangle=0),
    xaxis=dict(title="Desempenho (%)", range=[0, media_descritores_top10["Desempenho"].max() + 10]),
    margin=dict(l=40, r=20, t=100, b=80),
    height=1000
    )
    
    fig2.update_yaxes(
        categoryorder="array",
        categoryarray=media_descritores_top10["Descritor"].tolist()
    )

    st.plotly_chart(
        fig2, 
        use_container_width=True,
        config={'displayModeBar': False}
    )
