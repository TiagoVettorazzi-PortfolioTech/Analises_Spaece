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

# Remoção de colunas irrelevantes
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
# Configuração de página
st.set_page_config(
    page_title="Análise por Escola - Eixos e Intervenção",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📊 Análises por Eixo e Roteiro de Intervenção")

# 🏫 Filtro de escola
st.sidebar.header("🏫 Filtro de Escola")

lista_escolas = sorted(df_9ano["NM_ESCOLA"].dropna().unique())
escola_padrao = lista_escolas[0]

escola_selecionada = st.sidebar.selectbox(
    "Selecione a escola:",
    options=lista_escolas,
    index=0
)

df_filtrado = selecionar_escola(df_9ano, escola_selecionada)
st.write(f"### Escola selecionada: **{escola_selecionada}**")

#--------------------------------------------------------------------------------------------------------------------
# 🎯 Definição dos eixos e mapeamento dos descritores
eixos_classificados = {
    'Números e Operações': [
        'D01_5EF', 'D06_5EF', 'D07', 'D08', 'D10', 'D11', 'D12', 'D13',
        'D15', 'D17', 'D18', 'D19', 'D21', 'D24', 'D25', 'D26', 'D27',
        'D30_SAEB', 'D32_SAEB', 'D35_SAEB'
    ],
    'Espaço e Forma': [
        'D45_5EF', 'D48', 'D49', 'D50', 'D51', 'D52', 'D06_SAEB', 'D07_SAEB'
    ],
    'Grandezas e Medidas': [
        'D59_5EF', 'D65', 'D67', 'D69', 'D05_SAEB'
    ],
    'Tratamento da Informação': [
        'D09_SAEB', 'D75', 'D77', 'D37_SAEB'
    ]
}

df_eixos_descritores = pd.DataFrame([
    {'EIXO': eixo, 'DESCRITOR': descritor}
    for eixo, descritores in eixos_classificados.items()
    for descritor in descritores
])

# Colunas de descritores (úteis em vários pontos)
colunas_descritores = [col for col in df_filtrado.columns if col.startswith("D")]

#--------------------------------------------------------------------------------------------------------------------
# 📊 Gráfico 1: Desempenho médio por eixo (TFP da escola)
vals = (
    df_filtrado[colunas_descritores]
    .apply(pd.to_numeric, errors='coerce')
    .mean()
)

vals = pd.DataFrame({
    "DESCRITOR": vals.index,
    "VL": vals.values
})

base = df_eixos_descritores.merge(vals, on='DESCRITOR', how='inner')
res = base.groupby('EIXO', as_index=False)['VL'].mean()
res.sort_values(by='VL', ascending=False, inplace=True)
res['VL'] = res['VL'].round(1)
order_eixos = res['EIXO'].tolist()

fig3 = px.bar(
    res,
    x="EIXO",
    y="VL",
    text=res["VL"].map(lambda v: f"{v:.1f}"),
    color="EIXO",
    color_discrete_sequence=["#4C78A8", "#4C78A8", "#4C78A8", "#4C78A8"],
    title="Desempenho Médio por Eixo – Escola Selecionada"
)

fig3.update_traces(textposition="outside", cliponaxis=False)
fig3.update_layout(
    plot_bgcolor="#F0F2F6",
    paper_bgcolor="#F0F2F6",
    title_font_size=18,
    title_x=0.02,
    xaxis=dict(title="Eixo", tickangle=0),
    yaxis=dict(title="Desempenho", range=[0, res["VL"].max() + 10]),
    showlegend=False,
    margin=dict(l=40, r=20, t=100, b=80),
    height=500
)

fig3.update_xaxes(
    categoryorder="array",
    categoryarray=order_eixos
)

st.plotly_chart(fig3, use_container_width=True)

#--------------------------------------------------------------------------------------------------------------------
# 📊 Gráfico 2: Comparativo por eixo — TFP x Municipal x Estadual

# 1) Colunas de descritores no df_9ano (garantia)
descr_cols = [c for c in df_9ano.columns if c.startswith("D")]

# 2) TFP (escola selecionada) por descritor
tfp_descr = (
    df_filtrado[descr_cols]
    .apply(pd.to_numeric, errors="coerce")
    .mean()
    .dropna()
)
df_tfp_descr = (
    tfp_descr
    .rename("TFP")
    .rename_axis("DESCRITOR")
    .reset_index()
)

# 3) Médias MUNICIPAIS por descritor
mun_descr = (
    df_9ano[df_9ano["DC_REDE"].str.upper().eq("MUNICIPAL")][descr_cols]
    .apply(pd.to_numeric, errors="coerce")
    .mean()
    .dropna()
    .rename("Municipal")
    .rename_axis("DESCRITOR")
    .reset_index()
)

# 4) Médias ESTADUAIS por descritor
est_descr = (
    df_9ano[df_9ano["DC_REDE"].str.upper().eq("ESTADUAL")][descr_cols]
    .apply(pd.to_numeric, errors="coerce")
    .mean()
    .dropna()
    .rename("Estadual")
    .rename_axis("DESCRITOR")
    .reset_index()
)

# 5) Junta com mapa de eixos e agrega por eixo
base_comp = (
    df_eixos_descritores
    .merge(df_tfp_descr, on="DESCRITOR", how="inner")
    .merge(mun_descr,      on="DESCRITOR", how="left")
    .merge(est_descr,      on="DESCRITOR", how="left")
)

res_eixos = (
    base_comp
    .groupby("EIXO", as_index=False)
    .agg({"TFP": "mean", "Municipal": "mean", "Estadual": "mean"})
)

res_eixos[["TFP", "Municipal", "Estadual"]] = res_eixos[["TFP", "Municipal", "Estadual"]].round(1)
res_eixos['EIXO'] = pd.Categorical(res_eixos['EIXO'], categories=order_eixos, ordered=True)

res_melt = res_eixos.melt(
    id_vars="EIXO",
    value_vars=["TFP", "Municipal", "Estadual"],
    var_name="Grupo",
    value_name="VL_D"
).sort_values("EIXO")

res_melt["VL_D"] = res_melt["VL_D"].round(1)
res_melt["Grupo"] = res_melt["Grupo"].replace({
    "TFP": "Escola selecionada",
    "Municipal": "Rede Municipal",
    "Estadual": "Rede Estadual"
})

color_map_grupos = {
    "Escola selecionada": "#4C78A8",  
    "Rede Municipal": "#E45756",      
    "Rede Estadual": "#72B7B2"        
}

fig_comp_eixos = px.bar(
    res_melt,
    x="EIXO",
    y="VL_D",
    color="Grupo",
    barmode="group",
    text=res_melt["VL_D"].map(lambda v: f"{v:.1f}"),
    color_discrete_map=color_map_grupos,
    title="Escola Selecionada× Municipal × Estadual — Desempenho Médio por Eixo"
)

fig_comp_eixos.update_traces(textposition="outside", cliponaxis=False)

fig_comp_eixos.update_layout(
    plot_bgcolor="#F0F2F6",
    paper_bgcolor="#F0F2F6",
    title_font_size=18,
    title_x=0.02,
    xaxis=dict(title="Eixo", tickangle=0),
    yaxis=dict(title="Desempenho", range=[0, 100]),
    legend_title_text="",
    margin=dict(l=40, r=20, t=100, b=80),
)

fig_comp_eixos.update_xaxes(
    categoryorder="array",
    categoryarray=order_eixos
)

st.plotly_chart(
    fig_comp_eixos,
    use_container_width=True,
    config={'displayModeBar': False}
)

#--------------------------------------------------------------------------------------------------------------------
# 📋 Tabela: Roteiro de intervenção por eixo

# Top 10 geral (para base de comparação)
top10_geral_todos = df_9ano.sort_values(by="VL_D", ascending=False).head(10)

# Desempenho TFP por descritor (completo)
desempenho_tfp_completo = df_filtrado[colunas_descritores].T
desempenho_tfp_completo.columns = ["TFP - (%)"]
desempenho_tfp_completo = desempenho_tfp_completo.apply(pd.to_numeric, errors='coerce')
desempenho_tfp_completo.dropna(inplace=True)

# Média Top 10 por descritor
media_top10_completo = (
    top10_geral_todos[colunas_descritores]
    .apply(pd.to_numeric, errors='coerce')
    .mean()
    .dropna()
    .to_frame(name="Top 10 Geral - Média (%)")
)

comparativo_completo = desempenho_tfp_completo.join(media_top10_completo, how='inner')

# Adiciona eixo
comparativo_completo_com_eixo = comparativo_completo.copy()
comparativo_completo_com_eixo["Eixo"] = comparativo_completo_com_eixo.index.map(
    lambda d: next((eixo for eixo, descritores in eixos_classificados.items() if d in descritores), None)
)

comparativo_completo_com_eixo = comparativo_completo_com_eixo.dropna(subset=["Eixo"])

# Médias por eixo (Escola Selecionada x Top 10 Geral)
medias_por_eixo = comparativo_completo_com_eixo.groupby("Eixo")[
    ["TFP - (%)", "Top 10 Geral - Média (%)"]
].mean().round(1)

# Construção do roteiro de intervenção
roteiro_intervencao = []

for eixo, row in medias_por_eixo.iterrows():
    tfp = row["TFP - (%)"]
    top10 = row["Top 10 Geral - Média (%)"]
    lacuna = top10 - tfp

    if lacuna > 60:
        prioridade = "🔴 Alta"
    elif lacuna > 40:
        prioridade = "🟠 Média"
    else:
        prioridade = "🟢 Baixa"

    roteiro_intervencao.append({
        "Eixo": eixo,
        "Média Top 10 (%)": top10.round(1),
        "Lacuna (%)": lacuna.round(1),
        "Prioridade de Intervenção": prioridade,
        "Sugestão": f"Revisar conteúdos de {eixo.lower()}, com foco nos descritores mais críticos. Propor atividades diagnósticas, planos de aula específicos e grupos de reforço."
    })

df_roteiro_intervencao = pd.DataFrame(roteiro_intervencao)
df_roteiro_intervencao["Escola"] = escola_selecionada

df_roteiro_intervencao = df_roteiro_intervencao[[
    "Eixo",
    # "Média Top 10 (%)",
    # "Lacuna (%)",
    "Prioridade de Intervenção",
    "Sugestão"
]].reset_index(drop=True)

df_exibicao = df_roteiro_intervencao.copy()
df_exibicao.index = df_exibicao.index + 1
df_exibicao.index.name = ''

st.subheader("📝 Roteiro de Intervenção Pedagógica por Eixo")
st.table(df_exibicao)
