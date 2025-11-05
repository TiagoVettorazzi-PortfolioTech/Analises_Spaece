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

escola_selecionada = st.sidebar.selectbox(
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

if escola_selecionada == "Todas as escolas":
    df_filtrado = df_9ano.copy()
else:
    df_filtrado = selecionar_escola(df_9ano, escola_selecionada)
    st.write(f"### Escola selecionada: **{escola_selecionada}**")

#--------------------------------------------------------------------------------------------------------------------
# 📈 Geração do gráfico de desempenho por descritor (modelo Plotly)
if escola_selecionada != "Todas as escolas" and not df_filtrado.empty:
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
        title=f"Desempenho por Descritor - {escola_selecionada} (9º Ano)<br><sup>Barras vermelhas indicam ≤ 50%</sup>"
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

#--------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------




colunas_descritores = [col for col in df.columns if isinstance(col, str) and col.startswith("D")]

# Garantir que as colunas estão numéricas no top10_geral
top10_geral_todos = df_9ano.sort_values(by="VL_D", ascending=False).head(10)

# Substituir traços por NaN
top10_geral_todos[colunas_descritores] = top10_geral_todos[colunas_descritores].replace("-", pd.NA)
for col in colunas_descritores:
    top10_geral_todos[col] = pd.to_numeric(top10_geral_todos[col], errors='coerce')

# Reconstruir o DataFrame comparativo_completo com todos os descritores (com sufixos)
desempenho_tfp_completo = escola_selecionada[colunas_descritores].T
desempenho_tfp_completo.columns = ["TFP - (%)"]
desempenho_tfp_completo = desempenho_tfp_completo.apply(pd.to_numeric, errors='coerce')
desempenho_tfp_completo.dropna(inplace=True)

# Garantir a existência da média dos 10 melhores com sufixo
media_top10_completo = top10_geral_todos[colunas_descritores].mean().dropna().to_frame(name="Top 10 Geral - Média (%)")

# Recriar o comparativo
comparativo_completo = desempenho_tfp_completo.join(media_top10_completo, how='inner')

# Definição de eixos
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

# Converte em DataFrame para visualização
df_eixos_descritores = pd.DataFrame([
    {'EIXO': eixo, 'DESCRITOR': descritor}
    for eixo, descritores in eixos_classificados.items()
    for descritor in descritores
])
