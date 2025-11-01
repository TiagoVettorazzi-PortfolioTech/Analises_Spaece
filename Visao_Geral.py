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


#--------------------------------------------------------------------------------------------------------------------

def grafico_barras_vertical(
    df, x_col, y_col, titulo, bar_color, fmt="{:.2f}", x_max=None, max_label_len=40
):
    # 🔹 Copia o DataFrame para não alterar o original
    df_plot = df.copy()

    # 🔹 Trunca os rótulos longos do eixo Y
    def truncar_label(label, max_len=max_label_len):
        label = str(label)
        return label if len(label) <= max_len else label[:max_len - 3] + "..."

    df_plot[y_col] = df_plot[y_col].apply(truncar_label)

    # 🔹 Cria o gráfico
    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        orientation="h",
        text=df_plot[x_col].map(lambda v: fmt.format(v)),
        title=titulo,
        color_discrete_sequence=[bar_color]
    )

    # 🔹 Ajustes de exibição
    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11)
    )

    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=16,
        title_x=0.02,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            title="",
            range=[0, x_max] if x_max else None
        ),
        yaxis=dict(
            title="",
            automargin=True,  # ✅ Ajusta automaticamente para caber os labels truncados
        ),
        showlegend=False,
        margin=dict(l=200, r=100, t=70, b=20),
        height=500
    )

    fig.update_yaxes(autorange="reversed")
    return fig


# === 2️⃣ Gráfico comparativo com hue (legenda no topo) ===
def grafico_barras_vertical_2(
    df, x_col, y_col, titulo, fmt="{:.2f}", hue_col=None, palette=None, x_max=None
):
    import plotly.express as px

    color_map = None
    color_sequence = None
    if isinstance(palette, dict):
        color_map = palette
    elif isinstance(palette, (list, tuple)):
        color_sequence = list(palette)

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        color=hue_col,
        color_discrete_map=color_map,
        color_discrete_sequence=color_sequence,
        text=df[x_col].map(lambda v: fmt.format(v)),
        title=titulo
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11)
    )

    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=16,
        title_x=0.02,
        showlegend=False,  # <<< remove a legenda completamente
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            title="",
            range=[0, x_max] if x_max else None
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=120, r=30, t=120, b=20),
        height=500
    )

    fig.update_yaxes(autorange="reversed")
    return fig


# Top 10 de todas as escolas do estado do Ceará
top_10_geral = (
    df_9ano[["NM_MUNICIPIO", "NM_ESCOLA", "VL_D"]]
    .dropna()
    .sort_values(by="VL_D", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Adicionar coluna de ranking
top_10_geral.index += 1
top_10_geral.rename_axis("Ranking", inplace=True)


top_10_geral_grafico = grafico_barras_vertical (
    df = top_10_geral,
    x_col = "VL_D",
    y_col = "NM_ESCOLA",
    titulo = "Top 10 Escolas - 9º Ano (VL_D)",
    bar_color = "steelblue"
)

# Filtrar apenas escolas estaduais do 9º ano
df_9ano_estadual = df_9ano[df_9ano["DC_REDE"].str.upper() == "ESTADUAL"]

top_10_escolas_9ano_estadual = (
    df_9ano_estadual[["NM_MUNICIPIO", "NM_ESCOLA", "VL_D"]]
    .dropna()
    .sort_values(by="VL_D", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Adicionar coluna de ranking
top_10_escolas_9ano_estadual.index += 1
top_10_escolas_9ano_estadual.rename_axis("Ranking", inplace=True)

top10_estadual_grafico = grafico_barras_vertical (
    df = top_10_escolas_9ano_estadual,
    x_col = "VL_D",
    y_col = "NM_ESCOLA",
    titulo = "Top 10 Escolas Estaduais - 9º Ano (VL_D)",
    bar_color = "steelblue"
)

# Top 10 das escolas Municipais

# Filtrar apenas escolas municipais do 9º ano
df_9ano_municipal = df_9ano[df_9ano["DC_REDE"].str.upper() == "MUNICIPAL"]

# Selecionar colunas solicitadas e gerar ranking
top_10_escolas_9ano_municipal = (
    df_9ano_municipal[["NM_MUNICIPIO", "NM_ESCOLA", "VL_D"]]
    .dropna()
    .sort_values(by="VL_D", ascending=False)
    .head(30)
    .reset_index(drop=True)
)

# Adicionar coluna de ranking
top_10_escolas_9ano_municipal.index += 1
top_10_escolas_9ano_municipal.rename_axis("Ranking", inplace=True)

top10_municipal_grafico = grafico_barras_vertical (
    df = top_10_escolas_9ano_municipal,
    x_col = "VL_D",
    y_col = "NM_ESCOLA",
    titulo = "Top 10 Escolas Municipais - 9º Ano (VL_D)",
    bar_color = "steelblue"
)

# Top 10 das escolas de Fortaleza

# Filtrar apenas escolas de Fortaleza no 9º ano
df_9ano_fortaleza = df_9ano[df_9ano["NM_MUNICIPIO"].str.upper() == "FORTALEZA"]

# Selecionar colunas solicitadas e gerar ranking
top_10_escolas_9ano_fortaleza = (
    df_9ano_fortaleza[["DC_REDE", "NM_ESCOLA", "VL_D"]]
    .dropna()
    .sort_values(by="VL_D", ascending=False)
    .head(10)
    .reset_index(drop=True)
)

# Adicionar coluna de ranking
top_10_escolas_9ano_fortaleza.index += 1
top_10_escolas_9ano_fortaleza.rename_axis("Ranking", inplace=True)

paleta_cores = {
        "ESTADUAL": "steelblue",
        "MUNICIPAL": "orange"}

top10_fortaleza_grafico = grafico_barras_vertical_2 (
    df = top_10_escolas_9ano_fortaleza,
    x_col = "VL_D",
    y_col = "NM_ESCOLA",
    titulo = "Top 10 Escolas de Fortaleza - 9º Ano (Comparativo Estadual x Municipal)",
    hue_col = "DC_REDE",
    palette = paleta_cores
)
# top10_fortaleza_grafico = grafico_barras_vertical(
#     df = top_10_escolas_9ano_fortaleza,
#     x_col = "VL_D", 
#     y_col = "NM_ESCOLA",
#     titulo = "Top 10 Escolas de Fortaleza - 9º Ano (VL_D)",
#     bar_color = "steelblue"            
# )


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

#--------------------------------------------------------------------------------------------------------------------

# Configuração da página
st.set_page_config(page_title="Painel Análises Spaece", layout="wide")

st.title("📈 Painel Análises Spaece")
st.write("###  Análise das escola de acordo com o Valor de Desempenho (VL_D)")

# with st.sidebar:
#     # st.header("Filtros")

#--------------------------------------------------------------------------------------------------------------------
col1, col2 = st.columns(2)
    
with col1:
    st.plotly_chart(top_10_geral_grafico)

with col2:
    st.plotly_chart(top10_estadual_grafico)

with col1:
    st.plotly_chart(top10_municipal_grafico)

with col2:
    st.plotly_chart(top10_fortaleza_grafico)





























# # 2) Leitura com cache para não reler o arquivo a cada interação
# # @st.cache_data
# def carregar_dados(caminho: str, aba: str = "ESCOLA") -> pd.DataFrame:
#     xls = pd.ExcelFile(caminho)
#     df = xls.parse(aba)
#     # renomeia a coluna que representa o bairro (ajuste se for outra)
#     if "Unnamed: 3" in df.columns:
#         df = df.rename(columns={"Unnamed: 3": "bairro"})
#     # normaliza strings (remove espaços extras)
#     if "bairro" in df.columns:
#         df["bairro"] = df["bairro"].astype(str).str.strip()
#     return df

# df = carregar_dados("Planilhao_TCT_SPAECE_EF_2023_MT_240402-2 (1).xlsx", "ESCOLA")

# # 3) Filtro na barra lateral
# opcoes_bairro = sorted(df["bairro"].dropna().unique())
# bairro_selecionado = st.sidebar.selectbox("Selecione um bairro:", ["(Todos)"] + opcoes_bairro)

# # 4) Aplica o filtro
# if bairro_selecionado != "(Todos)":
#     df_filtrado = df[df["bairro"] == bairro_selecionado]
# else:
#     df_filtrado = df

# # 5) Exibição
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("Dados filtrados")
#     st.dataframe(df_filtrado, use_container_width=True)

# with col2:
#     st.subheader("Resumo")
#     st.write("Total de linhas:", len(df_filtrado))
#     st.write("Bairros distintos:", df_filtrado["bairro"].nunique() if "bairro" in df_filtrado else 0)

