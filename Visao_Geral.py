# Bibliotecas utilizadas
import streamlit as st
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- leitura e base (igual ao seu) ---
xls = pd.ExcelFile('Planilhao_TCT_SPAECE_EF_2023_MT_240402-2 (1).xlsx')
df = pd.DataFrame(xls.parse('ESCOLA'))
df.columns = df.iloc[0]; df = df[1:]
df = df.drop(['CD_REDE','CD_ETAPA', 'CD_REGIONAL','CD_MUNICIPIO', 'CD_ESCOLA'], axis=1)
df = df.dropna(axis=1, how='all')
df_9ano = df[df['DC_ETAPA'] == 'ENSINO FUNDAMENTAL DE 9 ANOS - 9º ANO'].reset_index(drop=True)

# ---------------------- seletor de escola (mesma lógica da outra página) ----------------------
st.set_page_config(page_title="Painel Análises Spaece", layout="wide")
st.title("📈 Painel Análises Spaece por Valor de Desempenho (VL_D)")

lista_escolas = sorted(df_9ano["NM_ESCOLA"].dropna().unique())
escola_selecionada = st.sidebar.selectbox("🏫 Selecione a escola:", options=lista_escolas, index=0)

st.write(f"### Escola Selecionada: {escola_selecionada}")
# st.write("### Análise das escolas de acordo com o Valor de Desempenho (VL_D)")

def nome_eq(a, b):
    return str(a).strip().casefold() == str(b).strip().casefold()

# ---------------------- helpers de gráfico (mantive suas funções) ----------------------
def grafico_barras_vertical(
    df, x_col, y_col, titulo, fmt="{:.2f}", hue_col=None, palette=None,
    x_max=None, max_label_len=40
):
    import plotly.express as px
    df_plot = df.copy()

    # 1) Ordena pelo valor (maior primeiro)
    df_plot = df_plot.sort_values(by=x_col, ascending=False).reset_index(drop=True)

    # 2) Cria rótulo abreviado SEM sobrescrever a categoria real
    def truncar_label(label, max_len=max_label_len):
        s = str(label)
        return s if len(s) <= max_len else s[:max_len-3] + "..."
    df_plot["label_escola"] = df_plot[y_col].apply(truncar_label)

    # 3) Cores
    color_map, color_sequence = (
        palette if isinstance(palette, dict) else None
    ), (
        list(palette) if isinstance(palette, (list, tuple)) else None
    )

    # 4) Agora: X = categoria (nome completo), Y = valor
    fig = px.bar(
        df_plot,
        x=y_col,                 # categoria real = nome completo
        y=x_col,                 # valor numérico
        # orientation padrão = 'v' (vertical)
        color=hue_col,
        color_discrete_map=color_map,
        color_discrete_sequence=color_sequence,
        text=df_plot[x_col].map(lambda v: fmt.format(v)),
        title=titulo,
        hover_data={y_col: True}  # mostra o nome completo no hover
    )

    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))

    # 5) Layout para barras verticais
    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=16,
        title_x=0.02,
        showlegend=False,
        xaxis=dict(
            title="",
            showgrid=False,
            tickangle=0,          # ajuda a caber rótulos
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False,
            range=[0, x_max] if x_max else None  # usa x_max como limite superior do Y
        ),
        margin=dict(l=60, r=40, t=70, b=140),     # mais espaço embaixo pros rótulos
        height=520,
    )

    # 6) Ordem e rótulos abreviados no eixo X (categoria)
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=df_plot[y_col].tolist(),    # mantém a ordem do DF
        ticktext=df_plot["label_escola"],         # rótulo abreviado exibido
        tickvals=df_plot[y_col],                  # categoria real (nome completo)
    )

    return fig



def grafico_barras_vertical_2(
    df, x_col, y_col, titulo, fmt="{:.2f}", hue_col=None, palette=None,
    x_max=None, max_label_len=40
):
    import plotly.express as px
    df_plot = df.copy()

    # 1) Ordena o DF que VAI pro gráfico (maior VL_D em cima)
    df_plot = df_plot.sort_values(by=x_col, ascending=False).reset_index(drop=True)

    # 2) Cria um rótulo abreviado, mas NÃO mexe na coluna original
    def truncar_label(label, max_len=max_label_len):
        s = str(label)
        return s if len(s) <= max_len else s[:max_len-3] + "..."
    df_plot["label_escola"] = df_plot[y_col].apply(truncar_label)

    # 3) Configuração de cores
    color_map, color_sequence = (
        palette if isinstance(palette, dict) else None
    ), (
        list(palette) if isinstance(palette, (list, tuple)) else None
    )

    # 4) Repara que o y continua sendo o NOME COMPLETO
    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,                 # <-- categoria real = nome completo
        orientation="h",
        color=hue_col,
        color_discrete_map=color_map,
        color_discrete_sequence=color_sequence,
        text=df_plot[x_col].map(lambda v: fmt.format(v)),
        title=titulo,
        hover_data={y_col: True}  # mostra o nome completo no hover
    )

    fig.update_traces(textposition="outside", cliponaxis=False, textfont=dict(size=11))
    fig.update_layout(
        plot_bgcolor="#F0F2F6",
        paper_bgcolor="#F0F2F6",
        title_font_size=16,
        title_x=0.02,
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            title="",
            range=[0, x_max] if x_max else None
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=260, r=100, t=70, b=20),  # pode ajustar o "l" se precisar
        height=700,
    )

    # 5) Ordem + rótulo abreviado no eixo Y
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=df_plot[y_col].tolist(),      # ordem baseada no nome completo
        autorange="reversed",
        ticktext=df_plot["label_escola"],           # o que aparece visualmente
        tickvals=df_plot[y_col]                     # ainda referenciando o nome completo
    )

    return fig


# ---------------------- helper para injetar a escola selecionada e colorir ----------------------
def preparar_topN_com_escola(
    df_topN: pd.DataFrame,
    df_todos: pd.DataFrame,
    escola_sel: str,
    n: int = 30
) -> pd.DataFrame:
    """
    - df_topN: já é o Top N do recorte (colunas: NM_MUNICIPIO, NM_ESCOLA, VL_D).
    - df_todos: base completa (df_9ano) de onde SEMPRE busco a escola selecionada.
    - escola_sel: texto do selectbox.
    - Retorna o DF + escola (se existir) ORDENADO por VL_D desc.
    """
    def eq(a, b): return str(a).strip().casefold() == str(b).strip().casefold()

    out = df_topN[['NM_MUNICIPIO','NM_ESCOLA','VL_D']].dropna().copy()
    out['__cor'] = 'Outras'

    # pega linha da escola na base completa
    sel = df_todos[df_todos['NM_ESCOLA'].apply(lambda x: eq(x, escola_sel))][
        ['NM_MUNICIPIO','NM_ESCOLA','VL_D']
    ].dropna().copy()

    if sel.empty:
        # não achou a escola na base completa -> só retorna o topN ordenado
        out = out.sort_values('VL_D', ascending=False).reset_index(drop=True)
        return out

    # se já está no topN, apenas marca
    mask = out['NM_ESCOLA'].apply(lambda x: eq(x, escola_sel))
    if mask.any():
        out.loc[mask, '__cor'] = 'Selecionada'
    else:
        sel['__cor'] = 'Selecionada'
        out = pd.concat([out, sel], ignore_index=True)

    # ORDENAR o DF a ser usado no GRÁFICO por VL_D desc (exatamente como você pediu)
    out = out.sort_values('VL_D', ascending=False).reset_index(drop=True)

    # opcional: limitar no máximo a N+1 linhas (TopN + Selecionada)
    if len(out) > n + 1:
        # remove do fim quem não é selecionada
        excesso = len(out) - (n + 1)
        idx = out[out['__cor'] != 'Selecionada'].tail(excesso).index
        out = out.drop(index=idx).reset_index(drop=True)

    return out

PALETA = {'Selecionada': 'red', 'Outras': 'steelblue'}

# Top 10 Geral
top_10_geral = (
    df_9ano[['NM_MUNICIPIO','NM_ESCOLA','VL_D']].dropna()
    .sort_values(by='VL_D', ascending=False).head(30).reset_index(drop=True)
)
top_10_geral_plot = preparar_topN_com_escola(top_10_geral, df_9ano, escola_selecionada, n=30)
top_10_geral_grafico = grafico_barras_vertical_2(
    df=top_10_geral_plot, x_col='VL_D', y_col='NM_ESCOLA',
    titulo='Top 30 Escolas - 9º Ano (VL_D)',
    hue_col='__cor', palette=PALETA
)

# Top 10 Estadual
df_9ano_estadual = df_9ano[df_9ano['DC_REDE'].str.upper() == 'ESTADUAL']
top_10_estadual = (
    df_9ano_estadual[['NM_MUNICIPIO','NM_ESCOLA','VL_D']].dropna()
    .sort_values(by='VL_D', ascending=False).head(30).reset_index(drop=True)
)
top_10_estadual_plot = preparar_topN_com_escola(top_10_estadual, df_9ano, escola_selecionada, n=30)
top10_estadual_grafico = grafico_barras_vertical_2(
    df=top_10_estadual_plot, x_col='VL_D', y_col='NM_ESCOLA',
    titulo='Top 30 Escolas Estaduais - 9º Ano (VL_D)',
    hue_col='__cor', palette=PALETA
)

# Top 10 Municipal
df_9ano_municipal = df_9ano[df_9ano['DC_REDE'].str.upper() == 'MUNICIPAL']
top_10_municipal = (
    df_9ano_municipal[['NM_MUNICIPIO','NM_ESCOLA','VL_D']].dropna()
    .sort_values(by='VL_D', ascending=False).head(30).reset_index(drop=True)
)
top_10_municipal_plot = preparar_topN_com_escola(top_10_municipal, df_9ano, escola_selecionada, n=30)
top10_municipal_grafico = grafico_barras_vertical_2(
    df=top_10_municipal_plot, x_col='VL_D', y_col='NM_ESCOLA',
    titulo='Top 30 Escolas Municipais - 9º Ano (VL_D)',
    hue_col='__cor', palette=PALETA
)

# Top 30 Fortaleza
df_9ano_fortaleza = df_9ano[df_9ano['NM_MUNICIPIO'].str.upper() == 'FORTALEZA']
top_10_fortaleza = (
    df_9ano_fortaleza[['NM_MUNICIPIO','NM_ESCOLA','VL_D']].dropna()
    .sort_values(by='VL_D', ascending=False).head(30).reset_index(drop=True)
)
top_10_fortaleza_plot = preparar_topN_com_escola(top_10_fortaleza, df_9ano, escola_selecionada, n=30)
top10_fortaleza_grafico = grafico_barras_vertical_2(
    df=top_10_fortaleza_plot, x_col='VL_D', y_col='NM_ESCOLA',
    titulo='Top 30 Escolas de Fortaleza - 9º Ano (VL_D)',
    hue_col='__cor', palette=PALETA
)

# ---------------------- layout ----------------------

# ====== MÉTRICAS GERAIS ======
media_geral      = float(df_9ano["VL_D"].mean().round(2))
media_municipal  = float(df_9ano[df_9ano["DC_REDE"].str.upper() == "MUNICIPAL"]["VL_D"].mean().round(2))
media_estadual   = float(df_9ano[df_9ano["DC_REDE"].str.upper() == "ESTADUAL"]["VL_D"].mean().round(2))
media_fortaleza  = float(df_9ano[df_9ano["NM_MUNICIPIO"].str.upper() == "FORTALEZA"]["VL_D"].mean().round(2))
escolas_analisadas = int(df_9ano["NM_ESCOLA"].nunique())

# ====== BASE PARA AS MÉTRICAS ======
mask_sel = df_9ano["NM_ESCOLA"].apply(lambda x: nome_eq(x, escola_selecionada))
df_sel   = df_9ano.loc[mask_sel].copy()
if df_sel.empty:
    st.warning("Escola selecionada não encontrada na base do 9º ano.")
    st.stop()

# VL_D da escola (média se houver múltiplas linhas)
vl_escola     = float(df_sel["VL_D"].mean().round(2))
rede_sel      = str(df_sel["DC_REDE"].iloc[0])
municipio_sel = str(df_sel["NM_MUNICIPIO"].iloc[0]).title()

# Ranking geral (1 = melhor)
df_rank_geral = df_9ano.sort_values("VL_D", ascending=False).reset_index(drop=True)
df_rank_geral["Ranking"] = df_rank_geral.index + 1
rk_geral = int(
    df_rank_geral.loc[
        df_rank_geral["NM_ESCOLA"].apply(lambda x: nome_eq(x, escola_selecionada)),
        "Ranking"
    ].min()
)

# Médias globais solicitadas
media_geral     = float(df_9ano["VL_D"].mean().round(2))
media_estadual  = float(df_9ano[df_9ano["DC_REDE"].str.upper() == "ESTADUAL"]["VL_D"].mean().round(2))
media_municipal = float(df_9ano[df_9ano["DC_REDE"].str.upper() == "MUNICIPAL"]["VL_D"].mean().round(2))
media_fortaleza = float(df_9ano[df_9ano["NM_MUNICIPIO"].str.upper() == "FORTALEZA"]["VL_D"].mean().round(2))

# Helper: delta mostra quanto a ESCOLA está acima/abaixo da média (positivo = escola melhor)
def delta_vs_escola(media):
    return f"{(vl_escola - float(media)):+.1f}"

# ====== LINHA 1: Escola, Rede, Município, Ranking (sem deltas) ======
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="Escola Selecionada (VL_D)", value=f"{vl_escola:.1f}")
with c2:
    st.metric(label="Rede da Escola", value=rede_sel)
with c3:
    st.metric(label="Município da Escola", value=municipio_sel)
with c4:
    st.metric(label="Ranking Geral", value=rk_geral)

# ====== LINHA 2: Médias com delta vs escola selecionada ======
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(label="Média Geral (VL_D)", value=f"{media_geral:.1f}",    delta=delta_vs_escola(media_geral))
with c2:
    st.metric(label="Média Estadual (VL_D)", value=f"{media_estadual:.1f}", delta=delta_vs_escola(media_estadual))
with c3:
    st.metric(label="Média Municipal (VL_D)", value=f"{media_municipal:.1f}", delta=delta_vs_escola(media_municipal))
with c4:
    st.metric(label="Média Fortaleza (VL_D)", value=f"{media_fortaleza:.1f}", delta=delta_vs_escola(media_fortaleza))


# ---------------------------------------------------------------------------------------------------------
# GRÁFICOS
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(
        top_10_geral_grafico,
        use_container_width=True,
        config={'displayModeBar': False}
    )
with col2:
    st.plotly_chart(
        top10_estadual_grafico,
        use_container_width=True,
        config={'displayModeBar': False}
    )
with col1:
    st.plotly_chart(
        top10_municipal_grafico,
        use_container_width=True,
        config={'displayModeBar': False}
    )
with col2:
    st.plotly_chart(
        top10_fortaleza_grafico,
        use_container_width=True,
        config={'displayModeBar': False}
    )

# ====== Comparativo: Escola vs médias das Top 30 por recorte ======

def media_top30(df_base: pd.DataFrame) -> float:
    # pega as 30 maiores e calcula média
    top30 = (
        df_base[['NM_ESCOLA', 'VL_D']].dropna()
        .sort_values('VL_D', ascending=False)
        .head(30)
    )
    return float(top30['VL_D'].mean())

vl_escola_sel    = vl_escola 
media_top30_mun  = media_top30(df_9ano[df_9ano['DC_REDE'].str.upper() == 'MUNICIPAL'])
media_top30_est  = media_top30(df_9ano[df_9ano['DC_REDE'].str.upper() == 'ESTADUAL'])
media_top30_for  = media_top30(df_9ano[df_9ano['NM_MUNICIPIO'].str.upper() == 'FORTALEZA'])

df_comp = pd.DataFrame({
    'NM_ESCOLA': [
        'Escola Selecionada',
        'Municipal',
        'Estadual',
        'Fortaleza',
    ],
    'VL_D': [vl_escola_sel, media_top30_mun, media_top30_est, media_top30_for],
    '__cor': ['Selecionada', 'Outras', 'Outras', 'Outras']
})

grafico_comp = grafico_barras_vertical(
    df=df_comp,
    x_col='VL_D',
    y_col='NM_ESCOLA',
    titulo='Comparativo de Escola Selecionada com Médias das Top 30',
    hue_col='__cor',
    palette=PALETA,
    max_label_len=20  # rótulo abreviado no eixo, mantendo categoria única
)

st.plotly_chart(
    grafico_comp,
    use_container_width=True,
    config={'displayModeBar': False}
)


# Cálculo dos desvios em relação às médias Top 30
desvio_mun = vl_escola_sel - media_top30_mun
desvio_est = vl_escola_sel - media_top30_est
desvio_for = vl_escola_sel - media_top30_for

def formatar_desvio(desvio: float) -> str:
    """Retorna um span HTML com cor, fundo e ícone conforme o sinal do desvio."""
    if desvio >= 0:
        cor_texto = "#166534"      # verde escuro
        cor_fundo = "#dcfce7"      # verde bem claro
        icone = "⬆️"
        sinal = "+"
    else:
        cor_texto = "#991b1b"      # vermelho escuro
        cor_fundo = "#fee2e2"      # vermelho bem claro
        icone = "⬇️"
        sinal = ""  # o sinal negativo já vem no número

    return (
        f"<span style='background-color:{cor_fundo}; color:{cor_texto}; "
        f"padding:2px 6px; border-radius:4px; font-weight:bold; "
        f"font-size:0.95rem;'>"
        f"{icone} {sinal}{desvio:.1f}%"
        f"</span>"
    )

desvio_mun_html = formatar_desvio(desvio_mun)
desvio_est_html = formatar_desvio(desvio_est)
desvio_for_html = formatar_desvio(desvio_for)

texto_comp = (
    f"Em comparativo às top 30 escolas da rede estadual, municipal e da cidade de Fortaleza, "
    f"a escola selecionada possui um desvio de {desvio_mun_html} em relação às escolas municipais, "
    f"{desvio_est_html} em relação às escolas estaduais e "
    f"{desvio_for_html} em relação às escolas da cidade de Fortaleza."
)

st.markdown(texto_comp, unsafe_allow_html=True)



# Tabela completa das escolas do 9º ano ---------------------------------------------------------------------------------
st.write("### Tabela Completa das Escolas do 9º Ano")

df_exibicao_9ano = (
    df_9ano[['DC_REDE','NM_MUNICIPIO','NM_ESCOLA','VL_D']]
    .dropna()
    .sort_values(by='VL_D', ascending=False)
    .reset_index(drop=True)
)

df_exibicao_9ano.index = df_exibicao_9ano.index + 1
df_exibicao_9ano.index.name = 'RANKING'

st.dataframe(df_exibicao_9ano)
#---------------------------------------------------------------------------------------------------------













