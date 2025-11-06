import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Abrir o arquivo Excel
xls = pd.ExcelFile('Planilhao_TCT_SPAECE_EF_2023_MT_240402-2 (1).xlsx')

df = pd.DataFrame(xls.parse('ESCOLA'))

df.columns = df.iloc[0]
df = df[1:]

df = df.drop(['CD_REDE','CD_ETAPA', 'CD_REGIONAL','CD_MUNICIPIO', 'CD_ESCOLA'], axis=1)

# Remover colunas completamente vazias (se existirem)
df = df.dropna(axis=1, how='all')

df_9ano = df[df['DC_ETAPA'] == 'ENSINO FUNDAMENTAL DE 9 ANOS - 9º ANO'].reset_index(drop=True)