import streamlit as st
import pandas as pd
import plotly.express as px
engine='python'

# ----- Page configs
st.set_page_config(layout= "wide")


def formataNumero(valor):
    if valor >= 1_000_000_000:
        return f'{valor / 1_000_000_000:.1f} b'
    if valor >= 1_000_000:
        return f'{valor / 1_000_000:.1f} m'
    if valor >= 1000:
        return f'{valor / 1000:.1f} k'
    
    return str(valor)


# -----------------------------------------
#       Dados
#-------------------------------------------

#importa os dados de acordo com a formatação do arquivo CSV da WMC 
#Pedidos analiticos
df = pd.read_csv("familias_sem_compras.csv", sep=';', header=0)
# Exibindo o DataFrame  
#st.dataframe(df, use_container_width=True)
#Verificando o tamanho da tabela em colunas e linhas
#st.write(f"Quantidade de linhas em familias_sem_compras.csv: {df.shape[0]}")
#st.write(f"Quantidade de colunas em familias_sem_compras.csv: {df.shape[1]}")
#st.write("### Informações do DataFrame familias_sem_compras.csv")

#-------------------------------------------
#verificando quais colunas tem
#st.subheader("Headers (nomes das colunas) da base familias_sem_compras.csv")
#st.write(list(df.columns))


#-------------------------------------------
# Filtros e painel de selecão
#-------------------------------------------

# Filtro por REGIÃO em formato de botão (radio), padrão: primeira região
if 'REGIÃO' in df.columns:
    regioes = df['REGIÃO'].dropna().unique()
    regiao_selecionada = st.sidebar.radio(
        "Selecione a região:", 
        options=regioes, 
        index=0
    )
    df_filtrado = df[df['REGIÃO'] == regiao_selecionada]
else:
    df_filtrado = df.copy()

# Filtro por VENDEDOR em formato dropdown (multiselect), padrão: todos marcados
if 'VENDEDOR' in df_filtrado.columns:
    vendedores = df_filtrado['VENDEDOR'].dropna().unique()
    vendedor_selecionado = st.sidebar.selectbox(
        "Selecione o vendedor:",
        options=vendedores,
        index=0
    )
    df_filtrado = df_filtrado[df_filtrado['VENDEDOR'] == vendedor_selecionado]


# Filtro por ROTA em formato selectbox (seleção única)
if 'ROTA' in df_filtrado.columns:
    rotas = df_filtrado['ROTA'].dropna().unique()
    rota_selecionada = st.sidebar.selectbox(
        "Selecione uma rota:",
        options=rotas,
        index=0
    )
    df_filtrado = df_filtrado[df_filtrado['ROTA'] == rota_selecionada]

# Filtro por DIA em formato selectbox (seleção única)
if 'DIA' in df_filtrado.columns:
    dias = df_filtrado['DIA'].dropna().unique()
    dia_selecionado = st.sidebar.selectbox(
        "Selecione um dia:",
        options=dias,
        index=0
    )
    df_filtrado = df_filtrado[df_filtrado['DIA'] == dia_selecionado]


# Filtro para selecionar colunas de produtos específicos
colunas_produtos = [
    'BIS WAFER', 
    'BISC. OREO', 
    'BOMBOM 540GR', 
    'BOMBOM KG', 
    'HALLS', 
    "TRIDENT 21'S"
]
# Filtro para selecionar um produto específico usando botões (um para cada produto)
colunas_produtos_existentes = [col for col in colunas_produtos if col in df_filtrado.columns]

produto_selecionado = None
if colunas_produtos_existentes:
    st.sidebar.subheader("Selecione um produto para exibir")
    cols = st.sidebar.columns(3)  # 3 colunas para distribuir até 6 botões
    for i, produto in enumerate(colunas_produtos_existentes):
        if cols[i % 3].button(produto):
            produto_selecionado = produto
    # Se nenhum botão foi pressionado, seleciona o primeiro por padrão
    if not produto_selecionado:
        produto_selecionado = colunas_produtos_existentes[0]
    produtos_selecionados = [produto_selecionado]
else:
    produtos_selecionados = []

# Sempre mostrar as colunas 'ROTA', 'DIA', 'CÓD' no início, depois a coluna filtrada de produto, depois 'CLIENTE' e 'CIDADE' (se existirem)
colunas_fixas_inicio = []
colunas_fixas_fim = ['CÓD','CLIENTE','FANTASIA', 'CIDADE']

colunas_inicio_existentes = [col for col in colunas_fixas_inicio if col in df_filtrado.columns]
colunas_fim_existentes = [col for col in colunas_fixas_fim if col in df_filtrado.columns]

# Adiciona a coluna de produto selecionada (apenas uma)
colunas_exibir = colunas_inicio_existentes + [col for col in produtos_selecionados if col not in colunas_inicio_existentes + colunas_fim_existentes] + colunas_fim_existentes

# Ordena o DataFrame filtrado pela coluna do produto selecionado (ordem crescente)
if produtos_selecionados and produtos_selecionados[0] in df_filtrado.columns:
    df_exibir = df_filtrado[colunas_exibir].sort_values(by=produtos_selecionados[0], ascending=False)
else:
    df_exibir = df_filtrado[colunas_exibir]

#-------------------------------------------
# Exibir o título do dashboard
#-------------------------------------------
# Exibir o título do dashboard centralizado
if produtos_selecionados and len(produtos_selecionados) > 0:
    titulo = f"Resumo de Cobertura de {produtos_selecionados[0]}"
else:
    titulo = "Resumo de Cobertura"
st.markdown(f"<h1 style='text-align: center;'>{titulo}</h1>", unsafe_allow_html=True)


# Exibir o vendedor selecionado no centro do dashboard
if 'VENDEDOR' in df_filtrado.columns and 'vendedor_selecionado' in locals():
    st.markdown(
        f"<h3 style='text-align: center;'>Vendedor: {vendedor_selecionado}</h3>",
        unsafe_allow_html=True
    )

# Contar itens específicos ❌ (0) e ✅ (1) na coluna de produto selecionado
if produtos_selecionados and produtos_selecionados[0] in df_exibir.columns:
    produto_col = produtos_selecionados[0]
    sim = (df_exibir[produto_col] == "✅").sum()
    nao = (df_exibir[produto_col] == "❌").sum()
 
# Gráfico de tree map usando as variáveis nao e sim
if produtos_selecionados and produtos_selecionados[0] in df_exibir.columns:
    produto_col = produtos_selecionados[0]
    fig_treemap = px.treemap(
        names=["Não compraram", "Compraram"],
        parents=["", ""],
        values=[nao, sim],
        color=["Não compraram", "Compraram"],
        color_discrete_map={"Não compraram": "#FF6961", "Compraram": "#77DD77"},
    )
    fig_treemap.update_traces(
        texttemplate='%{label}<br>%{value} (%{percentParent:.1%})',
        textfont_size=20,
        marker=dict(pad=dict(t=0, l=0, r=0, b=0))  # Remove padding interno
    )
    # Reduz ainda mais a área de plotagem
    fig_treemap.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        autosize=False,
        width=4000,
        height=85
    )
    st.plotly_chart(fig_treemap, use_container_width=False)


# Exibir o DataFrame filtrado com as colunas selecionadas, ocultando o índice
if (
    'ROTA' in df_filtrado.columns and 'rota_selecionada' in locals() and
    'DIA' in df_filtrado.columns and 'dia_selecionado' in locals()
):
    st.subheader(f'{rota_selecionada} | {dia_selecionado}ª-feira')
st.dataframe(df_exibir, use_container_width=True, hide_index=True)
