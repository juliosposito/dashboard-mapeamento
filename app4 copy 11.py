# streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import unicodedata
from unicodedata import normalize

# --- Auto-launch Streamlit if executed via "python app.py" (safe) ---
import os, sys, pathlib

def _running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__" and not _running_inside_streamlit():
    # evita relançar mais de 1x
    if os.environ.get("APP_AUTOLAUNCHED") != "1":
        os.environ["APP_AUTOLAUNCHED"] = "1"
        here = pathlib.Path(__file__).resolve()
        # porta automática (evita "port in use")
        os.execvpe(sys.executable,
                   [sys.executable, "-m", "streamlit", "run", str(here), "--server.headless", "false"],
                   os.environ)
# --- fim do autolançador ---



st.set_page_config(page_title="Mapa do Forms (BDR)", layout="wide")

# ======================
# UPLOAD DO ARQUIVO
# ======================
st.sidebar.markdown("### 📂 Selecione o arquivo do Forms")
uploaded_file = st.sidebar.file_uploader("Carregar Excel (.xlsx)", type=["xlsx"])

if uploaded_file is None:
    st.info("👉 Envie o arquivo `.xlsx` na barra lateral para começar.")
    st.stop()

# ======================
# HELPERS
# ======================

# === Destaque visual (CSS + helpers) ===
st.markdown("""
<style>
.hl {
  background-color: #fff3b0;  /* amarelo suave */
  color: #000;
  font-weight: 700;
  padding: 6px 8px;
  border-radius: 6px;
  display: inline-block;
  margin: 2px 0 8px 0;
}
.small-id { color:#6c757d; font-weight:600; margin-right:6px; }
</style>
""", unsafe_allow_html=True)

def should_highlight(texto, highlight_set=None, keywords=None, normalizer=None):
    t = texto if normalizer is None else normalizer(texto)
    if highlight_set and texto in highlight_set:
        return True
    if keywords:
        for kw in keywords:
            if kw and kw.lower() in t.lower():
                return True
    return False

def ensure_state_set(key):
    if key not in st.session_state:
        st.session_state[key] = set()
    return st.session_state[key]

def norm(text: str) -> str:
    """Normaliza para busca: minúsculas + sem acento."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()


def highlight_block(section_key: str, cat: str, df_sub: pd.DataFrame, normalizer=norm):
    """
    section_key: 'K' | 'Q' | 'R'
    cat: nome da categoria
    df_sub: DataFrame com ['BDR_ID','texto'] dessa categoria
    """
    st.markdown(f"**{cat}**")

    # conjunto persistente de destaques por categoria
    sel_key = f"HLSET_{section_key}_{cat}"
    chosen = ensure_state_set(sel_key)  # -> set()

    kw_key = f"kw_{section_key}_{cat}"
    clear_flag_key = f"__clear_flag__{section_key}_{cat}"

# Se a rotina de limpar sinalizou que devemos zerar o campo,
# faça isso ANTES de instanciar o text_input
    if st.session_state.get(clear_flag_key):
        st.session_state[kw_key] = ""
        st.session_state.pop(clear_flag_key, None)


    # ---------- Busca por termos (OR) + botão limpar ----------
    c1, c2 = st.columns([6, 1])
    with c1:
        term_str = st.text_input(
            f"Destacar por termo ({section_key} • {cat}) — separe por ';'",
            key=f"kw_{section_key}_{cat}"
        )
    with c2:
        if st.button("Limpar", key=f"clear_{section_key}_{cat}"):
            # 1) limpa set de destaques
            chosen.clear()
            st.session_state[sel_key] = chosen
            # 2) reseta TODOS os toggles dessa categoria
            for i, row in df_sub.iterrows():
                ck_key = f"chk_{section_key}_{cat}_{row['BDR_ID']}_{i}"
                st.session_state.pop(ck_key, None)
            # 3) pede para zerar o campo na PRÓXIMA execução
            st.session_state[clear_flag_key] = True
            # 4) força reexecução agora
            st.rerun()



    term_str = (term_str or "").strip()
    if term_str == "":
        # campo vazio -> desmarca todos (se houver)
        # (comente as 2 linhas abaixo se NÃO quiser esse comportamento)
        if chosen:
            chosen.clear()
            st.session_state[sel_key] = chosen
    else:
        # separa por ; , / |  (qualquer um serve)
        raw_terms = [t.strip() for t in re.split(r"[;,\|/]+", term_str) if t.strip()]
        terms = [normalizer(t) for t in raw_terms]
        if terms:
            for _, row in df_sub.iterrows():
                txt = str(row["texto"]).strip()
                tnorm = normalizer(txt)
                if any(t in tnorm for t in terms):
                    chosen.add(txt)
            st.session_state[sel_key] = chosen

# ---------- Lista com toggles por frase (layout em 2 colunas) ----------
    for i, row in df_sub.iterrows():
        bdr = row["BDR_ID"]
        txt = str(row["texto"]).strip()
        ck_key = f"chk_{section_key}_{cat}_{bdr}_{i}"

        col_tog, col_txt = st.columns([1, 12])
        with col_tog:
            on = st.toggle(
                "",
                value=(txt in chosen),
                key=ck_key,
                label_visibility="collapsed"  # oculta o label do toggle
            )

        # atualiza o set conforme o toggle
        if on: chosen.add(txt)
        else:  chosen.discard(txt)

        # monta HTML da frase + ID (ID sutil, ao lado)
        is_hl = (txt in chosen)
        cls = "hl" if is_hl else ""   # usa sua classe de destaque amarela
        html = f"<span class='{cls}'>“{txt}”</span> <span class='small-id'>#{bdr}</span>"

        with col_txt:
            st.markdown(html, unsafe_allow_html=True)

    st.divider()




def render_frases(sub_df, highlight_set=None, keywords=None, normalizer=lambda x: x):
    """
    sub_df: DataFrame com colunas ['BDR_ID','texto']
    """
    for _, row in sub_df.iterrows():
        bdr = row["BDR_ID"]
        txt = str(row["texto"]).strip()
        if should_highlight(txt, highlight_set, keywords, normalizer):
            st.markdown(f"<span class='small-id'>#{bdr}</span> <span class='hl'>“{txt}”</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='small-id'>#{bdr}</span> “{txt}”", unsafe_allow_html=True)


def norm(s: str) -> str:
    if not isinstance(s, str): return ""
    s = normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"\s+"," ", s.replace("\xa0"," ")).strip().lower()
    return s

def find_col(df, needles):
    cols = {norm(c): c for c in df.columns}
    for needle in needles:
        key = norm(needle)
        for nname, real in cols.items():
            if key in nname:
                return real
    return None

def split_multi(series):
    s = (series.dropna().astype(str).str.split(";").explode().str.strip())
    return s[s.ne("")]  # remove strings vazias

def normalize_g_value(v: str) -> str:
    if not isinstance(v, str): 
        return ""
    t = norm(v)
    if any(k in t for k in ["automatiz", "automacao"]):
        return "Automação"
    if t in {"nao", "não", "n", "no"}:
        return "Não"
    return v.strip()


def donut(title, counts, colors=None):
    df = counts.reset_index()
    df.columns = ["Categoria", "Qtd"]

    fig = px.pie(
        df,
        names="Categoria",
        values="Qtd",
        hole=0.45,
        color="Categoria",
        color_discrete_sequence=colors
    )

    # 👉 mostra rótulo + % + (valor), e mantém o hover completo
    fig.update_traces(
        textinfo="label+percent+value",
        texttemplate="%{label}<br>%{percent} (%{value})",
        textposition="inside",
        insidetextorientation="auto",
        hovertemplate="%{label}: %{value} (%{percent})"
    )

    fig.update_layout(
        title=title,
        showlegend=False,                  # ligue se quiser legenda extra
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig


def barh(title, counts, colors=None):
    df = counts.sort_values(ascending=False).reset_index()
    df.columns = ["Categoria","Qtd"]
    fig = px.bar(df, x="Qtd", y="Categoria", orientation="h",
                 color="Categoria", color_discrete_sequence=colors)
    fig.update_traces(texttemplate="%{x}", textposition="outside")
    fig.update_layout(title=title, showlegend=False, margin=dict(l=10,r=10,t=60,b=10))
    return fig

def stacked_100(title, df_levels):
    """
    df_levels: linhas = áreas, colunas = [1,2,3] com percentuais (0–100)
    """
    # 1) média e ordem (maior -> menor)
    media = (df_levels[1]*1 + df_levels[2]*2 + df_levels[3]*3) / 100
    ordem = media.sort_values(ascending=False).index

    # 2) dados "longos"
    long = (
        df_levels.reset_index()
                 .melt(id_vars="index", var_name="Nível", value_name="Perc")
                 .rename(columns={"index": "Área"})
    )
    long["Área"] = pd.Categorical(long["Área"], categories=ordem, ordered=True)

    # 3) barras 100% + rótulos de % dentro
    fig = px.bar(
        long, x="Área", y="Perc", color="Nível",
        text="Perc",  # usamos Perc como texto
        color_discrete_map={1:"#ff99ac", 2:"#ffca3a", 3:"#8ac926"}
    )
    fig.update_traces(
        texttemplate="%{text:.0f}%",   # mostra 45%
        textposition="inside",         # texto dentro
        insidetextanchor="middle",
        textangle=0,                   # evita girar (resolve o 7% “de lado”)
        cliponaxis=False
    )
    fig.update_layout(
        barmode="stack",
        title=title,
        yaxis=dict(range=[0,110], title="Perc"),
        legend_title="Nível",
        margin=dict(l=10, r=10, t=60, b=10)
    )
    # mantém ordem esquerda->direita
    fig.update_xaxes(categoryorder="array", categoryarray=list(ordem))

    # 4) anotações da média (fica visível em tema escuro)
    for area in ordem:
        fig.add_annotation(
            x=area, y=103,
            text=f"Média {media[area]:.2f}",
            showarrow=False,
            font=dict(size=12, color="white")  # troque para "black" se usar tema claro
        )

    return fig


# ======================
# CATEGORIZADORES K / Q / R
# ======================
PALETTE = {
    "Clientes":"#ff99ac",
    "Sistemas / Integração":"#1982c4",
    "Automação":"#ffca3a",
    "Comunicação interna":"#8ac926",
    "Gestão / Processos":"#c8a2c8",

    "Ferramentas / Automação / Integração":"#1982c4",
    "Padronização / Comunicação Interna":"#8ac926",
    "Clientes / Atendimento":"#ff99ac",
    "Gestão de Processos":"#c8a2c8",

    "Painel SIGA / CIGAH (Automação de registros)":"#1982c4",
    "Emails / Comunicação Escrita":"#8ac926",
    "Planilhas / Relatórios / Bases":"#c8a2c8",
    "Comunicação / Integração de Sistemas":"#ffca3a",
    "WhatsApp / Outros Canais":"#ff99ac",
}

def cat_K(text):
    t = norm(text)
    if any(k in t for k in ["whatsapp","zap","cliente","contato","localizar"]):
        return "Clientes"
    if any(k in t for k in ["automatiz","automacao"]) and not any(k in t for k in ["siga","cigah","sistema"]):
        return "Automação"
    if any(k in t for k in ["sistema","integracao","interaxa","lentidao","migrasse","simultaneamente","siga","cigah"]):
        return "Sistemas / Integração"
    if any(k in t for k in ["comunicacao","teams","divulga","rede"]):
        return "Comunicação interna"
    return "Gestão / Processos"

def cat_Q(text):
    t = norm(text)
    if any(k in t for k in ["automatiz","automacao","sistema","integrad","autobase","interface","siga","cigah","sinha","tildes","relatorio"]):
        return "Ferramentas / Automação / Integração"
    if any(k in t for k in ["cliente","whatsapp","localizar","qualidade"]):
        return "Clientes / Atendimento"
    if any(k in t for k in ["banco de respostas","padronizacao","padroniz","email padrao","copie e cole"]):
        return "Padronização / Comunicação Interna"
    return "Gestão de Processos"

def cat_R(text):
    t = norm(text)
    if any(k in t for k in ["siga","cigah","cigad","registro","inclusao","texto no painel"]):
        return "Painel SIGA / CIGAH (Automação de registros)"
    if "email" in t or "e-mail" in t:
        return "Emails / Comunicação Escrita"
    if any(k in t for k in ["planilha","portal fluxo uc","base","verificar diariamente"]):
        return "Planilhas / Relatórios / Bases"
    if "whatsapp" in t:
        return "WhatsApp / Outros Canais"
    if "sistemas internos" in t or "comunicacao entre os sistemas" in t or "integracao" in t:
        return "Comunicação / Integração de Sistemas"
    return "Planilhas / Relatórios / Bases"

def categorize(series, fn):
    s = series.dropna().astype(str)
    dfc = pd.DataFrame({"texto": s, "categoria": s.map(fn)})
    counts = dfc["categoria"].value_counts()
    return dfc, counts

# ======================
# CARGA DO EXCEL → “BDR” em memória
# ======================
@st.cache_data
def load_forms(file):
    df = pd.read_excel(file, sheet_name=0)
    df.insert(0, "BDR_ID", np.arange(1, len(df)+1))  # chave por linha
    return df

df = load_forms(uploaded_file)
st.success(f"Arquivo carregado: **{uploaded_file.name}** — {len(df)} respostas")

# ======================
# MAPEAR COLUNAS (seus nomes reais)
# ======================
COL_E = find_col(df, ["Qual a sua célula"])
COL_F = find_col(df, ["Quais ferramentas e sistemas digitais sua célula utiliza"])
COL_G = find_col(df, ["gostaria de usar mais", "nao domina bem"])
COL_H = find_col(df, ["onde voce registra o atendimento"])
COL_I = find_col(df, ["qual ferramenta abaixo e a que produz mais resultados"])
COL_J = find_col(df, ["Quais o principais desafios enfrentados", "principais desafios", "dia a dia"])
COL_K = find_col(df, ["voce percebe algum outro desafio", "pergunta opcional"])
COL_Q = find_col(df, ["voce identifica algum outro tipo de solucao"])
COL_R = find_col(df, ["se voce pudesse automatizar uma tarefa repetitiva"])
COL_S = find_col(df, ["como voce enxerga a cultura de inovacao"])
COL_T = find_col(df, ["voce gostaria de participar de iniciativas de inovacao"])

# Detectar L–P (impacto 1–3)
IMPACT_COLS = []
for c in df.columns:
    s = pd.to_numeric(df[c], errors="coerce")
    vals = set(s.dropna().astype(int).unique().tolist())
    if vals and vals.issubset({1,2,3}):
        IMPACT_COLS.append(c)

# ======================
# FILTROS GLOBAIS
# ======================
with st.sidebar:
    st.markdown("### 🔎 Filtros Globais (cross-filter)")
    if COL_E:
        options = ["(todas)"] + sorted(df[COL_E].dropna().astype(str).unique().tolist())
        filtro_celula = st.selectbox("Célula", options, index=0)
    else:
        filtro_celula = "(todas)"
    busca = st.text_input("Busca livre (qualquer coluna)", "")

df_filt = df.copy()
if COL_E and filtro_celula != "(todas)":
    df_filt = df_filt[df_filt[COL_E].astype(str) == filtro_celula]
if busca.strip():
    patt = re.compile(re.escape(busca.strip()), re.IGNORECASE)
    mask = df_filt.apply(lambda row: row.astype(str).str.contains(patt).any(), axis=1)
    df_filt = df_filt[mask]

st.sidebar.success(f"Linhas ativas no BDR: {len(df_filt)}")

# ======================
# LAYOUT (ordem do Forms)
# ======================
st.title("📋 Mapa do Forms")
st.caption("Os filtros globais afetam todas as visões abaixo.")

# 1) Identificação
st.header("1) Identificação")
if COL_E:
    st.plotly_chart(
        barh("Participação no forms por Célula", df_filt[COL_E].value_counts(),
             colors=["#1982c4","#ff99ac","#8ac926","#ffca3a","#c8a2c8"]),
        use_container_width=True
    )
st.dataframe(df_filt.head(50))

st.markdown("---")

# 2) Ferramentas & Processos (E–I)
st.header("2) Ferramentas & Processos")

# ---- Linha de cima: F e G ----
top_left, top_right = st.columns(2)
with top_left:
    if COL_F:
        st.plotly_chart(
            barh("Quais ferramentas e sistemas digitais sua célula utiliza atualmente <br> para realizar suas atividades? (múltipla escolha)",
                 split_multi(df_filt[COL_F]).value_counts(),
                 colors=px.colors.qualitative.Set3),
            use_container_width=True
        )
with top_right:
    if COL_G:
        serie_g = df_filt[COL_G].dropna().astype(str).map(normalize_g_value)
        cont_g = serie_g[serie_g != ""].value_counts()
        if not cont_g.empty:
            st.plotly_chart(
                barh("Existe alguma ferramenta/sistema que gostaria de usar mais, <br> mas sente que não tem acesso ou não domina bem?",
                     cont_g,
                     colors=px.colors.qualitative.Set2),
                use_container_width=True
            )

# ---- Linha de baixo: H e I ----
bottom_left, bottom_right = st.columns(2)
with bottom_left:
    if COL_H:
        st.plotly_chart(
            barh("Onde você registra o atendimento/acionamento? (múltipla escolha)",
                 split_multi(df_filt[COL_H]).value_counts(),
                 colors=px.colors.qualitative.Dark24),
            use_container_width=True
        )
with bottom_right:
    if COL_I:
        # 1) Lista de opções completas (ordem opcional)
        opcoes_I = [
            "Whatsapp",
            "Email",
            "Não se aplica à minha atividade",
            "Telefone (gravação vocal)",
            "Outro",
        ]

        # 2) Contagem real
        cont = df_filt[COL_I].dropna().str.strip().value_counts()

        # 3) Reindexa para incluir opções com zero
        cont_all = cont.reindex(opcoes_I, fill_value=0)

        # 4) Separa as que serão plotadas (>0) e as zeradas
        cont_plot = cont_all[cont_all > 0]
        zeradas = cont_all[cont_all == 0].index.tolist()

        # 5) Constrói DataFrame com nomes consistentes
        df_plot = cont_plot.rename_axis("Opção").reset_index(name="Votos")

        # Ajusta categoria que tinha ficado mal formatada no gráfico 
        df_plot["Opção"] = df_plot["Opção"].replace({
    "Não se aplica à minha atividade": "Não se aplica<br>à minha atividade"
        })

        # 6) Gráfico de rosca com % + absoluto
        fig = px.pie(
            df_plot,
            names="Opção",
            values="Votos",
            hole=0.45,
            color="Opção",
            color_discrete_sequence=["#ff595e","#1982c4","#8ac926","#ffca3a","#c8a2c8"],
        )
        fig.update_traces(
    textinfo="label+percent+value",
    texttemplate="%{label}<br>%{percent} (%{value})",
    textposition="inside",
    insidetextorientation="auto",
    hovertemplate="%{label}: %{value} (%{percent})"
        )
        fig.update_layout(
            title={
                "text": "Da sua base de acionamento, qual ferramenta abaixo é a que produz mais resultados?",
                "x": 0.0,          # alinha à esquerda
                "xanchor": "left", # ancora à esquerda
                "y": 0.98,         # opcional: sobe um pouco
                "yanchor": "top"
            },
            showlegend=False,
            margin=dict(l=10, r=10, t=90, b=10)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 7) Exibe opções sem votos (limpo e informativo)
        if zeradas:
            st.markdown("**Sem votos:** " + ", ".join(f"*{z}*" for z in zeradas))

st.markdown("---")


# ---------- 3) Dificuldades & Gargalos (J e K) ----------
st.header("3) Dificuldades & Gargalos")

col_j, col_k = st.columns(2)

# J — múltipla escolha: principais desafios
with col_j:
    if COL_J:
        serie_j = split_multi(df_filt[COL_J])              # já remove vazios (pela sua opção B)
        if not serie_j.empty:
            st.plotly_chart(
                barh("Quais os principais desafios enfrentados pela sua célula no dia a dia?",
                     serie_j.value_counts(),
                     colors=px.colors.qualitative.Set2),
                use_container_width=True
            )

# K — aberta categorizada
with col_k:
    if COL_K:
        respK = df_filt[COL_K].dropna().astype(str)
        dfK = pd.DataFrame({"BDR_ID": df_filt.loc[respK.index, "BDR_ID"], "texto": respK})
        dfK["categoria"] = dfK["texto"].map(cat_K)
        cntK = dfK["categoria"].value_counts()

        # 1) cria a figura normalmente com o helper
        figK = donut(
            "Você percebe algum outro desafio/dificuldade além dos listados ao lado? qual?<br>(Resposta aberta e opcional – categorizada por IA) [Gráfico K]",
            cntK,
            colors=[PALETTE[k] for k in cntK.index]
        )

        # 2) adiciona % + valor E mantém o rótulo (label)
        figK.update_traces(
            textinfo="label+percent+value",
            texttemplate="%{label}<br>%{percent} (%{value})",
            textposition="inside",
            insidetextorientation="auto",
            hovertemplate="%{label}: %{value} (%{percent})"
        )


        # (opcional) se alguma fatia for pequena e o texto sumir, exiba legenda:
        # figK.update_layout(showlegend=True)

        st.plotly_chart(figK, use_container_width=True)

with st.expander("Respostas completas [Gráfico K]"):
    for cat in cntK.index:
        subK = dfK[dfK["categoria"] == cat][["BDR_ID", "texto"]]
        highlight_block("K", cat, subK, normalizer=norm)



st.markdown("---")


# ---------- 4) Ideias & Soluções (L–P primeiro, depois Q e R) ----------
st.header("4) Ideias & Soluções")

# 4.1) L–P (impacto 1–3) — AGORA VEM PRIMEIRO
if IMPACT_COLS:
    st.markdown(
        "**Que tipo de solução ajudaria sua célula a ser mais produtiva ou assertiva?**<br>"
        "Marque 1, 2 ou 3, conforme o nível de impacto que você avalia que teria na atividade de sua célula:<br>"
        "1 – Pouco impacto &nbsp;&nbsp; 2 – Impacto intermediário &nbsp;&nbsp; 3 – Alto impacto",
        unsafe_allow_html=True
    )

    counts = {}
    for c in IMPACT_COLS:
        s = pd.to_numeric(df_filt[c], errors="coerce").dropna().astype(int)
        vc = s.value_counts().reindex([1,2,3], fill_value=0)
        tot = max(vc.sum(), 1)
        counts[c] = (vc / tot * 100).round(1)
    pct_df = pd.DataFrame(counts).T[[1,2,3]]

    st.plotly_chart(
        stacked_100("Distribuição dos níveis de impacto por área", pct_df),
        use_container_width=True
    )

st.markdown("---")

# 4.2) Q e R embaixo, lado a lado
col_q, col_r = st.columns(2)

# Q — Soluções adicionais
with col_q:
    if COL_Q:
        respQ = df_filt[COL_Q].dropna().astype(str)
        dfQ = pd.DataFrame({"BDR_ID": df_filt.loc[respQ.index, "BDR_ID"], "texto": respQ})
        dfQ["categoria"] = dfQ["texto"].map(cat_Q)
        cntQ = dfQ["categoria"].value_counts()

        # 🔹 Texto explicativo fora do gráfico
        st.markdown(
            "**Você identifica algum outro tipo de solução que contribuiria para a produtividade e resultado na sua atividade? Qual?**  \n"
            "_(Resposta aberta e opcional – categorizada por IA)_"
        )

        # 🔹 Gráfico agora só com título curtinho
        st.plotly_chart(
            donut("Gráfico Q", cntQ, colors=[PALETTE[k] for k in cntQ.index]),
            use_container_width=True
        )
with st.expander("Respostas completas [Gráfico Q]"):
    for cat in cntQ.index:
        subQ = dfQ[dfQ["categoria"] == cat][["BDR_ID", "texto"]]
        highlight_block("Q", cat, subQ, normalizer=norm)


# R — Automatizar tarefa
with col_r:
    if COL_R:
        respR = df_filt[COL_R].dropna().astype(str)
        dfR = pd.DataFrame({"BDR_ID": df_filt.loc[respR.index, "BDR_ID"], "texto": respR})
        dfR["categoria"] = dfR["texto"].map(cat_R)
        cntR = dfR["categoria"].value_counts()

        st.markdown(
            "**Se você pudesse automatizar uma tarefa repetitiva, qual seria?**  \n"
            "_(Resposta aberta e opcional – categorizada por IA)_"
        )

        st.plotly_chart(
            donut("Gráfico R", cntR, colors=[PALETTE[k] for k in cntR.index]),
            use_container_width=True
        )
with st.expander("Respostas completas [Gráfico R]"):
    for cat in cntR.index:
        subR = dfR[dfR["categoria"] == cat][["BDR_ID", "texto"]]
        highlight_block("R", cat, subR, normalizer=norm)




# 5) Cultura de Inovação (S, T)
st.header("5) Cultura de Inovação")
col_s, col_t = st.columns(2)
with col_s:
    if COL_S:
        st.plotly_chart(
            donut("Como você enxerga a Cultura de Inovação na unidade hoje?",
                  df_filt[COL_S].value_counts(),
                  colors=["#8ac926","#ffca3a","#ff595e"]),
            use_container_width=True
        )
with col_t:
    if COL_T:
        st.plotly_chart(
            donut("Você gostaria de participar de iniciativas de inovação (comissão, projetos, testes de soluções)?",
                  df_filt[COL_T].value_counts(),
                  colors=["#8ac926","#ffca3a","#ff595e"]),
            use_container_width=True
        )

# EXPORTS
st.markdown("### Exportar visões")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("Base filtrada (CSV)",
                       df_filt.to_csv(index=False).encode("utf-8"),
                       file_name="bdr_filtrado.csv", mime="text/csv")
with c2:
    if COL_K:
        st.download_button("K categorizado (CSV)",
                           pd.DataFrame({"BDR_ID": dfK["BDR_ID"], "Categoria": dfK["categoria"], "Texto": dfK["texto"]}).to_csv(index=False).encode("utf-8"),
                           file_name="K_categorizado.csv", mime="text/csv")
with c3:
    if COL_Q and COL_R:
        st.download_button("Q categorizado (CSV)",
                           pd.DataFrame({"BDR_ID": dfQ["BDR_ID"], "Categoria": dfQ["categoria"], "Texto": dfQ["texto"]}).to_csv(index=False).encode("utf-8"),
                           file_name="Q_categorizado.csv", mime="text/csv")
        st.download_button("R categorizado (CSV)",
                           pd.DataFrame({"BDR_ID": dfR["BDR_ID"], "Categoria": dfR["categoria"], "Texto": dfR["texto"]}).to_csv(index=False).encode("utf-8"),
                           file_name="R_categorizado.csv", mime="text/csv")

st.caption("Cada linha do BDR mantém vínculo por BDR_ID. Todos os gráficos/tabelas respeitam os filtros globais.")
