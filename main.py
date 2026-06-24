import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time as _time
from io import BytesIO
import os

st.set_page_config(page_title="Reorganização de Rotas", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH      = os.path.join(BASE_DIR, "rotas.csv")
ENDERECO_PATH = os.path.join(BASE_DIR, "endereco.csv")

DIAS_SEMANA = {1: "Segunda", 2: "Terça", 3: "Quarta", 4: "Quinta", 5: "Sexta"}

COL_VISITAS = "VISITAR QUANTAS X"   # column that holds "1 VISITA" / "2 VISITAS"
CITY_MAX_SINGLE_DAY = 20             # cidades com até este nº de clientes → todos no mesmo dia

# OSRM (OpenStreetMap routing) — usa hierarquia real de vias
OSRM_BASE       = "https://router.project-osrm.org"
OSRM_TIMEOUT    = 15      # segundos por requisição
OSRM_MAX_COORDS = 99      # limite de coordenadas do servidor público


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.fillna("")

    # String normalisation
    for col in ["CLIENTE", "CIDADE", "BAIRRO", "ENDEREÇO", "AÇÃO", "VENDEDOR",
                COL_VISITAS]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["CIDADE"] = df["CIDADE"].str.upper()
    df["BAIRRO"] = df["BAIRRO"].str.upper()
    df["ENDEREÇO"] = df["ENDEREÇO"].str.upper()
    df[COL_VISITAS] = df[COL_VISITAS].str.upper()

    # Parse coordinates – Brazilian decimal comma → float
    for col in ["LATITUDE", "LONGITUDE"]:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", ".", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Geographic sort keys
#   · Valid lat/lon → sort by actual coordinates (numeric sweep)
#   · lat or lon == 0 → fallback to city-level centroid from valid rows;
#     if city has no valid rows → text sort (CIDADE/BAIRRO/ENDEREÇO) at tail
# ─────────────────────────────────────────────────────────────────────────────
def add_geo_sort_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds geographic sort keys following the hierarchy:
      1. Cidade   – ordered by city centroid
      2. Bairro   – ordered by bairro centroid within city
      3. Endereço – text grouping within bairro
      4. Ponto    – actual coordinates for proximity within same endereço

    Clients without valid lat/lon inherit the bairro centroid (→ city centroid
    as final fallback), so they are placed geographically among peers.
    """
    df = df.copy()
    has_coords = (df["LATITUDE"] != 0) & (df["LONGITUDE"] != 0)

    # City centroid — only from geocoded rows
    city_center = (
        df[has_coords]
        .groupby("CIDADE")[["LATITUDE", "LONGITUDE"]]
        .median()
        .rename(columns={"LATITUDE": "_city_lat", "LONGITUDE": "_city_lon"})
    )

    # Bairro centroid — only from geocoded rows
    bairro_center = (
        df[has_coords]
        .groupby(["CIDADE", "BAIRRO"])[["LATITUDE", "LONGITUDE"]]
        .median()
        .rename(columns={"LATITUDE": "_bairro_lat", "LONGITUDE": "_bairro_lon"})
    )

    df = df.join(city_center, on="CIDADE")
    df = df.join(bairro_center, on=["CIDADE", "BAIRRO"])

    # Cities with no geocoded clients sort to tail (0.0)
    df["_city_lat"] = df["_city_lat"].fillna(0.0)
    df["_city_lon"] = df["_city_lon"].fillna(0.0)

    # Bairros with no geocoded clients inherit city centroid
    df["_bairro_lat"] = df["_bairro_lat"].fillna(df["_city_lat"])
    df["_bairro_lon"] = df["_bairro_lon"].fillna(df["_city_lon"])

    # Individual point: actual coords → bairro centroid fallback
    df["_sort_lat"] = np.where(has_coords, df["LATITUDE"], df["_bairro_lat"])
    df["_sort_lon"] = np.where(has_coords, df["LONGITUDE"], df["_bairro_lon"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6_371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data
def load_home_addresses() -> dict:
    """Load salesperson home addresses from endereco.csv.
    Returns {VENDEDOR: (lat, lon)} or {} if the file is missing/unreadable."""
    if not os.path.exists(ENDERECO_PATH):
        return {}
    try:
        df_e = pd.read_csv(ENDERECO_PATH, sep=";", encoding="utf-8")
        df_e.columns = df_e.columns.str.strip()
        addresses: dict = {}
        for _, row in df_e.iterrows():
            v = str(row.get("VENDEDOR", "")).strip()
            lat = pd.to_numeric(
                str(row.get("LATITUDE", "0")).replace(",", "."), errors="coerce"
            )
            lon = pd.to_numeric(
                str(row.get("LONGITUDE", "0")).replace(",", "."), errors="coerce"
            )
            if v and not pd.isna(lat) and not pd.isna(lon) and lat != 0:
                addresses[v] = (float(lat), float(lon))
        return addresses
    except Exception:
        return {}


def _osrm_table(coords: list) -> list | None:
    """
    Chama o serviço OSRM /table/v1/driving e retorna a matriz NxN de durações (seg).
    coords: lista de tuplas (lat, lon).
    Retorna None em caso de falha.
    """
    if len(coords) < 2:
        return None
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_BASE}/table/v1/driving/{coord_str}?annotations=duration"
    try:
        r = requests.get(url, timeout=OSRM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("code") == "Ok":
            return data.get("durations")
    except Exception:
        pass
    return None


def _osrm_route_km(coords: list) -> float | None:
    """
    OSRM /route/v1/driving: retorna a distância real do trajeto em km.
    coords: lista de tuplas (lat, lon) na ordem de visita.
    Retorna None em caso de falha.
    """
    if len(coords) < 2:
        return None
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_BASE}/route/v1/driving/{coord_str}?overview=false&annotations=false"
    try:
        r = requests.get(url, timeout=OSRM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if data.get("code") == "Ok" and data.get("routes"):
            return data["routes"][0]["distance"] / 1_000.0   # metros → km
    except Exception:
        pass
    return None


def _nn_order(matrix: list, start: int = 0) -> list:
    """Nearest-neighbor TSP a partir de `start`. Retorna lista de índices na ordem ótima."""
    n = len(matrix)
    visited = [False] * n
    order   = [start]
    visited[start] = True
    for _ in range(n - 1):
        last   = order[-1]
        best_j = -1
        best_d = float("inf")
        for j in range(n):
            if not visited[j]:
                d = matrix[last][j]
                if d is not None and d < best_d:
                    best_d, best_j = d, j
        if best_j == -1:                          # vizinhos inacessíveis
            best_j = next(i for i in range(n) if not visited[i])
        order.append(best_j)
        visited[best_j] = True
    return order


def _reorder_with_osrm(
    result: pd.DataFrame,
    home_lat=None,
    home_lon=None,
) -> pd.DataFrame:
    """
    Reordena os clientes dentro de cada ROTA×DIA usando distâncias reais por estrada
    (OSRM / OpenStreetMap) com algoritmo Nearest-Neighbor.

    · Se houver endereço base, o trajeto começa da casa do vendedor (Dia 1).
    · Clientes sem coordenadas ficam no final do dia (ordem inalterada).
    · Faz fallback para a ordem atual caso o OSRM não responda.
    """
    result = result.copy()
    groups = list(result.groupby(["NOVA ROTA", "NOVO DIA"], sort=True))

    for step, ((rota, dia), grp) in enumerate(groups, 1):
        grp    = grp.sort_values("NOVA ORDEM")
        geo    = grp[(grp["LATITUDE"] != 0) & (grp["LONGITUDE"] != 0)]
        no_geo = grp[(grp["LATITUDE"] == 0) | (grp["LONGITUDE"] == 0)]

        if len(geo) < 2:
            for rank, i in enumerate(grp.index.tolist(), 1):
                result.loc[i, "NOVA ORDEM"] = rank
            continue

        raw_coords = list(zip(geo["LATITUDE"].tolist(), geo["LONGITUDE"].tolist()))
        use_home   = home_lat is not None and home_lon is not None
        all_coords = ([(home_lat, home_lon)] + raw_coords) if use_home else raw_coords

        # Respeita limite do servidor público
        if len(all_coords) > OSRM_MAX_COORDS:
            all_coords = all_coords[:OSRM_MAX_COORDS]

        matrix = _osrm_table(all_coords)

        if matrix is None:
            # API indisponível — mantém ordem atual
            ordered = geo.index.tolist() + no_geo.index.tolist()
        else:
            offset      = 1 if use_home else 0
            nn          = _nn_order(matrix, start=0)          # começa em casa ou cliente 0
            client_nn   = [o - offset for o in nn if o >= offset]
            geo_indices = geo.index.tolist()
            ordered_geo = [geo_indices[i] for i in client_nn if i < len(geo_indices)]
            ordered     = ordered_geo + no_geo.index.tolist()

        for rank, idx in enumerate(ordered, 1):
            result.loc[idx, "NOVA ORDEM"] = rank

        if step < len(groups):
            _time.sleep(0.2)   # rate-limit cortês ao servidor OSRM público

    return result.sort_values(["NOVA ROTA", "NOVO DIA", "NOVA ORDEM"]).reset_index(drop=True)


def detect_territory_conflicts(df: pd.DataFrame):
    """
    Detecta conflitos de território: CIDADE+BAIRRO ou CIDADE atendida por mais de um VENDEDOR.
    Retorna (bairro_df, cidade_df) com as ocorrências encontradas.
    """
    df_v = df[df["VENDEDOR"].astype(str).str.strip() != ""].copy()

    bairro_rows, city_rows = [], []

    # Nível bairro: mesmo CIDADE + BAIRRO, vendedores diferentes
    for (cidade, bairro), grp in df_v[df_v["BAIRRO"] != ""].groupby(["CIDADE", "BAIRRO"]):
        vc = grp.groupby("VENDEDOR").size().sort_values(ascending=False)
        if len(vc) > 1:
            bairro_rows.append({
                "CIDADE"       : cidade,
                "BAIRRO"       : bairro,
                "_vendedores"  : vc.index.tolist(),
                "Nº VENDEDORES": len(vc),
                "CLIENTES"     : int(vc.sum()),
                "DISTRIBUIÇÃO" : " | ".join(f"{v}: {c}" for v, c in vc.items()),
            })

    # Nível cidade: mesma CIDADE, vendedores diferentes
    for cidade, grp in df_v.groupby("CIDADE"):
        vc = grp.groupby("VENDEDOR").size().sort_values(ascending=False)
        if len(vc) > 1:
            city_rows.append({
                "CIDADE"       : cidade,
                "_vendedores"  : vc.index.tolist(),
                "Nº VENDEDORES": len(vc),
                "CLIENTES"     : int(vc.sum()),
                "DISTRIBUIÇÃO" : " | ".join(f"{v}: {c}" for v, c in vc.items()),
            })

    bc = (pd.DataFrame(bairro_rows).sort_values(["CIDADE", "BAIRRO"])
          if bairro_rows else
          pd.DataFrame(columns=["CIDADE", "BAIRRO", "_vendedores",
                                 "Nº VENDEDORES", "CLIENTES", "DISTRIBUIÇÃO"]))
    cc = (pd.DataFrame(city_rows).sort_values("CIDADE")
          if city_rows else
          pd.DataFrame(columns=["CIDADE", "_vendedores",
                                 "Nº VENDEDORES", "CLIENTES", "DISTRIBUIÇÃO"]))
    return bc.reset_index(drop=True), cc.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core routing algorithm
# ─────────────────────────────────────────────────────────────────────────────
def assign_new_routes(
    df_input: pd.DataFrame,
    home_lat=None,
    home_lon=None,
    use_osrm: bool = False,
) -> pd.DataFrame:
    """
    Distributes clients across 4 ROTAs × 5 working days (= 20 dias úteis).

    Sorting:
      · Clients with valid lat/lon are ordered by actual coordinates.
      · Clients with lat/lon = 0 use the city centroid for geographic placement
        and fall back to CIDADE → BAIRRO → ENDEREÇO as tiebreaker.

    Visit rules:
      · 1 VISITA  → client appears in one ROTA only.
      · 2 VISITAS → client appears twice:
            first half  → ROTA 01  +  ROTA 03  (~10 working days apart)
            second half → ROTA 02  +  ROTA 04
    """
    df = add_geo_sort_keys(df_input)

    # Hierarchy: cidade (centroid) → bairro (centroid) → proximidade (coords) → endereço (texto)
    SORT_COLS = [
        "_city_lat", "_city_lon", "CIDADE",
        "_bairro_lat", "_bairro_lon", "BAIRRO",
        "_sort_lat", "_sort_lon", "ENDEREÇO", "CLIENTE",
    ]
    df = df.sort_values(SORT_COLS).reset_index(drop=True)

    is_2v = df[COL_VISITAS].str.contains("2")
    df_1v = df[~is_2v].reset_index(drop=True)
    df_2v = df[is_2v].reset_index(drop=True)

    n_1v, n_2v = len(df_1v), len(df_2v)

    # Split 2-visit clients into two geographic halves
    split = (n_2v + 1) // 2
    df_2v_a = df_2v.iloc[:split].copy()    # → ROTA 01 & ROTA 03
    df_2v_b = df_2v.iloc[split:].copy()    # → ROTA 02 & ROTA 04

    # Split 1-visit clients into 4 sequential geographic groups
    base, rem = divmod(n_1v, 4)
    cuts_1v, start = [], 0
    for i in range(4):
        size = base + (1 if i < rem else 0)
        cuts_1v.append(df_1v.iloc[start: start + size].copy())
        start += size

    route_specs = [
        (1, cuts_1v[0], df_2v_a),
        (2, cuts_1v[1], df_2v_b),
        (3, cuts_1v[2], df_2v_a),
        (4, cuts_1v[3], df_2v_b),
    ]

    output_rows = []
    n_days = 5

    for rota_num, clients_1v, clients_2v in route_specs:
        rota_label = f"ROTA {rota_num:02d}"

        rota_df = (
            pd.concat([clients_1v, clients_2v], ignore_index=True)
            .sort_values(SORT_COLS)
            .reset_index(drop=True)
        )
        n = len(rota_df)

        # ── Day assignment: city-aware + home-aware ────────────────────────────────
        #
        # City blocks are processed in geographic order (or in home-distance order
        # when a home address is provided).  A single sequential pass fills days
        # left-to-right, so each city occupies consecutive days and cities are
        # never interleaved.
        #
        #  · Small city (≤ CITY_MAX_SINGLE_DAY): all clients land on the same day.
        #  · Large city (> CITY_MAX_SINGLE_DAY): fills days sequentially.
        # ─────────────────────────────────────────────────────────────────────

        # Build ordered city blocks in current sort order
        city_blocks: list[tuple[str, list[int]]] = []
        seen_cities: set[str] = set()
        for i in range(n):
            city = rota_df.iloc[i]["CIDADE"]
            if city not in seen_cities:
                positions = rota_df.index[rota_df["CIDADE"] == city].tolist()
                city_blocks.append((city, positions))
                seen_cities.add(city)

        # Re-ordena cidades minimizando deslocamento inter-cidade
        # ─ OSRM ativo: Nearest-Neighbor TSP sobre distâncias reais por estrada entre centroides
        # ─ só endereço base: Haversine (linha reta) da casa para cada cidade
        # ─ nenhum: mantém ordenação geográfica dos centroides (SORT_COLS)

        def _centroid(positions: list):
            """Retorna (lat, lon) mediano dos clientes geocodificados da cidade."""
            rows  = rota_df.iloc[positions]
            valid = rows[(rows["LATITUDE"] != 0) & (rows["LONGITUDE"] != 0)]
            if valid.empty:
                return None
            return (float(valid["LATITUDE"].median()), float(valid["LONGITUDE"].median()))

        centroids = [_centroid(pos) for _, pos in city_blocks]

        if use_osrm and len(city_blocks) >= 2:
            # ─ Nível 1: OSRM entre centroides de cidade (considera rodovias/hierarquia)
            valid_pairs = [(i, c) for i, c in enumerate(centroids) if c is not None]
            if len(valid_pairs) >= 2:
                use_home_c = home_lat is not None and home_lon is not None
                osrm_c     = ([(home_lat, home_lon)] + [c for _, c in valid_pairs]
                              if use_home_c else [c for _, c in valid_pairs])

                if len(osrm_c) <= OSRM_MAX_COORDS:
                    city_mat = _osrm_table(osrm_c)
                    if city_mat is not None:
                        offset      = 1 if use_home_c else 0
                        nn_cities   = _nn_order(city_mat, start=0)
                        nn_cities   = [o - offset for o in nn_cities if o >= offset]
                        valid_idx   = [i for i, _ in valid_pairs]
                        ordered     = [valid_idx[o] for o in nn_cities if o < len(valid_idx)]
                        missing     = [i for i in range(len(city_blocks)) if i not in set(ordered)]
                        city_blocks = [city_blocks[i] for i in ordered + missing]
                        _time.sleep(0.2)   # rate-limit cortês (chamada por ROTA)

        elif home_lat is not None and home_lon is not None:
            # Haversine fallback (OSRM desativado): cidade mais próxima de casa primeiro
            def _dist_home_hav(positions: list) -> float:
                c = _centroid(positions)
                return _haversine_km(home_lat, home_lon, c[0], c[1]) if c else float("inf")
            city_blocks.sort(key=lambda cb: _dist_home_hav(cb[1]))

        # Balanced per-day capacity
        base_pd, rem_pd = divmod(n, n_days)
        day_cap = [base_pd + (1 if d < rem_pd else 0) for d in range(n_days)]

        day_loads      = [0] * n_days
        day_assignment = [0] * n
        cur_day        = 0

        for city, positions in city_blocks:
            count = len(positions)
            if count <= CITY_MAX_SINGLE_DAY:
                # Small city — keep all on cur_day (single-day block)
                for pos in positions:
                    day_assignment[pos] = cur_day + 1
                    day_loads[cur_day] += 1
                # Advance once the day reaches capacity
                while cur_day < n_days - 1 and day_loads[cur_day] >= day_cap[cur_day]:
                    cur_day += 1
            else:
                # Large city — fill sequentially across days
                for pos in positions:
                    day_assignment[pos] = cur_day + 1
                    day_loads[cur_day] += 1
                    while cur_day < n_days - 1 and day_loads[cur_day] >= day_cap[cur_day]:
                        cur_day += 1
        # ─────────────────────────────────────────────────────────────────────

        day_order: dict[int, int] = {}
        for idx in range(n):
            row = rota_df.iloc[idx]
            day = day_assignment[idx]
            day_order[day] = day_order.get(day, 0) + 1

            out = row.to_dict()
            out["NOVA ROTA"] = rota_label
            out["NOVO DIA"] = day
            out["DIA SEMANA"] = DIAS_SEMANA[day]
            out["NOVA ORDEM"] = day_order[day]
            output_rows.append(out)

    result = (
        pd.DataFrame(output_rows)
        .sort_values(["NOVA ROTA", "NOVO DIA", "NOVA ORDEM"])
        .reset_index(drop=True)
    )

    result.drop(
        columns=["_sort_lat", "_sort_lon", "_city_lat", "_city_lon",
                 "_bairro_lat", "_bairro_lon"],
        inplace=True, errors="ignore",
    )

    if use_osrm:
        result = _reorder_with_osrm(result, home_lat=home_lat, home_lon=home_lon)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Excel export
# ─────────────────────────────────────────────────────────────────────────────
def to_excel(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Nova Rota")
        ws = writer.sheets["Nova Rota"]
        for col in ws.columns:
            max_len = max(
                (len(str(cell.value)) if cell.value is not None else 0)
                for cell in col
            )
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 60)
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.title("📍 Reorganização de Rotas")
st.caption(
    "Redistribui o roteiro em **4 ROTAS × 5 dias úteis = 20 dias úteis**. "
    "Clientes com **2 VISITAS** recebem ROTA 01 + ROTA 03  ou  ROTA 02 + ROTA 04. "
    f"Cidades com até **{CITY_MAX_SINGLE_DAY} clientes** na rota são mantidas em um único dia. "
    "Ordenação: **Cidade → Bairro → Endereço → Proximidade (coords)**; "
    "sem coordenadas usa centroide do bairro/cidade como fallback."
)

df = load_data()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filtros")

    vendedores = sorted(df["VENDEDOR"].unique().tolist())
    vendedor_sel = st.selectbox(
        "Vendedor / Rota:",
        options=["— Todos —"] + vendedores,
        index=0,
    )

    st.markdown("---")

    acoes_disponiveis = sorted(df["AÇÃO"].unique().tolist())
    acoes_selecionadas = st.multiselect(
        "Incluir registros com AÇÃO:",
        options=acoes_disponiveis,
        default=[a for a in acoes_disponiveis if a != "NÃO CONTABILIZADOS"],
    )

    st.markdown("---")
    st.markdown("**Regra 2 VISITAS**")
    st.markdown(
        "- 1ª visita → ROTA 01 → 2ª visita → ROTA 03\n"
        "- 1ª visita → ROTA 02 → 2ª visita → ROTA 04"
    )

    st.markdown("---")
    df_sidebar = df if vendedor_sel == "— Todos —" else df[df["VENDEDOR"] == vendedor_sel]
    total_s = len(df_sidebar)
    geocoded_s = int(((df_sidebar["LATITUDE"] != 0) & (df_sidebar["LONGITUDE"] != 0)).sum())
    st.metric("Registros", total_s)
    pct = f"{geocoded_s/total_s*100:.0f}%" if total_s else "—"
    st.metric("Com coordenadas", f"{geocoded_s} ({pct})")
    st.metric("Sem coordenadas (fallback)", total_s - geocoded_s)

    st.markdown("---")
    _ha = load_home_addresses()
    if vendedor_sel != "— Todos —":
        if vendedor_sel in _ha:
            _hl, _hlo = _ha[vendedor_sel]
            st.success("🏠 Endereço base carregado")
            st.caption(f"Lat {_hl:.5f} · Lon {_hlo:.5f}")
        else:
            st.warning("🏠 Sem endereço base\nAdicione em **endereco.csv**")
    elif _ha:
        st.info(f"🏠 {len(_ha)} endereço(s) base carregado(s)")
    else:
        st.caption("📄 endereco.csv não encontrado")


# ── apply filters ─────────────────────────────────────────────────────────────
df_work = df.copy()
if vendedor_sel != "— Todos —":
    df_work = df_work[df_work["VENDEDOR"] == vendedor_sel]
if acoes_selecionadas:
    df_work = df_work[df_work["AÇÃO"].isin(acoes_selecionadas)]


# ── tabs ───────────────────────────────────────────────────────────────────────
tab_orig, tab_nova, tab_resumo, tab_mapa = st.tabs(
    ["📄 Dados Originais", "🗺️ Nova Rota", "📊 Resumo", "🗺️ Mapa"]
)

with tab_orig:
    st.dataframe(df_work.reset_index(drop=True), use_container_width=True, height=500)
    vc = df_work[COL_VISITAS].value_counts().to_dict()
    st.caption(f"{len(df_work)} registros  ·  {vc}")

with tab_nova:
    # ── Verificação de conflitos de território ──────────────────────────────────────
    # Roda sempre na base completa (df), independente do filtro de vendedor
    _bc_all, _cc_all = detect_territory_conflicts(df)

    # Filtra pelo vendedor selecionado (quando aplicável)
    def _filter_conflicts(cdf, vendedor):
        if cdf.empty or vendedor == "— Todos —":
            return cdf
        mask = cdf["_vendedores"].apply(lambda vs: vendedor in vs)
        return cdf[mask].reset_index(drop=True)

    _bc = _filter_conflicts(_bc_all, vendedor_sel)
    _cc = _filter_conflicts(_cc_all, vendedor_sel)
    _n_b, _n_c = len(_bc), len(_cc)

    if _n_b == 0 and _n_c == 0:
        st.success(
            "✅ Sem conflitos de território"
            + (f" para **{vendedor_sel}**" if vendedor_sel != "— Todos —" else "") + "."
        )
    else:
        _parts = []
        if _n_c:
            _parts.append(f"{_n_c} cidade(s) compartilhada(s)")
        if _n_b:
            _parts.append(f"{_n_b} bairro(s) compartilhado(s)")
        with st.expander(
            f"⚠️ Conflitos de território detectados: {' · '.join(_parts)}",
            expanded=True,
        ):
            if _n_c:
                st.markdown("**Cidades atendidas por mais de um vendedor**")
                st.dataframe(
                    _cc.drop(columns=["_vendedores"], errors="ignore"),
                    use_container_width=True, hide_index=True,
                )
            if _n_b:
                st.markdown("**Bairros atendidos por mais de um vendedor**")
                st.dataframe(
                    _bc.drop(columns=["_vendedores"], errors="ignore"),
                    use_container_width=True, hide_index=True, height=280,
                )
            st.info(
                "💡 Redistribua os clientes desses bairros/cidades para um único vendedor "
                "antes de gerar os roteiros, evitando deslocamentos sobrepostos."
            )

    st.divider()
    # ── Geração de rota ────────────────────────────────────────────────────────
    _c1, _c2 = st.columns([1, 2])
    with _c1:
        use_osrm_chk = st.checkbox(
            "🛣️ Otimizar por estradas reais",
            value=True,
            key="use_osrm",
            help=(
                "Ativa otimização em dois níveis usando OpenStreetMap / OSRM:\n\n"
                "**Nível 1 — Ordem das cidades:** reordena as cidades de cada rota "
                "usando distâncias reais por estrada entre seus centroides (rodovias "
                "federais, estaduais, vias arteriais), evitando grandes deslocamentos "
                "no meio do dia.\n\n"
                "**Nível 2 — Ordem dos clientes:** dentro de cada dia, reordena os "
                "clientes com Nearest-Neighbor TSP sobre tempo de viagem por estrada, "
                "respeitando hierarquia de vias.\n\n"
                "Requer conexão com a internet. Pode adicionar 15-60 s ao cálculo."
            ),
        )
    if st.button("🔄 Gerar Nova Rota", type="primary"):
        if df_work.empty:
            st.warning("Nenhum registro com os filtros selecionados.")
        else:
            _home_addrs = load_home_addresses()
            _home       = _home_addrs.get(vendedor_sel) if vendedor_sel != "— Todos —" else None
            _hlat       = _home[0] if _home else None
            _hlon       = _home[1] if _home else None
            _spinner_msg = (
                "🛣️ Calculando roteiro e otimizando por estradas reais (OSRM)..."
                if use_osrm_chk else
                "Calculando roteiro..."
            )
            with st.spinner(_spinner_msg):
                result = assign_new_routes(
                    df_work,
                    home_lat=_hlat,
                    home_lon=_hlon,
                    use_osrm=use_osrm_chk,
                )
            st.session_state["result"]          = result
            st.session_state["result_vendedor"] = vendedor_sel
            st.session_state["home_lat"]        = _hlat
            st.session_state["home_lon"]        = _hlon
            # Limpa cache de distâncias do mapa (novo roteiro)
            for _k in [k for k in list(st.session_state) if k.startswith("_dist_")]:
                del st.session_state[_k]

    if "result" in st.session_state:
        result = st.session_state["result"]
        lbl = st.session_state.get("result_vendedor", "")
        if lbl and lbl != "— Todos —":
            st.info(f"Roteiro gerado para: **{lbl}**")

        clients_2v = result[result[COL_VISITAS].str.contains("2")]["CLIENTE"].nunique()
        st.success(
            f"✅ {len(result)} visitas distribuídas em 4 ROTAs × 5 dias  ·  "
            f"{clients_2v} clientes com 2 visitas"
        )

        priority_cols = [
            "NOVA ROTA", "NOVO DIA", "DIA SEMANA", "NOVA ORDEM",
            "CLIENTE", "VENDEDOR", "CIDADE", "BAIRRO", "ENDEREÇO",
            "LATITUDE", "LONGITUDE",
            COL_VISITAS, "AÇÃO",
            "MED QTDE PED", "MED R$", "REGIÃO",
            "ROTA HOJE", "DIA HOJE", "ORDEM HOJE", "ROTA ORIGINAL",
        ]
        cols_display = [c for c in priority_cols if c in result.columns]
        cols_display += [c for c in result.columns if c not in cols_display]

        st.dataframe(result[cols_display], use_container_width=True, height=500)

        excel_bytes = to_excel(result[cols_display])
        fname = (
            f"nova_rota_{vendedor_sel.split(' - ')[0].strip()}.xlsx"
            if vendedor_sel != "— Todos —"
            else "nova_rota_todos.xlsx"
        )
        st.download_button(
            label="📥 Exportar Excel",
            data=excel_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Selecione um vendedor (opcional), ajuste os filtros e clique em **Gerar Nova Rota**.")

with tab_resumo:
    if "result" in st.session_state:
        result = st.session_state["result"]

        st.subheader("Visitas por Rota × Dia")
        summary = (
            result.groupby(["NOVA ROTA", "NOVO DIA", "DIA SEMANA"])
            .size()
            .reset_index(name="QTD")
        )
        pivot = (
            summary.pivot_table(
                index=["NOVO DIA", "DIA SEMANA"],
                columns="NOVA ROTA",
                values="QTD",
                aggfunc="sum",
            )
            .fillna(0)
            .astype(int)
        )
        pivot.columns.name = None
        total_row = pivot.sum().to_frame().T
        total_row.index = pd.MultiIndex.from_tuples([(0, "TOTAL")])
        pivot = pd.concat([pivot, total_row])
        pivot = pivot.reset_index()
        pivot.columns = ["DIA", "DIA SEMANA"] + [c for c in pivot.columns[2:]]
        st.dataframe(pivot, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Clientes com 2 Visitas — Verificação de Pareamento")
        df_2v_chk = result[result[COL_VISITAS].str.contains("2")][
            ["CLIENTE", "CIDADE", "BAIRRO", "LATITUDE", "LONGITUDE",
             "NOVA ROTA", "NOVO DIA", "DIA SEMANA"]
        ]
        par = (
            df_2v_chk.groupby("CLIENTE")
            .apply(
                lambda g: pd.Series({
                    "CIDADE": g["CIDADE"].iloc[0],
                    "BAIRRO": g["BAIRRO"].iloc[0],
                    "LATITUDE": g["LATITUDE"].iloc[0],
                    "LONGITUDE": g["LONGITUDE"].iloc[0],
                    "1ª ROTA": g.sort_values("NOVA ROTA")["NOVA ROTA"].iloc[0],
                    "1º DIA": g.sort_values("NOVA ROTA")["DIA SEMANA"].iloc[0],
                    "2ª ROTA": g.sort_values("NOVA ROTA")["NOVA ROTA"].iloc[-1],
                    "2º DIA": g.sort_values("NOVA ROTA")["DIA SEMANA"].iloc[-1],
                }),
                include_groups=False,
            )
            .reset_index()
        )
        st.dataframe(par, use_container_width=True, height=400)

        st.markdown("---")
        st.subheader("Clientes sem Coordenadas (lat/lon = 0)")
        no_coords = result[
            (result["LATITUDE"] == 0) | (result["LONGITUDE"] == 0)
        ][["CLIENTE", "CIDADE", "BAIRRO", "ENDEREÇO",
           "NOVA ROTA", "NOVO DIA", "DIA SEMANA", "NOVA ORDEM"]]
        if no_coords.empty:
            st.success("Todos os clientes possuem coordenadas.")
        else:
            st.warning(
                f"{no_coords['CLIENTE'].nunique()} clientes sem coordenadas "
                "— posicionados pelo centroide da cidade."
            )
            st.dataframe(no_coords.reset_index(drop=True), use_container_width=True, height=300)
    else:
        st.info("Gere a nova rota na aba **🗺️ Nova Rota** para ver o resumo.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab — Mapa
# ─────────────────────────────────────────────────────────────────────────────
ROTA_COLORS = {
    "ROTA 01": "#e63946",
    "ROTA 02": "#2a9d8f",
    "ROTA 03": "#e9c46a",
    "ROTA 04": "#457b9d",
}
DIA_COLORS = ["#e63946", "#2a9d8f", "#f4a261", "#457b9d", "#8338ec"]

with tab_mapa:
    if "result" not in st.session_state:
        st.info("Gere a nova rota na aba **🗺️ Nova Rota** para visualizar o mapa.")
    else:
        result = st.session_state["result"]

        rotas_disp = sorted(result["NOVA ROTA"].unique().tolist())
        dias_disp  = sorted(result["NOVO DIA"].unique().tolist())

        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
        with col_f1:
            rota_map = st.selectbox(
                "ROTA:",
                options=["— Todas —"] + rotas_disp,
                key="map_rota",
            )
        with col_f2:
            dia_map = st.selectbox(
                "Dia:",
                options=[0] + dias_disp,
                format_func=lambda d: "— Todos —" if d == 0 else f"{d} — {DIAS_SEMANA[d]}",
                key="map_dia",
            )
        with col_f3:
            show_lines = st.checkbox("Mostrar trajeto", value=True, key="map_lines")

        # ── filter ────────────────────────────────────────────────────────
        df_map = result.copy()
        if rota_map != "— Todas —":
            df_map = df_map[df_map["NOVA ROTA"] == rota_map]
        if dia_map != 0:
            df_map = df_map[df_map["NOVO DIA"] == dia_map]
        df_map = df_map.sort_values(["NOVA ROTA", "NOVO DIA", "NOVA ORDEM"]).reset_index(drop=True)

        if df_map.empty:
            st.warning("Nenhum cliente encontrado para os filtros selecionados.")
        else:
            df_geo   = df_map[(df_map["LATITUDE"] != 0) & (df_map["LONGITUDE"] != 0)]
            df_nogeo = df_map[(df_map["LATITUDE"] == 0) | (df_map["LONGITUDE"] == 0)]

            # Home coordinates from session state
            home_lat_m = st.session_state.get("home_lat")
            home_lon_m = st.session_state.get("home_lon")

            # ── métricas: distância por trajeto real (OSRM) ───────────────────
            # Resultado é cacheado em session_state para não recalcular a cada
            # re-render do Streamlit.  Limpo automaticamente quando nova rotaé gerada.
            _dist_key = f"_dist_{rota_map}_{dia_map}_{len(df_geo)}"

            if _dist_key not in st.session_state:
                _route_km  = 0.0
                _home_dist = 0.0
                _used_osrm = True

                # Distância dos trajetos de cada grupo ROTA×DIA
                for (_, __), grp in df_geo.groupby(["NOVA ROTA", "NOVO DIA"]):
                    pts = list(zip(
                        grp.sort_values("NOVA ORDEM")["LATITUDE"].tolist(),
                        grp.sort_values("NOVA ORDEM")["LONGITUDE"].tolist(),
                    ))
                    if len(pts) >= 2:
                        d = _osrm_route_km(pts)
                        if d is not None:
                            _route_km += d
                        else:
                            _used_osrm = False
                            for i in range(len(pts) - 1):
                                _route_km += _haversine_km(
                                    pts[i][0], pts[i][1],
                                    pts[i + 1][0], pts[i + 1][1],
                                )

                # Distância casa → primeiro cliente geocodificado
                if home_lat_m is not None and home_lon_m is not None and not df_geo.empty:
                    _first = df_geo.sort_values(["NOVO DIA", "NOVA ORDEM"]).iloc[0]
                    d_home = _osrm_route_km(
                        [(home_lat_m, home_lon_m),
                         (_first["LATITUDE"], _first["LONGITUDE"])]
                    )
                    if d_home is not None:
                        _home_dist = d_home
                    else:
                        _used_osrm = False
                        _home_dist = _haversine_km(
                            home_lat_m, home_lon_m,
                            _first["LATITUDE"], _first["LONGITUDE"],
                        )

                st.session_state[_dist_key] = (_route_km, _home_dist, _used_osrm)

            total_km, home_dist_km, _osrm_dist_ok = st.session_state[_dist_key]
            _dist_lbl = "por trajeto real" if _osrm_dist_ok else "linha reta \u26a0️"

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Clientes", len(df_map))
            m2.metric("Com coordenadas", len(df_geo))
            m3.metric("Sem coordenadas", len(df_nogeo))
            if home_lat_m is not None:
                m4.metric(
                    f"Distância ({_dist_lbl})",
                    f"{total_km + home_dist_km:.1f} km",
                    delta=f"+{home_dist_km:.1f} km saída",
                    delta_color="off",
                )
            else:
                m4.metric(f"Distância ({_dist_lbl})", f"{total_km:.1f} km")

            # ── build figure ──────────────────────────────────────────────
            center_lat = df_geo["LATITUDE"].mean()  if not df_geo.empty else -23.5
            center_lon = df_geo["LONGITUDE"].mean() if not df_geo.empty else -46.6

            fig = go.Figure()

            # Home marker — ponto de saída
            if home_lat_m is not None and home_lon_m is not None:
                lbl_v = st.session_state.get("result_vendedor", "Vendedor")
                fig.add_trace(go.Scattermapbox(
                    lat=[home_lat_m],
                    lon=[home_lon_m],
                    mode="markers+text",
                    marker=dict(size=22, color="#111111"),
                    text=["🏠"],
                    textposition="top right",
                    textfont=dict(size=13),
                    name="🏠 Ponto de saída",
                    hovertemplate=(
                        "<b>🏠 Ponto de saída</b><br>"
                        f"Vendedor: {lbl_v}<br>"
                        f"Dist. ao 1º cliente: {home_dist_km:.1f} km"
                        "<extra></extra>"
                    ),
                ))

            groups = df_geo.groupby(["NOVA ROTA", "NOVO DIA"])
            color_idx = 0
            for (rota_k, dia_k), grp in groups:
                grp = grp.sort_values("NOVA ORDEM").reset_index(drop=True)
                color = ROTA_COLORS.get(rota_k, DIA_COLORS[color_idx % len(DIA_COLORS)])
                color_idx += 1
                label = f"{rota_k} · {dia_k} {DIAS_SEMANA.get(dia_k, '')}"

                lats = grp["LATITUDE"].tolist()
                lons = grp["LONGITUDE"].tolist()

                # Route line
                if show_lines and len(grp) > 1:
                    fig.add_trace(go.Scattermapbox(
                        lat=lats,
                        lon=lons,
                        mode="lines",
                        line=dict(width=2, color=color),
                        name=label,
                        legendgroup=label,
                        showlegend=False,
                        hoverinfo="skip",
                    ))

                # Client markers
                extra_cols = [c for c in ["MED R$", "MED QTDE PED"] if c in grp.columns]
                hover_extra = "".join(
                    f"<br>{c}: %{{customdata[{4 + i}]}}"
                    for i, c in enumerate(extra_cols)
                )
                customdata_cols = ["CLIENTE", "CIDADE", "BAIRRO", "ENDEREÇO", "NOVA ORDEM"] + extra_cols
                fig.add_trace(go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="markers+text",
                    marker=dict(size=13, color=color, opacity=0.85),
                    text=grp["NOVA ORDEM"].astype(str).tolist(),
                    textfont=dict(size=9, color="white"),
                    customdata=grp[customdata_cols].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]} — %{customdata[2]}<br>"
                        "%{customdata[3]}<br>"
                        "Ordem: <b>%{customdata[4]}</b>"
                        + hover_extra
                        + "<extra>" + label + "</extra>"
                    ),
                    name=label,
                    legendgroup=label,
                ))

            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=11,
                ),
                margin=dict(l=0, r=0, t=10, b=0),
                height=580,
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.01,
                    xanchor="left",  x=0,
                    bgcolor="rgba(255,255,255,0.8)",
                ),
                modebar=dict(
                    add=["zoomInMapbox", "zoomOutMapbox", "resetViewMapbox"],
                ),
            )

            st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

            # ── clients without coords ────────────────────────────────────
            if not df_nogeo.empty:
                with st.expander(f"⚠️ {len(df_nogeo)} clientes sem coordenadas — não plotados"):
                    show_cols = [c for c in
                        ["NOVA ROTA", "NOVO DIA", "DIA SEMANA", "NOVA ORDEM",
                         "CLIENTE", "CIDADE", "BAIRRO", "ENDEREÇO"]
                        if c in df_nogeo.columns]
                    st.dataframe(
                        df_nogeo[show_cols].reset_index(drop=True),
                        use_container_width=True,
                    )
