import streamlit as st
import pandas as pd
import os
import unicodedata
from datetime import timedelta

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="DASH-SBN | Analytics", page_icon="📉", layout="wide")

# ---------------------------------------------------------
# UTILITAIRES
# ---------------------------------------------------------
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return ascii_text.lower().strip()

# ---------------------------------------------------------
# CHARGEMENT
# ---------------------------------------------------------
@st.cache_data
def load_pipelines():
    pipelines_norm = {}
    am_files = {'NAJWA': 'NAJWA.csv', 'HOUDA': 'HOUDA.csv', 'CHAIMA': 'CHAIMA.csv'}
    for am, filename in am_files.items():
        if os.path.exists(filename):
            try:
                try:
                    df = pd.read_csv(filename, sep=',')
                    if len(df.columns)<2: raise ValueError
                except:
                    df = pd.read_csv(filename, sep=';')
                
                df.columns = df.columns.str.strip().str.lower()
                col = next((c for c in df.columns if 'restaurant' in c or 'name' in c), df.columns[0])
                pipelines_norm[am] = [normalize_text(x) for x in df[col].dropna().astype(str).tolist()]
            except: pipelines_norm[am] = []
        else: pipelines_norm[am] = []
    return pipelines_norm

@st.cache_data
def load_data(main_file, pipelines_norm):
    if hasattr(main_file, 'seek'): main_file.seek(0)
    try:
        df = pd.read_csv(main_file, sep=',')
        if 'order day' not in df.columns and len(df.columns) < 5: raise ValueError
    except:
        if hasattr(main_file, 'seek'): main_file.seek(0)
        df = pd.read_csv(main_file, sep=';')

    df.columns = df.columns.str.strip()
    
    # Dates
    df['order day'] = df['order day'].astype(str)
    df['order time'] = df['order time'].astype(str)
    
    def parse_dt(d_str):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
            try: return pd.to_datetime(d_str, format=fmt)
            except: continue
        return pd.to_datetime(d_str, errors='coerce')

    df['order_date_obj'] = df['order day'].apply(parse_dt)
    df['order_datetime'] = pd.to_datetime(
        df['order_date_obj'].dt.strftime('%Y-%m-%d') + ' ' + df['order time'], errors='coerce'
    )
    df['date'] = df['order_datetime'].dt.date
    df['year_month'] = df['order_datetime'].dt.to_period('M')

    # Numérique
    for c in ['item total', 'delivery amount']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Attribution AM
    df['restaurant_norm'] = df['restaurant name'].apply(normalize_text)
    
    def get_am(row):
        r_norm = row['restaurant_norm']
        city = str(row.get('city', '')).lower()
        r_raw = str(row.get('restaurant name', '')).lower()
        
        for am, p_list in pipelines_norm.items():
            if r_norm in p_list: return am
            for p in p_list:
                if len(p)>3 and p in r_norm: return am
        
        if any(x in r_raw for x in ['mcdonald', 'kfc', 'burger king', 'primos', 'papa john']): return 'NAJWA'
        if any(c in city for c in ['rabat', 'sale', 'temara', 'kenitra']): return 'HOUDA'
        return 'CHAIMA'

    df['AM'] = df.apply(get_am, axis=1)

    # Automatisation
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else: df['is_automated'] = False
    
    # Enseigne
    def get_brand(name):
        n = normalize_text(name)
        if 'mcdonald' in n: return "McDonald's"
        if 'kfc' in n: return "KFC"
        if 'burger king' in n: return "Burger King"
        if 'primos' in n: return "Primos"
        return "Autres"
    df['Enseigne_Groupe'] = df['restaurant name'].apply(get_brand)

    return df

# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Monitoring")
pipelines_norm = load_pipelines()

with st.sidebar:
    st.header("📂 Data")
    uploaded_file = st.file_uploader("Fichier CSV", type=['csv'])
    
    # Auto-load local pour faciliter les tests
    if not uploaded_file:
        local = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"
        # Si vous voulez tester en local sans upload, décommentez la ligne suivante
        # if os.path.exists(local): uploaded_file = local 
        pass

    if uploaded_file:
        df = load_data(uploaded_file, pipelines_norm)
    else:
        st.info("Chargez le fichier.")
        st.stop()
        
    st.divider()
    st.header("🔍 Filtres")
    
    # --- INTELLIGENCE DATE ---
    last_date = df['date'].max()
    first_date_of_month = last_date.replace(day=1)
    
    date_range = st.date_input("Période", [first_date_of_month, last_date])
    
    all_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
    sel_brand = st.selectbox("Enseigne", all_brands)

# FILTRE
mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if sel_brand != 'Tous': mask &= (df['Enseigne_Groupe'] == sel_brand)
df_filtered = df.loc[mask]

if df_filtered.empty:
    st.warning("Aucune donnée sur cette période.")
    st.stop()

# ---------------------------------------------------------
# 1. KPI GLOBAL (AM)
# ---------------------------------------------------------
st.subheader("📊 Performance par AM")
# Comparaison Période (Même durée avant)
delta_days = (date_range[1] - date_range[0]).days + 1
prev_start = date_range[0] - timedelta(days=delta_days)
prev_end = date_range[0] - timedelta(days=1)
df_prev = df[(df['date'] >= prev_start) & (df['date'] <= prev_end)]

summary = []
for am in ['NAJWA', 'HOUDA', 'CHAIMA']:
    d_am = df_filtered[df_filtered['AM']==am]
    rev = d_am['item total'].sum()
    orders = len(d_am)
    
    # Growth
    rev_prev = df_prev[df_prev['AM']==am]['item total'].sum()
    growth = ((rev - rev_prev)/rev_prev*100) if rev_prev > 0 else 0
    
    # Inactifs
    pipe = pipelines_norm.get(am, [])
    active = d_am['restaurant_norm'].unique().tolist()
    # Logique Inactifs
    matched = 0
    for p in pipe:
        if any(p in a for a in active): matched += 1
    inact = max(0, len(pipe) - matched)

    summary.append({
        "AM": am, "CA": rev, "Commandes": orders, "Growth %": growth, 
        "Pipeline": len(pipe), "Inactifs": inact
    })

# Ici on utilise un dictionnaire de formatage, donc pas de problème pour les colonnes texte
st.dataframe(pd.DataFrame(summary).style.format({"CA":"{:,.0f}","Growth %":"{:+.1f}%"}).background_gradient(subset=['Growth %'], cmap="Greens"), use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 2. FLOP AUTOMATIQUE (MOIS M vs MOIS M-1)
# ---------------------------------------------------------
st.subheader("📉 Top Flops (Comparaison Automatique Derniers Mois)")

all_months = sorted(df['year_month'].unique())
if len(all_months) >= 2:
    last_month = all_months[-1]
    prev_month = all_months[-2]
    
    col_info, col_table = st.columns([1, 3])
    with col_info:
        st.info(f"Comparaison de **{last_month}** par rapport à **{prev_month}**")

    with col_table:
        df_m = df[df['year_month'] == last_month]
        df_m_1 = df[df['year_month'] == prev_month]
        
        if sel_brand != 'Tous':
            df_m = df_m[df_m['Enseigne_Groupe'] == sel_brand]
            df_m_1 = df_m_1[df_m_1['Enseigne_Groupe'] == sel_brand]

        stats_m = df_m.groupby('restaurant name')['order id'].count()
        stats_m_1 = df_m_1.groupby('restaurant name')['order id'].count()
        
        flop_auto = pd.DataFrame({'Mois Préc': stats_m_1, 'Mois Actuel': stats_m}).fillna(0)
        flop_auto['Perte'] = flop_auto['Mois Actuel'] - flop_auto['Mois Préc']
        
        map_am = df.drop_duplicates('restaurant name').set_index('restaurant name')['AM'].to_dict()
        flop_auto['AM'] = flop_auto.index.map(map_am)
        
        vrais_flops = flop_auto[flop_auto['Perte'] < 0].sort_values('Perte')
        
        if not vrais_flops.empty:
            # CORRECTION ICI : Formatage spécifique par colonne
            st.dataframe(
                vrais_flops[['AM', 'Mois Préc', 'Mois Actuel', 'Perte']].style.format(
                    {'Mois Préc': "{:.0f}", 'Mois Actuel': "{:.0f}", 'Perte': "{:.0f}"}
                ).background_gradient(subset=['Perte'], cmap='Reds_r'),
                use_container_width=True
            )
        else:
            st.success("Aucune régression entre ces deux mois.")
else:
    st.warning("Pas assez d'historique.")

# ---------------------------------------------------------
# 3. RÉGRESSION PERSONNALISÉE
# ---------------------------------------------------------
st.divider()
st.subheader(f"🔍 Régression sur la période sélectionnée")

curr = df_filtered.groupby('restaurant name')['order id'].count()
if sel_brand != 'Tous': df_prev = df_prev[df_prev['Enseigne_Groupe'] == sel_brand]
prev = df_prev.groupby('restaurant name')['order id'].count()

reg_custom = pd.DataFrame({'Avant': prev, 'Pendant': curr}).fillna(0)
reg_custom['Delta'] = reg_custom['Pendant'] - reg_custom['Avant']
reg_custom['AM'] = reg_custom.index.map(map_am)

flops_custom = reg_custom[reg_custom['Delta'] < 0].sort_values('Delta')

if not flops_custom.empty:
    # CORRECTION ICI : Formatage spécifique par colonne
    st.dataframe(
        flops_custom[['AM', 'Avant', 'Pendant', 'Delta']].style.format(
            {'Avant': "{:.0f}", 'Pendant': "{:.0f}", 'Delta': "{:.0f}"}
        ).background_gradient(subset=['Delta'], cmap='Reds_r'),
        use_container_width=True
    )
else:
    st.info("Aucune régression sur cette plage de dates spécifique.")
