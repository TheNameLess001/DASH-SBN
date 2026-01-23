import streamlit as st
import pandas as pd
import os
import unicodedata
from datetime import timedelta

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="DASH-SBN | Analytics", page_icon="📊", layout="wide")

# ---------------------------------------------------------
# UTILITAIRES & CHARGEMENT
# ---------------------------------------------------------
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return ascii_text.lower().strip()

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
# INTERFACE PRINCIPALE
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Performance Analytics")
pipelines_norm = load_pipelines()

with st.sidebar:
    st.header("📂 Données")
    uploaded_file = st.file_uploader("Fichier CSV", type=['csv'])
    
    # Auto-load local (Optionnel)
    if not uploaded_file:
        local = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"
        # if os.path.exists(local): uploaded_file = local 
        pass

    if uploaded_file:
        df = load_data(uploaded_file, pipelines_norm)
    else:
        st.info("Chargez le fichier CSV.")
        st.stop()
        
    st.divider()
    st.header("🎯 Scope & Filtres")
    
    # 1. SCOPE (GLOBAL OU AM)
    scope_options = ['Global', 'NAJWA', 'HOUDA', 'CHAIMA']
    selected_scope = st.selectbox("Vue (Scope)", scope_options)
    
    # 2. FILTRE ENSEIGNE
    # On filtre les options d'enseigne selon le scope choisi pour éviter le bruit
    if selected_scope != 'Global':
        df_scope_preview = df[df['AM'] == selected_scope]
        available_brands = ['Tous'] + sorted(df_scope_preview['Enseigne_Groupe'].unique().tolist())
    else:
        available_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
        
    sel_brand = st.selectbox("Enseigne / Groupe", available_brands)
    
    # 3. DATE
    # On prend tout par défaut pour le tableau mensuel, mais on garde le datepicker pour l'analyse fine
    min_d, max_d = df['date'].min(), df['date'].max()
    st.caption(f"Données disponibles du {min_d} au {max_d}")

# ---------------------------------------------------------
# PRÉPARATION DES DONNÉES FILTRÉES
# ---------------------------------------------------------

# Filtre Scope (AM)
if selected_scope == 'Global':
    df_scope = df.copy()
    # Pipeline Global = Union de tous les pipelines
    current_pipeline = []
    for p_list in pipelines_norm.values():
        current_pipeline.extend(p_list)
    current_pipeline = list(set(current_pipeline)) # Unique
else:
    df_scope = df[df['AM'] == selected_scope]
    current_pipeline = pipelines_norm.get(selected_scope, [])

# Filtre Enseigne
if sel_brand != 'Tous':
    df_scope = df_scope[df_scope['Enseigne_Groupe'] == sel_brand]

if df_scope.empty:
    st.warning("Aucune donnée pour ce scope.")
    st.stop()

# ---------------------------------------------------------
# 1. TABLEAU KPI MENSUEL (L'élément central demandé)
# ---------------------------------------------------------
st.subheader(f"📊 Performance Mensuelle : {selected_scope}")

# Agrégation par mois
monthly_stats = df_scope.groupby('year_month').agg({
    'item total': 'sum',
    'order id': 'count',
    'is_automated': 'sum',
    'status': lambda x: (x == 'Restaurant Rejected').sum() # Compte les rejets
}).reset_index().sort_values('year_month')

monthly_stats['Mois'] = monthly_stats['year_month'].dt.strftime('%Y-%m')

# Calculs Metrics
monthly_stats['AOV'] = monthly_stats['item total'] / monthly_stats['order id']
monthly_stats['Auto %'] = (monthly_stats['is_automated'] / monthly_stats['order id'] * 100)
monthly_stats['Accept %'] = ((monthly_stats['order id'] - monthly_stats['status']) / monthly_stats['order id'] * 100) # status contient le nb de rejects ici

# Calcul Growth (MoM)
monthly_stats['CA Prev'] = monthly_stats['item total'].shift(1)
monthly_stats['Growth %'] = ((monthly_stats['item total'] - monthly_stats['CA Prev']) / monthly_stats['CA Prev'] * 100).fillna(0)

# Calcul Inactifs Mensuels
# Pour chaque mois, on regarde quels restaurants du pipeline N'ONT PAS commandé
inactifs_list = []
total_pipeline_len = len(current_pipeline)

for ym in monthly_stats['year_month']:
    # Restos actifs ce mois-ci
    actifs_this_month = df_scope[df_scope['year_month'] == ym]['restaurant_norm'].unique().tolist()
    
    # Combien du pipeline sont dedans ?
    found_count = 0
    if total_pipeline_len > 0:
        for p in current_pipeline:
            # Match flexible
            if any(p in a for a in actifs_this_month):
                found_count += 1
        
        nb_inactifs = max(0, total_pipeline_len - found_count)
    else:
        nb_inactifs = 0
    inactifs_list.append(nb_inactifs)

monthly_stats['Inactifs'] = inactifs_list
monthly_stats['Pipeline'] = total_pipeline_len

# Mise en forme Tableau
display_cols = ['Mois', 'item total', 'order id', 'AOV', 'Growth %', 'Auto %', 'Accept %', 'Inactifs']
rename_map = {
    'item total': 'CA (MAD)', 
    'order id': 'Commandes', 
    'AOV': 'Panier Moy.',
    'Accept %': 'Taux Accept.'
}

final_table = monthly_stats[display_cols].rename(columns=rename_map)

st.dataframe(
    final_table.style.format({
        "CA (MAD)": "{:,.0f}",
        "Commandes": "{:.0f}",
        "Panier Moy.": "{:.1f}",
        "Growth %": "{:+.1f}%",
        "Auto %": "{:.1f}%",
        "Taux Accept.": "{:.1f}%",
        "Inactifs": "{:.0f}"
    }).background_gradient(subset=['Growth %'], cmap="RdYlGn", vmin=-20, vmax=20),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# 2. TOP FLOP & PROGRESSION (Automatique Derniers Mois)
# ---------------------------------------------------------
st.subheader("📈 Tops & 📉 Flops (Comparaison Automatique)")

all_months = sorted(df['year_month'].unique())
if len(all_months) >= 2:
    last_month = all_months[-1] # Ex: Jan
    prev_month = all_months[-2] # Ex: Dec
    
    st.info(f"Comparaison : **{last_month}** vs **{prev_month}**")
    
    # Préparation des données pour la comparaison
    # On utilise df_scope (déjà filtré par Scope et Enseigne)
    df_m = df_scope[df_scope['year_month'] == last_month]
    df_m_1 = df_scope[df_scope['year_month'] == prev_month]
    
    stats_m = df_m.groupby('restaurant name')['order id'].count()
    stats_m_1 = df_m_1.groupby('restaurant name')['order id'].count()
    
    comp_df = pd.DataFrame({'Mois Préc': stats_m_1, 'Mois Actuel': stats_m}).fillna(0)
    comp_df['Delta'] = comp_df['Mois Actuel'] - comp_df['Mois Préc']
    
    # On ajoute la colonne AM pour info
    map_am = df.drop_duplicates('restaurant name').set_index('restaurant name')['AM'].to_dict()
    comp_df['AM'] = comp_df.index.map(map_am)

    # COLONNES
    col_flop, col_top = st.columns(2)
    
    # --- FLOP (Régression) ---
    with col_flop:
        st.markdown("### 🚨 Top Régressions")
        flops = comp_df[comp_df['Delta'] < 0].sort_values('Delta', ascending=True).head(10)
        
        if not flops.empty:
            st.dataframe(
                flops[['AM', 'Mois Préc', 'Mois Actuel', 'Delta']].style.format(
                    {'Mois Préc': "{:.0f}", 'Mois Actuel': "{:.0f}", 'Delta': "{:.0f}"}
                ).background_gradient(subset=['Delta'], cmap='Reds_r'),
                use_container_width=True
            )
        else:
            st.success("Aucune baisse significative.")

    # --- TOP (Progression) ---
    with col_top:
        st.markdown("### 🚀 Top Progressions")
        tops = comp_df[comp_df['Delta'] > 0].sort_values('Delta', ascending=False).head(10)
        
        if not tops.empty:
            st.dataframe(
                tops[['AM', 'Mois Préc', 'Mois Actuel', 'Delta']].style.format(
                    {'Mois Préc': "{:.0f}", 'Mois Actuel': "{:.0f}", 'Delta': "{:+.0f}"}
                ).background_gradient(subset=['Delta'], cmap='Greens'),
                use_container_width=True
            )
        else:
            st.info("Aucune hausse significative.")
            
else:
    st.warning("Pas assez d'historique (besoin de 2 mois min) pour calculer les Tops/Flops.")

# ---------------------------------------------------------
# 3. DONNÉES BRUTES
# ---------------------------------------------------------
with st.expander("Voir les données brutes"):
    st.dataframe(df_scope)
