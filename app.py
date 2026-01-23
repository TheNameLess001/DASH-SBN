import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import timedelta

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="DASH-SBN | Performance & Pipeline",
    page_icon="📉",
    layout="wide"
)

# ---------------------------------------------------------
# FONCTIONS DE CHARGEMENT
# ---------------------------------------------------------
@st.cache_data
def load_pipelines():
    """Charge les listes de restaurants (Pipelines) depuis les fichiers CSV AM."""
    pipelines = {}
    am_files = {'NAJWA': 'NAJWA.csv', 'HOUDA': 'HOUDA.csv', 'CHAIMA': 'CHAIMA.csv'}
    
    for am, filename in am_files.items():
        if os.path.exists(filename):
            try:
                # On suppose une colonne 'Restaurant Name' ou la première colonne
                df_p = pd.read_csv(filename, sep=None, engine='python')
                # Nettoyage des noms de colonnes
                df_p.columns = df_p.columns.str.strip().str.lower()
                
                # Trouver la colonne nom
                col_name = next((c for c in df_p.columns if 'restaurant' in c or 'name' in c), df_p.columns[0])
                
                # Stocker la liste normalisée des restaurants
                pipelines[am] = df_p[col_name].astype(str).str.strip().str.lower().tolist()
            except:
                pipelines[am] = []
        else:
            pipelines[am] = []
    return pipelines

@st.cache_data
def load_data(main_file, pipelines):
    # 1. Lecture du fichier principal (Commandes)
    if hasattr(main_file, 'seek'): main_file.seek(0)
    try:
        df = pd.read_csv(main_file, sep=',')
        if 'order day' not in df.columns and len(df.columns) < 5: raise ValueError
    except:
        if hasattr(main_file, 'seek'): main_file.seek(0)
        df = pd.read_csv(main_file, sep=';')

    df.columns = df.columns.str.strip()
    
    # 2. Parsing Dates Robuste
    df['order day'] = df['order day'].astype(str)
    df['order time'] = df['order time'].astype(str)
    
    def parse_dt(d_str):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
            try: return pd.to_datetime(d_str, format=fmt)
            except: continue
        return pd.to_datetime(d_str, errors='coerce')

    df['order_date_obj'] = df['order day'].apply(parse_dt)
    df['order_datetime'] = pd.to_datetime(
        df['order_date_obj'].dt.strftime('%Y-%m-%d') + ' ' + df['order time'], 
        errors='coerce'
    )
    df['date'] = df['order_datetime'].dt.date
    df['month_str'] = df['order_datetime'].dt.strftime('%Y-%m')

    # 3. Nettoyage Numérique
    for c in ['item total', 'delivery amount', 'Distance travel']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 4. Attribution AM (Priorité Pipeline > Logique Auto)
    def get_am(row):
        r_name = str(row.get('restaurant name', '')).lower().strip()
        
        # Check dans les pipelines chargés
        for am, resto_list in pipelines.items():
            if any(p_name in r_name for p_name in resto_list): # Match partiel
                return am
        
        # Fallback Logique
        city = str(row.get('city', '')).lower()
        if any(x in r_name for x in ['mcdonald', 'kfc', 'burger king', 'primos']): return 'NAJWA'
        if any(c in city for c in ['rabat', 'sale', 'temara', 'kenitra']): return 'HOUDA'
        return 'CHAIMA'

    df['AM'] = df.apply(get_am, axis=1)

    # 5. Autres Champs
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else:
        df['is_automated'] = False
        
    # Groupement Enseigne
    def get_brand(name):
        n = str(name).lower()
        if 'mcdonald' in n: return "McDonald's"
        if 'kfc' in n: return "KFC"
        if 'burger king' in n: return "Burger King"
        if 'chrono pizza' in n: return "Chrono Pizza"
        return "Autres"
    df['Enseigne_Groupe'] = df['restaurant name'].apply(get_brand)

    return df

# ---------------------------------------------------------
# INTERFACE & KPI
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Monitoring & Pipeline")

# CHARGEMENT
pipelines = load_pipelines()

with st.sidebar:
    st.header("Sources")
    uploaded_file = st.file_uploader("Fichier Commandes (Export Admin)", type=['csv'])
    
    # Fallback pour démo
    if not uploaded_file and os.path.exists("admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"):
       uploaded_file = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"

    if uploaded_file:
        df = load_data(uploaded_file, pipelines)
    else:
        st.info("Veuillez charger le fichier de commandes.")
        st.stop()
        
    st.divider()
    st.header("Filtres")
    
    # Select Période
    min_d, max_d = df['date'].min(), df['date'].max()
    date_range = st.date_input("Période Analysée", [min_d, max_d])
    
    # Select Enseigne
    all_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
    sel_brand = st.selectbox("Enseigne", all_brands)

# FILTRAGE DONNÉES
mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if sel_brand != 'Tous': mask &= (df['Enseigne_Groupe'] == sel_brand)
df_filtered = df.loc[mask]

# ---------------------------------------------------------
# 1. TABLEAU KPI SYNTHÉTIQUE (PAR AM)
# ---------------------------------------------------------
st.subheader("📊 Performance par Account Manager (AM)")

# Préparation des données pour le tableau
summary_data = []

ams_list = ['NAJWA', 'HOUDA', 'CHAIMA']
for am in ams_list:
    # Données filtrées pour cet AM
    data_am = df_filtered[df_filtered['AM'] == am]
    
    # 1. Metrics Base
    ca = data_am['item total'].sum()
    orders = len(data_am)
    aov = ca / orders if orders > 0 else 0
    
    # 2. Automatisation
    auto_cnt = data_am['is_automated'].sum()
    auto_rate = (auto_cnt / orders * 100) if orders > 0 else 0
    
    # 3. Taux d'Acceptation (1 - Taux de Rejet Restaurant)
    # On considère 'Restaurant Rejected' comme le seul refus impactant ce taux
    rejects = data_am[data_am['status'] == 'Restaurant Rejected'].shape[0]
    acc_rate = ((orders - rejects) / orders * 100) if orders > 0 else 100
    
    # 4. Inactifs (Pipeline vs Réalité)
    # Liste théorique (Pipeline)
    pipeline_names = pipelines.get(am, [])
    total_pipeline = len(pipeline_names)
    
    # Liste réelle (ceux qui ont commandé dans la période filtrée)
    # On normalise les noms pour la comparaison
    active_names_raw = data_am['restaurant name'].unique().tolist()
    active_names_norm = [str(x).lower().strip() for x in active_names_raw]
    
    # Compter les inactifs (Ceux du pipeline qui ne sont PAS dans les actifs)
    # Match partiel pour être gentil
    actives_count_in_pipeline = 0
    for p_name in pipeline_names:
        if any(p_name in a_name for a_name in active_names_norm):
            actives_count_in_pipeline += 1
            
    inactifs = total_pipeline - actives_count_in_pipeline
    inactifs = max(0, inactifs) # Sécurité
    
    # 5. Growth (Comparaison avec période précédente de même durée)
    # On prend une période de référence avant la date de début
    delta_days = (date_range[1] - date_range[0]).days + 1
    prev_start = date_range[0] - timedelta(days=delta_days)
    prev_end = date_range[0] - timedelta(days=1)
    
    mask_prev = (df['date'] >= prev_start) & (df['date'] <= prev_end) & (df['AM'] == am)
    prev_ca = df.loc[mask_prev, 'item total'].sum()
    
    growth = ((ca - prev_ca) / prev_ca * 100) if prev_ca > 0 else 0

    summary_data.append({
        "AM": am,
        "CA (MAD)": ca,
        "Panier Moy.": aov,
        "Commandes": orders,
        "Growth (%)": growth,
        "Auto (%)": auto_rate,
        "Accept. (%)": acc_rate,
        "Pipeline Total": total_pipeline,
        "Inactifs": inactifs
    })

# Création du DataFrame et Affichage
df_summary = pd.DataFrame(summary_data)
# Mise en forme
st.dataframe(
    df_summary.style.format({
        "CA (MAD)": "{:,.0f}",
        "Panier Moy.": "{:.1f}",
        "Growth (%)": "{:+.1f}%",
        "Auto (%)": "{:.1f}%",
        "Accept. (%)": "{:.1f}%",
        "Pipeline Total": "{:.0f}",
        "Inactifs": "{:.0f}"
    }).background_gradient(subset=['Growth (%)', 'CA (MAD)'], cmap="Greens"),
    use_container_width=True,
    hide_index=True
)

st.divider()

# ---------------------------------------------------------
# 2. ANALYSE DE RÉGRESSION (TOP FLOP)
# ---------------------------------------------------------
st.subheader("🚨 Restaurants en Régression (Volume Commandes)")
col_reg1, col_reg2 = st.columns([3, 1])

with col_reg1:
    st.markdown("Comparaison : **Période Actuelle** vs **Période Précédente** (même durée)")
    
    # Calculs pour la régression
    # Période Actuelle (déjà filtrée dans df_filtered)
    curr_counts = df_filtered.groupby(['restaurant name', 'AM'])['order id'].count().reset_index().rename(columns={'order id': 'Orders Current'})
    
    # Période Précédente
    delta_days = (date_range[1] - date_range[0]).days + 1
    prev_start = date_range[0] - timedelta(days=delta_days)
    prev_end = date_range[0] - timedelta(days=1)
    
    mask_prev_all = (df['date'] >= prev_start) & (df['date'] <= prev_end)
    if sel_brand != 'Tous': mask_prev_all &= (df['Enseigne_Groupe'] == sel_brand)
    
    df_prev = df.loc[mask_prev_all]
    prev_counts = df_prev.groupby('restaurant name')['order id'].count().reset_index().rename(columns={'order id': 'Orders Prev'})
    
    # Merge
    reg_df = pd.merge(curr_counts, prev_counts, on='restaurant name', how='outer').fillna(0)
    
    # Calcul Delta
    reg_df['Delta'] = reg_df['Orders Current'] - reg_df['Orders Prev']
    reg_df['Variation (%)'] = ((reg_df['Orders Current'] - reg_df['Orders Prev']) / reg_df['Orders Prev'] * 100).fillna(0)
    
    # Filtrer uniquement les régressions (Delta < 0) et trier
    flop_df = reg_df[reg_df['Delta'] < 0].sort_values('Delta', ascending=True) # Les plus grosses pertes en haut
    
    # Affichage
    if not flop_df.empty:
        st.dataframe(
            flop_df[['restaurant name', 'AM', 'Orders Prev', 'Orders Current', 'Delta', 'Variation (%)']].style.format({
                'Orders Prev': '{:.0f}',
                'Orders Current': '{:.0f}',
                'Delta': '{:.0f}',
                'Variation (%)': '{:+.1f}%'
            }).background_gradient(subset=['Delta'], cmap="Reds_r"),
            use_container_width=True
        )
    else:
        st.success("Aucun restaurant en régression sur cette période ! 🎉")

with col_reg2:
    st.info(f"**Période Réf :**\n{prev_start} au {prev_end}")
    if not flop_df.empty:
        worst_am = flop_df['AM'].mode()[0] if not flop_df['AM'].mode().empty else "N/A"
        st.write(f"AM le plus impacté : **{worst_am}**")
        st.write(f"Total Perte Vol : **{flop_df['Delta'].sum()}**")

# ---------------------------------------------------------
# 3. DONNÉES DÉTAILLÉES
# ---------------------------------------------------------
with st.expander("Voir le détail des commandes"):
    st.dataframe(df_filtered)
