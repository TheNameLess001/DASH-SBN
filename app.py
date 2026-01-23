import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import unicodedata
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
# FONCTIONS UTILITAIRES (NORMALISATION)
# ---------------------------------------------------------
def normalize_text(text):
    """
    Nettoie le texte pour le matching :
    - Enlève les accents (é -> e)
    - Minuscule
    - Enlève la ponctuation et les espaces superflus
    """
    if pd.isna(text): return ""
    # Normalisation unicode (sépare les accents des lettres)
    text = str(text)
    nfkd_form = unicodedata.normalize('NFKD', text)
    # Garde uniquement les caractères ASCII (enlève les accents)
    text_ascii = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Minuscule et strip
    return text_ascii.lower().strip()

# ---------------------------------------------------------
# FONCTIONS DE CHARGEMENT
# ---------------------------------------------------------
@st.cache_data
def load_pipelines():
    """Charge les listes de restaurants (Pipelines) depuis les fichiers CSV AM."""
    pipelines = {}
    # On stocke aussi la version normalisée pour le matching
    pipelines_norm = {} 
    
    am_files = {'NAJWA': 'NAJWA.csv', 'HOUDA': 'HOUDA.csv', 'CHAIMA': 'CHAIMA.csv'}
    
    for am, filename in am_files.items():
        if os.path.exists(filename):
            try:
                # Lecture flexible
                df_p = pd.read_csv(filename, sep=None, engine='python')
                
                # Trouver la colonne qui contient le nom (souvent la 1ère ou 'Name'/'Restaurant')
                df_p.columns = df_p.columns.str.strip().str.lower()
                col_name = next((c for c in df_p.columns if 'restaurant' in c or 'name' in c), df_p.columns[0])
                
                # Liste brute
                raw_list = df_p[col_name].dropna().astype(str).tolist()
                pipelines[am] = raw_list
                
                # Liste normalisée pour la comparaison
                pipelines_norm[am] = [normalize_text(x) for x in raw_list]
                
            except:
                pipelines[am] = []
                pipelines_norm[am] = []
        else:
            pipelines[am] = []
            pipelines_norm[am] = []
            
    return pipelines, pipelines_norm

@st.cache_data
def load_data(main_file, pipelines_norm):
    # 1. Lecture du fichier principal
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
        # Essayer les formats courants
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
    # Pour le tri mensuel
    df['year_month'] = df['order_datetime'].dt.to_period('M')
    df['month_str'] = df['order_datetime'].dt.strftime('%Y-%m')

    # 3. Nettoyage Numérique
    for c in ['item total', 'delivery amount', 'Distance travel']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 4. Création de la colonne Restaurant Normalisée (pour les joins)
    df['restaurant_norm'] = df['restaurant name'].apply(normalize_text)

    # 5. Attribution AM (Priorité Pipeline Normalisé > Logique Auto)
    def get_am(row):
        r_norm = row['restaurant_norm']
        
        # Check dans les pipelines chargés (match exact ou partiel sur version normalisée)
        for am, resto_list_norm in pipelines_norm.items():
            # On cherche si le nom normalisé de la commande est DANS la liste normalisée pipeline
            # Ou si une partie match (ex: "mcdonalds maarif" contient "mcdonalds")
            if r_norm in resto_list_norm:
                return am
            # Match partiel (plus lent mais utile)
            for p_norm in resto_list_norm:
                if p_norm in r_norm or r_norm in p_norm:
                    if len(p_norm) > 3: # Eviter les faux positifs courts
                        return am
        
        # Fallback Logique (si pas dans les fichiers CSV)
        city = str(row.get('city', '')).lower()
        r_raw = str(row.get('restaurant name', '')).lower()
        
        if any(x in r_raw for x in ['mcdonald', 'kfc', 'burger king', 'primos', 'papa john']): return 'NAJWA'
        if any(c in city for c in ['rabat', 'sale', 'temara', 'kenitra']): return 'HOUDA'
        return 'CHAIMA'

    df['AM'] = df.apply(get_am, axis=1)

    # 6. Autres Champs
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else:
        df['is_automated'] = False
        
    # Groupement Enseigne
    def get_brand(name):
        n = normalize_text(name)
        if 'mcdonald' in n: return "McDonald's"
        if 'kfc' in n: return "KFC"
        if 'burger king' in n: return "Burger King"
        if 'chrono pizza' in n: return "Chrono Pizza"
        if 'tacos' in n: return "Tacos"
        if 'sushi' in n or 'asia' in n: return "Asian"
        return "Autres"
    
    df['Enseigne_Groupe'] = df['restaurant name'].apply(get_brand)

    return df

# ---------------------------------------------------------
# INTERFACE & KPI
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Monitoring & Pipeline")

# CHARGEMENT
pipelines, pipelines_norm = load_pipelines()

with st.sidebar:
    st.header("Sources")
    uploaded_file = st.file_uploader("Fichier Commandes (Export Admin)", type=['csv'])
    
    # Fallback Local (pour test facile)
    default_csv = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"
    if not uploaded_file and os.path.exists(default_csv):
       # Optionnel : charger automatiquement le fichier local s'il existe
       # uploaded_file = default_csv
       pass

    if uploaded_file:
        df = load_data(uploaded_file, pipelines_norm)
    else:
        st.info("Veuillez charger le fichier de commandes.")
        st.stop()
        
    st.divider()
    st.header("Filtres")
    
    # Select Période
    if not df.empty:
        min_d, max_d = df['date'].min(), df['date'].max()
        date_range = st.date_input("Période Analysée", [min_d, max_d])
    else:
        st.stop()
    
    # Select Enseigne
    all_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
    sel_brand = st.selectbox("Enseigne", all_brands)

# FILTRAGE DONNÉES
mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if sel_brand != 'Tous': mask &= (df['Enseigne_Groupe'] == sel_brand)
df_filtered = df.loc[mask]

if df_filtered.empty:
    st.warning("Aucune donnée pour cette sélection.")
    st.stop()

# ---------------------------------------------------------
# 1. TABLEAU KPI SYNTHÉTIQUE (PAR AM)
# ---------------------------------------------------------
st.subheader("📊 Performance par Account Manager (AM)")

summary_data = []
ams_list = ['NAJWA', 'HOUDA', 'CHAIMA']

# Calcul de la période précédente (pour le Growth du tableau principal)
delta_days = (date_range[1] - date_range[0]).days + 1
prev_start = date_range[0] - timedelta(days=delta_days)
prev_end = date_range[0] - timedelta(days=1)
mask_prev_period = (df['date'] >= prev_start) & (df['date'] <= prev_end)
df_prev_period = df.loc[mask_prev_period]

for am in ams_list:
    # Données filtrées pour cet AM (Période Actuelle)
    data_am = df_filtered[df_filtered['AM'] == am]
    
    # 1. Metrics Base
    ca = data_am['item total'].sum()
    orders = len(data_am)
    aov = ca / orders if orders > 0 else 0
    
    # 2. Automatisation
    auto_cnt = data_am['is_automated'].sum()
    auto_rate = (auto_cnt / orders * 100) if orders > 0 else 0
    
    # 3. Taux d'Acceptation
    rejects = data_am[data_am['status'] == 'Restaurant Rejected'].shape[0]
    acc_rate = ((orders - rejects) / orders * 100) if orders > 0 else 100
    
    # 4. Inactifs (Matching Normalisé)
    pipeline_names_norm = pipelines_norm.get(am, [])
    total_pipeline = len(pipeline_names_norm)
    
    # Liste réelle normalisée
    active_norm = data_am['restaurant_norm'].unique().tolist()
    
    # Compter les inactifs
    # On regarde combien de noms du pipeline ne sont PAS dans les actifs
    actives_count_in_pipeline = 0
    for p_norm in pipeline_names_norm:
        # Match exact ou partiel
        found = False
        if p_norm in active_norm:
            found = True
        else:
            # Essai partiel si pas de match exact
            for a_norm in active_norm:
                if p_norm in a_norm or a_norm in p_norm:
                    found = True
                    break
        if found:
            actives_count_in_pipeline += 1
            
    inactifs = total_pipeline - actives_count_in_pipeline
    inactifs = max(0, inactifs)
    
    # 5. Growth (Vs Période Précédente)
    prev_ca = df_prev_period[df_prev_period['AM'] == am]['item total'].sum()
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

df_summary = pd.DataFrame(summary_data)
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
# 2. ANALYSE MENSUELLE (MONTH OVER MONTH)
# ---------------------------------------------------------
st.subheader("📅 Évolution Mensuelle (Comparaison des Mois)")

# On groupe par Mois pour tout le dataset (pas seulement la sélection)
# Mais on respecte le filtre Enseigne si appliqué
mask_brand = (df['Enseigne_Groupe'] == sel_brand) if sel_brand != 'Tous' else [True] * len(df)
df_monthly_base = df[mask_brand].copy()

if not df_monthly_base.empty:
    monthly_stats = df_monthly_base.groupby('year_month').agg({
        'item total': 'sum',
        'order id': 'count',
        'is_automated': 'mean',
        'AM': lambda x: x.mode()[0] if not x.mode().empty else 'Mix' # AM dominant
    }).reset_index()
    
    monthly_stats = monthly_stats.sort_values('year_month')
    monthly_stats['Mois'] = monthly_stats['year_month'].dt.strftime('%Y-%m')
    
    # Calcul des variations (Shift)
    monthly_stats['CA Précédent'] = monthly_stats['item total'].shift(1)
    monthly_stats['Growth MoM (%)'] = ((monthly_stats['item total'] - monthly_stats['CA Précédent']) / monthly_stats['CA Précédent'] * 100).fillna(0)
    
    # Mise en forme
    monthly_display = monthly_stats[['Mois', 'item total', 'Growth MoM (%)', 'order id', 'is_automated']].copy()
    monthly_display.columns = ['Mois', 'CA (MAD)', 'Croissance MoM (%)', 'Commandes', 'Auto (%)']
    monthly_display['Auto (%)'] = (monthly_display['Auto (%)'] * 100)

    st.dataframe(
        monthly_display.style.format({
            "CA (MAD)": "{:,.0f}",
            "Croissance MoM (%)": "{:+.1f}%",
            "Commandes": "{:.0f}",
            "Auto (%)": "{:.1f}%"
        }).background_gradient(subset=['Croissance MoM (%)'], cmap="RdYlGn", vmin=-20, vmax=20),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Pas assez de données historiques pour l'évolution mensuelle.")

st.divider()

# ---------------------------------------------------------
# 3. ANALYSE DE RÉGRESSION (TOP FLOP)
# ---------------------------------------------------------
st.subheader("🚨 Top Régressions (Période vs Période Précédente)")
col_reg1, col_reg2 = st.columns([3, 1])

with col_reg1:
    # Comparaison Période Actuelle (df_filtered) vs Période Précédente (df_prev_period, avec filtre enseigne)
    if sel_brand != 'Tous':
        df_prev_period_brand = df_prev_period[df_prev_period['Enseigne_Groupe'] == sel_brand]
    else:
        df_prev_period_brand = df_prev_period

    curr_counts = df_filtered.groupby(['restaurant name', 'AM'])['order id'].count().reset_index().rename(columns={'order id': 'Orders Current'})
    prev_counts = df_prev_period_brand.groupby('restaurant name')['order id'].count().reset_index().rename(columns={'order id': 'Orders Prev'})
    
    reg_df = pd.merge(curr_counts, prev_counts, on='restaurant name', how='outer').fillna(0)
    
    # Calcul Delta
    reg_df['Delta'] = reg_df['Orders Current'] - reg_df['Orders Prev']
    # On récupère l'AM correct (car parfois absent de curr ou prev)
    reg_df['AM'] = reg_df.apply(lambda x: x['AM_x'] if pd.notna(x['AM_x']) else df[df['restaurant name'] == x['restaurant name']]['AM'].iloc[0] if not df[df['restaurant name'] == x['restaurant name']].empty else "Unknown", axis=1)
    
    flop_df = reg_df[reg_df['Delta'] < 0].sort_values('Delta', ascending=True)
    
    if not flop_df.empty:
        st.dataframe(
            flop_df[['restaurant name', 'AM', 'Orders Prev', 'Orders Current', 'Delta']].style.format({
                'Orders Prev': '{:.0f}',
                'Orders Current': '{:.0f}',
                'Delta': '{:.0f}'
            }).background_gradient(subset=['Delta'], cmap="Reds_r"),
            use_container_width=True
        )
    else:
        st.success("Aucun restaurant en régression sur cette période !")

with col_reg2:
    st.markdown(f"**Comparatif :**")
    st.caption(f"Actuel: {date_range[0]} au {date_range[1]}")
    st.caption(f"Précédent: {prev_start} au {prev_end}")
    if not flop_df.empty:
        st.error(f"Perte Totale: {flop_df['Delta'].sum():.0f} cmds")

# ---------------------------------------------------------
# 4. DONNÉES DÉTAILLÉES
# ---------------------------------------------------------
with st.expander("Voir le détail des commandes brutes"):
    st.dataframe(df_filtered)
