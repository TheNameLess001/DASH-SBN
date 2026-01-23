import streamlit as st
import pandas as pd
import plotly.express as px
import os
import unicodedata
from datetime import timedelta

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="DASH-SBN | Performance & Pipeline",
    page_icon="🚀",
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
# CHARGEMENT DES PIPELINES (AM)
# ---------------------------------------------------------
@st.cache_data
def load_pipelines():
    """Charge les listes de restaurants (Pipelines) depuis les fichiers CSV AM."""
    pipelines_norm = {} # Dictionnaire {AM: [liste_noms_normalisés]}
    
    am_files = {'NAJWA': 'NAJWA.csv', 'HOUDA': 'HOUDA.csv', 'CHAIMA': 'CHAIMA.csv'}
    
    for am, filename in am_files.items():
        if os.path.exists(filename):
            try:
                # Lecture flexible (virgule ou point-virgule)
                try:
                    df_p = pd.read_csv(filename, sep=',')
                    if len(df_p.columns) < 2: raise ValueError
                except:
                    df_p = pd.read_csv(filename, sep=';')
                
                # Trouver la colonne qui contient le nom
                df_p.columns = df_p.columns.str.strip().str.lower()
                # On cherche une colonne qui contient 'restaurant' ou 'name', sinon la 1ère
                col_name = next((c for c in df_p.columns if 'restaurant' in c or 'name' in c), df_p.columns[0])
                
                # Création de la liste normalisée
                raw_list = df_p[col_name].dropna().astype(str).tolist()
                pipelines_norm[am] = [normalize_text(x) for x in raw_list]
                
            except Exception:
                pipelines_norm[am] = []
        else:
            pipelines_norm[am] = []
            
    return pipelines_norm

# ---------------------------------------------------------
# CHARGEMENT ET TRAITEMENT DES DONNÉES COMMANDES
# ---------------------------------------------------------
@st.cache_data
def load_data(main_file, pipelines_norm):
    # 1. Lecture Robuste
    if hasattr(main_file, 'seek'): main_file.seek(0)
    try:
        df = pd.read_csv(main_file, sep=',')
        # Vérif si mal lu
        if 'order day' not in df.columns and len(df.columns) < 5: raise ValueError
    except:
        if hasattr(main_file, 'seek'): main_file.seek(0)
        df = pd.read_csv(main_file, sep=';')

    df.columns = df.columns.str.strip()
    
    # 2. Parsing Dates
    df['order day'] = df['order day'].astype(str)
    df['order time'] = df['order time'].astype(str)
    
    def parse_dt(d_str):
        # Tente plusieurs formats
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
    
    # 3. Nettoyage Numérique
    for c in ['item total', 'delivery amount', 'Distance travel']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 4. Normalisation du nom du restaurant (Clé de jointure)
    df['restaurant_norm'] = df['restaurant name'].apply(normalize_text)

    # 5. Attribution AM (Logique : Pipeline > Grand Compte > Ville > Défaut)
    def get_am(row):
        r_norm = row['restaurant_norm']
        city = str(row.get('city', '')).lower()
        r_raw = str(row.get('restaurant name', '')).lower()

        # A. Check Pipeline (Match Exact ou Partiel Normalisé)
        for am, resto_list_norm in pipelines_norm.items():
            # Test 1: Le nom de la commande est dans la liste pipeline
            if r_norm in resto_list_norm:
                return am
            # Test 2: Inclusions (ex: "mcdo maarif" contient "mcdo")
            # Attention aux faux positifs courts, on filtre len > 3
            for p_norm in resto_list_norm:
                if len(p_norm) > 3 and (p_norm in r_norm):
                    return am

        # B. Fallback Logique "Hardcodée"
        if any(x in r_raw for x in ['mcdonald', 'kfc', 'burger king', 'primos', 'papa john', 'quick']): return 'NAJWA'
        if any(c in city for c in ['rabat', 'sale', 'temara', 'kenitra']): return 'HOUDA'
        return 'CHAIMA'

    df['AM'] = df.apply(get_am, axis=1)

    # 6. Automatisation
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else:
        df['is_automated'] = False
        
    # 7. Groupement Enseigne
    def get_brand(name):
        n = normalize_text(name)
        if 'mcdonald' in n: return "McDonald's"
        if 'kfc' in n: return "KFC"
        if 'burger king' in n: return "Burger King"
        if 'chrono pizza' in n: return "Chrono Pizza"
        if 'sushi' in n or 'asia' in n: return "Asian/Sushi"
        if 'tacos' in n: return "Tacos"
        return "Autres"
    
    df['Enseigne_Groupe'] = df['restaurant name'].apply(get_brand)

    return df

# ---------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Monitoring & Pipeline")

# Chargement des Pipelines au démarrage
pipelines_norm = load_pipelines()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📂 Sources")
    uploaded_file = st.file_uploader("Fichier Commandes (CSV)", type=['csv'])
    
    # Fallback pour démo locale (Optionnel)
    if not uploaded_file:
        local_path = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"
        if os.path.exists(local_path):
            # On ne charge pas auto pour laisser l'utilisateur upload, 
            # sauf si vous voulez forcer le mode démo.
            pass

    if uploaded_file:
        df = load_data(uploaded_file, pipelines_norm)
    else:
        st.info("Veuillez charger le fichier de commandes.")
        st.stop()
        
    st.divider()
    st.header("🔍 Filtres")
    
    # Sélecteur de Période
    if not df.empty:
        min_d, max_d = df['date'].min(), df['date'].max()
        date_range = st.date_input("Période Analysée", [min_d, max_d])
    else:
        st.stop()
    
    # Sélecteur Enseigne
    all_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
    sel_brand = st.selectbox("Enseigne / Groupe", all_brands)

# --- FILTRAGE PRINCIPAL ---
mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
if sel_brand != 'Tous': mask &= (df['Enseigne_Groupe'] == sel_brand)
df_filtered = df.loc[mask]

if df_filtered.empty:
    st.warning("⚠️ Aucune donnée pour cette sélection. Essayez d'élargir la période.")
    st.stop()

# ---------------------------------------------------------
# 1. TABLEAU DE BORD KPI (AM)
# ---------------------------------------------------------
st.subheader("📊 Performance par Account Manager (AM)")

# On prépare la comparaison temporelle pour le "Growth" global du tableau
# Période précédente = même durée juste avant la date de début
delta_days = (date_range[1] - date_range[0]).days + 1
prev_start = date_range[0] - timedelta(days=delta_days)
prev_end = date_range[0] - timedelta(days=1)
# Dataset global pour aller chercher l'historique hors filtre date actuel
df_prev_period = df[(df['date'] >= prev_start) & (df['date'] <= prev_end)]

summary_data = []
ams_list = ['NAJWA', 'HOUDA', 'CHAIMA']

for am in ams_list:
    # Données actuelles
    data_am = df_filtered[df_filtered['AM'] == am]
    
    # Metrics
    ca = data_am['item total'].sum()
    orders = len(data_am)
    aov = ca / orders if orders > 0 else 0
    
    # Taux
    auto_rate = (data_am['is_automated'].sum() / orders * 100) if orders > 0 else 0
    rejects = data_am[data_am['status'] == 'Restaurant Rejected'].shape[0]
    acc_rate = ((orders - rejects) / orders * 100) if orders > 0 else 100
    
    # Inactifs (Comparaison Pipeline vs Actifs Normalisés)
    pipeline = pipelines_norm.get(am, [])
    total_pipeline = len(pipeline)
    
    active_norm = data_am['restaurant_norm'].unique().tolist()
    
    # Compter les inactifs
    actives_in_pipeline = 0
    for p_norm in pipeline:
        # Est-ce que ce resto du pipeline a été actif ?
        # On vérifie si p_norm est contenu dans un des noms actifs ou l'inverse
        is_active = False
        if p_norm in active_norm:
            is_active = True
        else:
            # Recherche flexible
            for a_n in active_norm:
                if p_norm in a_n: 
                    is_active = True
                    break
        if is_active: actives_in_pipeline += 1
            
    inactifs = max(0, total_pipeline - actives_in_pipeline)
    
    # Growth (vs Période Précédente)
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

# Affichage avec style
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
st.subheader("📅 Évolution Mensuelle (Historique)")

# On prend tout l'historique disponible qui correspond au filtre Enseigne
mask_brand_hist = (df['Enseigne_Groupe'] == sel_brand) if sel_brand != 'Tous' else [True] * len(df)
df_history = df[mask_brand_hist].copy()

if not df_history.empty:
    # Agrégation par Mois
    monthly = df_history.groupby('year_month').agg({
        'item total': 'sum',
        'order id': 'count',
        'is_automated': 'mean'
    }).reset_index().sort_values('year_month')
    
    monthly['Mois'] = monthly['year_month'].dt.strftime('%Y-%m')
    
    # Calcul Variation MoM (Mois actuel vs Mois précédent)
    monthly['CA Précédent'] = monthly['item total'].shift(1)
    monthly['Growth MoM (%)'] = ((monthly['item total'] - monthly['CA Précédent']) / monthly['CA Précédent'] * 100).fillna(0)
    
    # Mise en forme pour affichage
    monthly_show = monthly[['Mois', 'item total', 'Growth MoM (%)', 'order id', 'is_automated']].copy()
    monthly_show.columns = ['Mois', 'CA (MAD)', 'Croissance MoM (%)', 'Commandes', 'Auto (%)']
    monthly_show['Auto (%)'] *= 100 # Passage en pourcentage

    st.dataframe(
        monthly_show.style.format({
            "CA (MAD)": "{:,.0f}",
            "Croissance MoM (%)": "{:+.1f}%",
            "Commandes": "{:.0f}",
            "Auto (%)": "{:.1f}%"
        }).background_gradient(subset=['Croissance MoM (%)'], cmap="RdYlGn", vmin=-20, vmax=20),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("Pas assez de données pour l'historique.")

st.divider()

# ---------------------------------------------------------
# 3. RÉGRESSION (TOP FLOP) - CORRIGÉ
# ---------------------------------------------------------
st.subheader("🚨 Restaurants en Régression (Volume Commandes)")
col_reg1, col_reg2 = st.columns([3, 1])

with col_reg1:
    # 1. Map de référence {Resto -> AM} pour éviter les trous
    # On prend le dernier AM connu pour chaque resto
    resto_am_map = df.sort_values('date').drop_duplicates('restaurant name', keep='last').set_index('restaurant name')['AM'].to_dict()

    # 2. Données Précédentes (filtrées par enseigne si besoin)
    if sel_brand != 'Tous':
        df_prev_filter = df_prev_period[df_prev_period['Enseigne_Groupe'] == sel_brand]
    else:
        df_prev_filter = df_prev_period

    # 3. GroupBy
    curr_counts = df_filtered.groupby('restaurant name')['order id'].count().reset_index().rename(columns={'order id': 'Orders Current'})
    prev_counts = df_prev_filter.groupby('restaurant name')['order id'].count().reset_index().rename(columns={'order id': 'Orders Prev'})
    
    # 4. Fusion
    reg_df = pd.merge(curr_counts, prev_counts, on='restaurant name', how='outer').fillna(0)
    
    # 5. Calcul Delta
    reg_df['Delta'] = reg_df['Orders Current'] - reg_df['Orders Prev']
    
    # 6. Récupération AM sécurisée
    reg_df['AM'] = reg_df['restaurant name'].map(resto_am_map).fillna('Autre')
    
    # 7. Filtre et Tri
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
        st.success("Aucune régression détectée sur cette période !")

with col_reg2:
    st.markdown("**Comparaison :**")
    st.caption(f"Actuel : {date_range[0]} au {date_range[1]}")
    st.caption(f"Précédent : {prev_start} au {prev_end}")
    if not flop_df.empty:
        perte = flop_df['Delta'].sum()
        st.error(f"Perte Totale : {perte:.0f} commandes")

# ---------------------------------------------------------
# 4. DATA DETAIL
# ---------------------------------------------------------
with st.expander("Voir les données brutes"):
    st.dataframe(df_filtered)
