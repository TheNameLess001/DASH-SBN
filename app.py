import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="DASH-SBN | Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# FONCTIONS UTILITAIRES (DATE & PARSING)
# ---------------------------------------------------------
def parse_dates_robust(date_series, time_series):
    """Combine date et heure en gérant plusieurs formats (DD/MM/YYYY ou YYYY-MM-DD)"""
    # Etape 1 : Convertir en string propre
    d_str = date_series.astype(str).str.strip()
    t_str = time_series.astype(str).str.strip()
    
    # Etape 2 : Essayer de convertir la partie Date
    # On teste d'abord le format ISO (2026-01-23) car c'est le plus probable dans tes fichiers récents
    dates = pd.to_datetime(d_str, format='%Y-%m-%d', errors='coerce')
    
    # Si on a des NaT (Not a Time), on essaye le format français (23/01/2026)
    mask_nat = dates.isna()
    if mask_nat.any():
        dates[mask_nat] = pd.to_datetime(d_str[mask_nat], format='%d/%m/%Y', errors='coerce')
    
    # Etape 3 : Combiner avec l'heure
    full_datetime = pd.to_datetime(
        dates.dt.strftime('%Y-%m-%d') + ' ' + t_str, 
        errors='coerce'
    )
    return full_datetime

def detect_brand(restaurant_name):
    """Extrait le nom de l'enseigne pour le groupement"""
    name = str(restaurant_name).lower()
    
    if 'mcdonald' in name: return "McDonald's"
    if 'kfc' in name: return "KFC"
    if 'burger king' in name: return "Burger King"
    if 'pizza hut' in name: return "Pizza Hut"
    if 'domino' in name: return "Domino's"
    if 'primos' in name: return "Primos"
    if 'papa john' in name: return "Papa John's"
    if 'quick' in name: return "Quick"
    if 'chrono pizza' in name: return "Chrono Pizza"
    if 'sushi' in name: return "Sushi / Asiatique"
    if 'tacos' in name: return "Tacos Indép."
    
    # Pour les autres, on retourne "Autres Indépendants" 
    # ou le nom complet si tu préfères trop de détails, mais "Autres" est mieux pour un dashboard global
    return "Autres / Indépendants"

# ---------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data(files_list):
    all_dfs = []
    
    for file in files_list:
        # Rembobiner le fichier si nécessaire (pour les uploads)
        if hasattr(file, 'seek'):
            file.seek(0)
            
        # Tentative de lecture (Virgule puis Point-Virgule)
        try:
            df = pd.read_csv(file, sep=',')
            # Vérification basique
            if 'order day' not in df.columns and len(df.columns) < 5:
                raise ValueError("Sep , failed")
        except:
            if hasattr(file, 'seek'): file.seek(0)
            df = pd.read_csv(file, sep=';')
            
        # Nettoyage des colonnes
        df.columns = df.columns.str.strip()
        
        # Si le fichier est vide ou mal lu, on saute
        if 'order day' not in df.columns:
            continue
            
        all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame() # Retourne vide si échec
        
    # Fusion des fichiers (NAJWA + HOUDA + CHAIMA)
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # --- TRAITEMENT DES DONNÉES ---
    
    # 1. Dates
    final_df['order_datetime'] = parse_dates_robust(final_df['order day'], final_df['order time'])
    final_df['date'] = final_df['order_datetime'].dt.date
    final_df['month_str'] = final_df['order_datetime'].dt.strftime('%Y-%m')
    
    # 2. Numériques
    cols_num = ['item total', 'delivery amount', 'Distance travel']
    for c in cols_num:
        if c in final_df.columns:
            final_df[c] = pd.to_numeric(final_df[c], errors='coerce').fillna(0)
            
    # 3. Attribution AM (Logique existante conservée pour sécurité)
    def assign_am_logic(row):
        rest = str(row.get('restaurant name', '')).lower()
        city = str(row.get('city', '')).lower()
        
        # AM NAJWA (Grands Comptes)
        key_accounts = ["mcdonald", "kfc", "burger king", "primos", "papa john", "quick", "chrono pizza"]
        if any(k in rest for k in key_accounts): return "NAJWA"
        
        # AM HOUDA (Rabat region)
        if any(c in city for c in ["rabat", "sale", "salé", "temara", "témara", "kenitra"]): return "HOUDA"
        
        # AM CHAIMA (Default)
        return "CHAIMA"
        
    final_df['AM'] = final_df.apply(assign_am_logic, axis=1)
    
    # 4. Groupement d'Enseigne (NOUVEAU)
    final_df['Enseigne_Groupe'] = final_df['restaurant name'].apply(detect_brand)
    
    # 5. Automatisation
    if 'Assigned By' in final_df.columns:
        final_df['is_automated'] = final_df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else:
        final_df['is_automated'] = False
        
    return final_df

# ---------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Performance Analytics")

# --- SIDEBAR : CHARGEMENT ---
with st.sidebar:
    st.header("🗂️ Sources de Données")
    
    # 1. Vérification des fichiers locaux (GitHub structure)
    local_files = ['NAJWA.csv', 'HOUDA.csv', 'CHAIMA.csv']
    found_files = [f for f in local_files if os.path.exists(f)]
    
    df = pd.DataFrame()
    
    if found_files:
        st.success(f"✅ {len(found_files)} fichiers détectés (Auto-load)")
        # On charge ces fichiers locaux
        # On doit les ouvrir en mode lecture pour pandas
        files_to_load = [open(f, 'r') for f in found_files]
        df = load_and_process_data(files_to_load)
        # On ferme les fichiers après lecture
        for f in files_to_load: f.close()
    else:
        # 2. Mode Upload Manuel (Fallback)
        uploaded_files = st.file_uploader(
            "Uploader les fichiers CSV (Un ou Plusieurs)", 
            type=['csv'], 
            accept_multiple_files=True
        )
        if uploaded_files:
            df = load_and_process_data(uploaded_files)

    if df.empty:
        st.info("En attente de données...")
        st.stop()

    # --- SIDEBAR : FILTRES ---
    st.divider()
    st.header("🔍 Filtres")
    
    # Filtre AM
    all_ams = ['Tous'] + sorted(list(df['AM'].unique()))
    selected_am = st.selectbox("Account Manager", all_ams)
    
    # Filtre GROUPE (NOUVEAU)
    all_groups = sorted(list(df['Enseigne_Groupe'].unique()))
    selected_groups = st.multiselect("Enseigne / Groupe", all_groups, default=all_groups)
    
    # Filtre DATE
    if not df['date'].isna().all():
        min_d, max_d = df['date'].min(), df['date'].max()
        date_range = st.date_input("Période", [min_d, max_d])
    else:
        date_range = []

# ---------------------------------------------------------
# APPLICATION DES FILTRES
# ---------------------------------------------------------
df_filtered = df.copy()

# 1. Date
if isinstance(date_range, list) and len(date_range) == 2:
    df_filtered = df_filtered[
        (df_filtered['date'] >= date_range[0]) & 
        (df_filtered['date'] <= date_range[1])
    ]

# 2. AM
if selected_am != 'Tous':
    df_filtered = df_filtered[df_filtered['AM'] == selected_am]

# 3. Groupe (Nouveau)
if selected_groups:
    df_filtered = df_filtered[df_filtered['Enseigne_Groupe'].isin(selected_groups)]

# ---------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------

# KPIs
total_rev = df_filtered['item total'].sum()
total_orders = len(df_filtered)
aov = total_rev / total_orders if total_orders > 0 else 0
auto_rate = (df_filtered['is_automated'].sum() / total_orders * 100) if total_orders > 0 else 0

cancel_mask = df_filtered['status'].astype(str).str.contains('Cancel|Reject', case=False)
cancel_rate = (cancel_mask.sum() / total_orders * 100) if total_orders > 0 else 0

st.markdown(f"### Vue : **{selected_am}**")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Chiffre d'Affaires", f"{total_rev:,.0f} MAD")
k2.metric("Commandes", f"{total_orders}")
k3.metric("Panier Moyen", f"{aov:.0f} MAD")
k4.metric("Taux Annulation", f"{cancel_rate:.1f}%")
k5.metric("Automatisation", f"{auto_rate:.1f}%")

st.divider()

# GRAPHIQUES
g1, g2 = st.columns([2, 1])

with g1:
    st.subheader("📈 Évolution du CA")
    daily = df_filtered.groupby('date')['item total'].sum().reset_index()
    if not daily.empty:
        fig = px.line(daily, x='date', y='item total', markers=True, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Pas de données temporelles.")

with g2:
    st.subheader("🍔 Répartition par Groupe")
    # On utilise le nouveau groupement pour ce camembert
    pie_data = df_filtered.groupby('Enseigne_Groupe')['item total'].sum().reset_index()
    if not pie_data.empty:
        fig_pie = px.pie(pie_data, values='item total', names='Enseigne_Groupe', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("Pas de données.")

# TABLEAU DÉTAILLÉ PAR ENSEIGNE
st.subheader("Détail par Enseigne (Top 20)")
top_brands = df_filtered.groupby('restaurant name').agg({
    'item total': 'sum',
    'order id': 'count',
    'Enseigne_Groupe': 'first'
}).sort_values('item total', ascending=False).head(20).reset_index()

top_brands.columns = ['Restaurant', 'CA (MAD)', 'Commandes', 'Groupe']
st.dataframe(top_brands, use_container_width=True)
