import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# CONFIGURATION DE LA PAGE
# ---------------------------------------------------------
st.set_page_config(
    page_title="DASH-SBN | Performance Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# CSS PERSONNALISÉ
# ---------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# FONCTION DE CHARGEMENT ET DE CLEANING DES DONNÉES
# ---------------------------------------------------------
@st.cache_data
def load_data(file):
    # Fonction pour remettre le curseur au début du fichier (essentiel pour Streamlit)
    def reset_buffer(f):
        if hasattr(f, 'seek'):
            f.seek(0)

    # 1. Détection et Lecture Robuste
    df = None
    # Essai 1 : Virgule (Standard CSV)
    try:
        reset_buffer(file)
        df = pd.read_csv(file, sep=',')
        if 'order day' not in df.columns and len(df.columns) < 5:
            raise ValueError("Mauvais séparateur")
    except:
        # Essai 2 : Point-virgule (Excel CSV Europe)
        try:
            reset_buffer(file)
            df = pd.read_csv(file, sep=';')
        except:
            st.error("❌ Impossible de lire le fichier. Vérifiez qu'il s'agit bien d'un CSV.")
            st.stop()
    
    # Nettoyage des noms de colonnes (enlève espaces avant/après)
    df.columns = df.columns.str.strip()

    # Vérification de la colonne critique
    if 'order day' not in df.columns:
        st.error("⚠️ Colonne 'order day' manquante. Colonnes détectées :")
        st.write(df.columns.tolist())
        st.stop()

    # 2. Conversion des Dates (Gestion multi-formats)
    # On nettoie d'abord les colonnes date pour éviter les erreurs de type
    df['order day'] = df['order day'].astype(str)
    df['order time'] = df['order time'].astype(str)

    # Fonction de parsing date flexible
    def parse_dates(date_str):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str, errors='coerce')

    # Application de la conversion
    df['order_date_obj'] = df['order day'].apply(parse_dates)
    
    # Création du champ datetime complet
    df['order_datetime'] = pd.to_datetime(
        df['order_date_obj'].dt.strftime('%Y-%m-%d') + ' ' + df['order time'], 
        errors='coerce'
    )
    
    df['date'] = df['order_datetime'].dt.date
    df['month_str'] = df['order_datetime'].dt.strftime('%Y-%m')

    # 3. Nettoyage des colonnes numériques
    numeric_cols = ['item total', 'delivery amount', 'Distance travel']
    for col in numeric_cols:
        if col in df.columns:
            # Force la conversion en nombre, remplace les erreurs par 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 4. Logique d'attribution des Account Managers (AM)
    def assign_am(row):
        rest_name = str(row.get('restaurant name', '')).lower()
        city = str(row.get('city', '')).lower()
        
        # Liste des Grands Comptes (NAJWA)
        key_accounts = [
            "mcdonald", "kfc", "burger king", "primos", 
            "papa john", "quick", "chrono pizza"
        ]
        
        # Villes HOUDA
        houda_cities = ["rabat", "sale", "salé", "temara", "témara", "kenitra", "kénitra"]
        
        # Règle 1 : Grands Comptes -> NAJWA
        if any(acc in rest_name for acc in key_accounts):
            return "NAJWA"
        
        # Règle 2 : Zone Géographique -> HOUDA
        if any(c in city for c in houda_cities):
            return "HOUDA"
        
        # Règle 3 : Par défaut (Casa & alentours) -> CHAIMA
        return "CHAIMA"

    df['AM'] = df.apply(assign_am, axis=1)

    # 5. Logique d'Automatisation
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else:
        df['is_automated'] = False
    
    return df

# ---------------------------------------------------------
# INTERFACE UTILISATEUR
# ---------------------------------------------------------

# Titre
st.title("🚀 DASH-SBN | Performance Analytics")
st.markdown("Analyse des ventes, opérations et performance des Account Managers.")

# Sidebar pour l'upload et les filtres
with st.sidebar:
    st.header("Paramètres")
    uploaded_file = st.file_uploader("Charger le fichier CSV (Export Admin)", type=['csv'])
    
    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        st.info("Veuillez uploader un fichier CSV pour commencer.")
        st.stop()

    # Filtres
    st.subheader("Filtres")
    
    # Sélecteur d'AM
    am_options = ['Global'] + sorted(list(df['AM'].unique()))
    selected_am = st.selectbox("Choisir l'Account Manager (AM)", am_options)
    
    # Sélecteur de Date
    if not df['date'].isna().all():
        min_date = df['date'].min()
        max_date = df['date'].max()
        date_range = st.date_input("Période", [min_date, max_date])
    else:
        st.warning("Problème avec les dates dans le fichier.")
        st.stop()

# ---------------------------------------------------------
# LOGIQUE DE FILTRAGE
# ---------------------------------------------------------

# 1. Filtre Date
if isinstance(date_range, list) and len(date_range) == 2:
    mask_date = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
    df_filtered = df.loc[mask_date]
elif isinstance(date_range, list) and len(date_range) == 1:
    # Cas où l'utilisateur n'a sélectionné qu'une date de début
    mask_date = (df['date'] >= date_range[0])
    df_filtered = df.loc[mask_date]
else:
    df_filtered = df.copy()

# 2. Filtre AM
if selected_am != 'Global':
    df_filtered = df_filtered[df_filtered['AM'] == selected_am]

# ---------------------------------------------------------
# CALCUL DES KPIs
# ---------------------------------------------------------

# Metrics principales
total_orders = len(df_filtered)
total_revenue = df_filtered['item total'].sum()
aov = total_revenue / total_orders if total_orders > 0 else 0

# Taux d'annulation
# Statuts indiquant une annulation
cancel_statuses = ['Cancelled', 'Canceled', 'Cancelled by user', 'Restaurant Rejected', 'Auto Cancelled']
cancelled_orders = df_filtered[df_filtered['status'].isin(cancel_statuses)].shape[0]
cancel_rate = (cancelled_orders / total_orders * 100) if total_orders > 0 else 0

# Taux d'automatisation
auto_orders = df_filtered[df_filtered['is_automated'] == True].shape[0]
auto_rate = (auto_orders / total_orders * 100) if total_orders > 0 else 0

# ---------------------------------------------------------
# AFFICHAGE DES KPIs (Ligne du haut)
# ---------------------------------------------------------

st.markdown(f"### Vue : **{selected_am}**")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Chiffre d'Affaires", f"{total_revenue:,.0f} MAD")
with col2:
    st.metric("Commandes", f"{total_orders}")
with col3:
    st.metric("Panier Moyen (AOV)", f"{aov:.1f} MAD")
with col4:
    st.metric("Taux Annulation", f"{cancel_rate:.1f}%", delta_color="inverse")
with col5:
    st.metric("Automatisation", f"{auto_rate:.1f}%")

st.divider()

# ---------------------------------------------------------
# GRAPHIQUES ET ANALYSE DÉTAILLÉE
# ---------------------------------------------------------

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📈 Évolution du Chiffre d'Affaires")
    # Group by Date
    daily_sales = df_filtered.groupby('date')['item total'].sum().reset_index()
    
    if not daily_sales.empty:
        fig_evol = px.line(daily_sales, x='date', y='item total', markers=True, 
                           title="CA Quotidien", labels={'item total': 'CA (MAD)', 'date': 'Date'})
        fig_evol.update_layout(xaxis_title=None)
        st.plotly_chart(fig_evol, use_container_width=True)
    else:
        st.info("Pas assez de données pour afficher l'évolution.")

with c2:
    st.subheader("🏆 Top Restaurants")
    top_restos = df_filtered.groupby('restaurant name')['item total'].sum().sort_values(ascending=False).head(10).reset_index()
    
    if not top_restos.empty:
        fig_bar = px.bar(top_restos, x='item total', y='restaurant name', orientation='h',
                         title="Par Chiffre d'Affaires", text_auto='.2s')
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="CA (MAD)", yaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Aucune donnée.")

# ---------------------------------------------------------
# SECTION CROISSANCE (GROWTH) & DÉTAILS
# ---------------------------------------------------------

st.subheader("📊 Détail de la Performance Mensuelle")

# Tableau croisé dynamique par Mois
if not df_filtered.empty:
    monthly_kpis = df_filtered.groupby('month_str').agg({
        'item total': 'sum',
        'order id': 'count',
        'is_automated': 'mean'
    }).reset_index()

    monthly_kpis.columns = ['Mois', 'Chiffre d\'Affaires', 'Commandes', '% Auto']
    monthly_kpis['% Auto'] = (monthly_kpis['% Auto'] * 100).round(1)
    monthly_kpis['Panier Moyen'] = (monthly_kpis['Chiffre d\'Affaires'] / monthly_kpis['Commandes']).round(1)

    # Calcul Growth (Croissance CA)
    monthly_kpis['Croissance (%)'] = monthly_kpis['Chiffre d\'Affaires'].pct_change().mul(100).round(1).fillna(0)

    # Réordonner les colonnes
    monthly_kpis = monthly_kpis[['Mois', 'Chiffre d\'Affaires', 'Croissance (%)', 'Commandes', 'Panier Moyen', '% Auto']]

    st.dataframe(monthly_kpis, use_container_width=True, hide_index=True)
else:
    st.info("Aucune donnée disponible pour la période sélectionnée.")

# ---------------------------------------------------------
# APERÇU DES DONNÉES BRUTES
# ---------------------------------------------------------
with st.expander("Voir les données brutes"):
    st.dataframe(df_filtered)
