import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Yassir Restaurant Dashboard",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THÈME YASSIR (CSS CUSTOM) ---
st.markdown("""
    <style>
    /* Couleur principale Violet Yassir */
    :root {
        --primary-color: #6c35de;
    }
    .stButton>button {
        background-color: #6c35de;
        color: white;
    }
    .stMetric {
        background-color: #f3f0ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6c35de;
    }
    h1, h2, h3 {
        color: #4b2c92;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    # 1. Chargement des fichiers Pipeline (concaténation de tous les fichiers pipeline)
    pipeline_files = glob.glob("pipline AM.xlsx - *.csv")
    df_pipeline_list = []
    
    for filename in pipeline_files:
        try:
            temp_df = pd.read_csv(filename)
            # Nettoyage des noms de colonnes pour éviter les espaces
            temp_df.columns = temp_df.columns.str.strip()
            df_pipeline_list.append(temp_df)
        except Exception as e:
            st.warning(f"Impossible de lire {filename}: {e}")
            
    if df_pipeline_list:
        df_pipeline = pd.concat(df_pipeline_list, ignore_index=True)
    else:
        st.error("Aucun fichier 'pipline AM' trouvé.")
        return None, None

    # Sélection et renommage des colonnes utiles du Pipeline
    # On s'assure d'avoir l'ID pour la jointure
    # Colonnes attendues : ID, Restaurant Name, Created At, MAIN CITY, Commission %, Priority
    cols_to_keep = ['ID', 'Restaurant Name', 'Created At', 'MAIN CITY', 'Commission %', 'Priority']
    # Vérification si les colonnes existent
    cols_to_keep = [c for c in cols_to_keep if c in df_pipeline.columns]
    df_pipeline = df_pipeline[cols_to_keep].drop_duplicates(subset=['ID'])
    
    # 2. Chargement du fichier Data Extraction (Orders)
    # On cherche le fichier qui commence par admin-earnings
    order_files = glob.glob("admin-earnings-orders-export*.csv")
    if order_files:
        df_orders = pd.read_csv(order_files[0])
    else:
        st.error("Fichier d'extraction des commandes (admin-earnings...) introuvable.")
        return None, None

    # --- NETTOYAGE & TRANSFORMATION ---
    
    # Conversion des dates
    df_orders['order day'] = pd.to_datetime(df_orders['order day'], errors='coerce')
    
    # Calcul des KPIs au niveau Commande
    # GMV = item total
    # Statut Annulé = Si status != 'Delivered' (ou check cancelled at)
    df_orders['is_cancelled'] = df_orders['status'].apply(lambda x: 1 if x != 'Delivered' else 0)
    df_orders['GMV'] = df_orders['item total']
    
    # JOINTURE (Merge)
    # Pipeline (ID) -> Orders (Restaurant ID)
    df_merged = pd.merge(
        df_orders, 
        df_pipeline, 
        left_on='Restaurant ID', 
        right_on='ID', 
        how='left'
    )
    
    # Remplir les noms de restaurants manquants par "Inconnu" ou celui du fichier orders
    df_merged['Restaurant Name'] = df_merged['Restaurant Name'].fillna(df_merged['restaurant name'])
    df_merged['MAIN CITY'] = df_merged['MAIN CITY'].fillna(df_merged['city'])
    
    return df_merged

# --- CHARGEMENT ---
df = load_data()

if df is not None:
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filtres")
    
    # Filtre Date
    min_date = df['order day'].min()
    max_date = df['order day'].max()
    
    start_date, end_date = st.sidebar.date_input(
        "Période",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtre Ville
    cities = sorted(df['MAIN CITY'].dropna().unique())
    selected_cities = st.sidebar.multiselect("Ville", cities, default=cities)
    
    # Filtrage des données
    mask = (
        (df['order day'].dt.date >= start_date) & 
        (df['order day'].dt.date <= end_date) &
        (df['MAIN CITY'].isin(selected_cities))
    )
    df_filtered = df[mask]

    # --- MAIN DASHBOARD ---
    st.title("🟣 Dashboard de Performance Restaurants")
    st.markdown("Suivi des KPIs clés : GMV, Commandes, Annulations & Performance Partenaires")
    
    # --- KPI HEADER (GLOBAL) ---
    total_gmv = df_filtered['GMV'].sum()
    total_orders = len(df_filtered)
    total_cancelled = df_filtered['is_cancelled'].sum()
    cancellation_rate = (total_cancelled / total_orders * 100) if total_orders > 0 else 0
    active_restaurants = df_filtered['Restaurant ID'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 GMV Total", f"{total_gmv:,.0f} DH")
    col2.metric("📦 Commandes Totales", f"{total_orders:,.0f}")
    col3.metric("⚠️ Taux d'Annulation", f"{cancellation_rate:.2f} %")
    col4.metric("🏪 Restaurants Actifs", f"{active_restaurants}")
    
    st.divider()

    # --- SECTION 1: TOP 10 & FLOP 10 ---
    st.subheader("🏆 Top & Flop Performance")
    
    # Calcul des métriques par restaurant
    df_rest_kpi = df_filtered.groupby('Restaurant Name').agg({
        'GMV': 'sum',
        'order id': 'count',
        'is_cancelled': 'sum',
        'MAIN CITY': 'first'
    }).rename(columns={'order id': 'Total Orders', 'is_cancelled': 'Cancelled Orders'})
    
    df_rest_kpi['Cancellation Rate (%)'] = (df_rest_kpi['Cancelled Orders'] / df_rest_kpi['Total Orders'] * 100).round(2)
    
    # Sélecteur de métrique pour le classement
    sort_metric = st.selectbox("Classer par :", ["GMV", "Total Orders", "Cancellation Rate (%)"])
    
    col_top, col_flop = st.columns(2)
    
    with col_top:
        st.markdown("### 🚀 Top 10")
        # Tri descendant pour GMV/Orders, Ascendant pour Cancellation Rate (si on veut le meilleur taux)
        # Mais pour "Top" on veut généralement le plus haut chiffre (sauf pour taux d'annulation où c'est l'inverse)
        ascending = True if sort_metric == "Cancellation Rate (%)" else False
        
        top_10 = df_rest_kpi.sort_values(by=sort_metric, ascending=ascending).head(10)
        st.dataframe(top_10.style.background_gradient(cmap="Purples"))
        
    with col_flop:
        st.markdown("### 📉 Flop 10")
        # Inverse du Top
        flop_10 = df_rest_kpi.sort_values(by=sort_metric, ascending=not ascending).head(10)
        st.dataframe(flop_10.style.background_gradient(cmap="Reds"))

    st.divider()

    # --- SECTION 2: COURBES DE TENDANCE ---
    st.subheader("📈 Analyse de Tendance")
    
    col_trend_1, col_trend_2 = st.columns([1, 3])
    
    with col_trend_1:
        # Filtres spécifiques au graphe
        trend_metric = st.radio("Métrique à visualiser", ["GMV", "Total Orders", "Cancellation Rate"])
        
        # Liste des restos triés par GMV pour faciliter la recherche
        top_restos_list = df_rest_kpi.sort_values(by='GMV', ascending=False).index.tolist()
        selected_restos_trend = st.multiselect(
            "Comparer Restaurants (Max 5 recommandés)", 
            top_restos_list,
            default=top_restos_list[:3] # Par défaut les 3 plus gros
        )
        
    with col_trend_2:
        if selected_restos_trend:
            # Préparation données temporelles
            df_trend = df_filtered[df_filtered['Restaurant Name'].isin(selected_restos_trend)].copy()
            df_trend = df_trend.groupby(['order day', 'Restaurant Name']).agg({
                'GMV': 'sum',
                'order id': 'count',
                'is_cancelled': 'mean' # mean de 0/1 donne le %
            }).reset_index()
            
            # Renommage pour le graphique
            if trend_metric == "Cancellation Rate":
                df_trend['Value'] = df_trend['is_cancelled'] * 100
                y_label = "Taux d'Annulation (%)"
            elif trend_metric == "Total Orders":
                df_trend['Value'] = df_trend['order id']
                y_label = "Nombre de Commandes"
            else:
                df_trend['Value'] = df_trend['GMV']
                y_label = "GMV (DH)"
            
            fig = px.line(
                df_trend, 
                x='order day', 
                y='Value', 
                color='Restaurant Name',
                title=f"Évolution {y_label} par Restaurant",
                color_discrete_sequence=px.colors.sequential.Bluered_r
            )
            fig.update_layout(xaxis_title="Date", yaxis_title=y_label)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Veuillez sélectionner au moins un restaurant à gauche.")

    st.divider()

    # --- SECTION 3: VUE GLOBALE DÉTAILLÉE ---
    st.subheader("📋 Vue Globale & Détails")
    
    with st.expander("Voir le tableau complet des performances"):
        # On reprend le df_rest_kpi calculé plus haut
        st.dataframe(
            df_rest_kpi.sort_values(by="GMV", ascending=False),
            use_container_width=True,
            column_config={
                "GMV": st.column_config.NumberColumn(format="%.0f DH"),
                "Cancellation Rate (%)": st.column_config.ProgressColumn(format="%.2f %%", min_value=0, max_value=100)
            }
        )
    
    # Petit bonus : Histogramme de distribution des commissions si la donnée existe
    if 'Commission %' in df_filtered.columns and df_filtered['Commission %'].notna().any():
        st.subheader("📊 Distribution des Commissions")
        fig_comm = px.histogram(
            df_filtered.drop_duplicates(subset=['Restaurant ID']), 
            x='Commission %', 
            nbins=20, 
            color_discrete_sequence=['#6c35de'],
            title="Répartition des Taux de Commission (Partenaires Uniques)"
        )
        st.plotly_chart(fig_comm, use_container_width=True)

else:
    st.stop()
