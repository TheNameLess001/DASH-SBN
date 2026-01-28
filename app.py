import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import os
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Yassir Performance Dashboard",
    page_icon="🟣",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THÈME YASSIR & CSS ---
st.markdown("""
    <style>
    /* Couleur principale Violet Yassir #6c35de */
    :root { --primary-color: #6c35de; }
    
    /* Titres */
    h1, h2, h3 { color: #4b2c92; font-family: 'Sans-serif'; }
    
    /* Metrics Cards */
    .stMetric {
        background-color: #f8f6ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6c35de;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Boutons */
    .stButton>button {
        background-color: #6c35de;
        color: white;
        border-radius: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #6c35de !important;
        color: white !important;
    }
    
    /* Signature Footer */
    .footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        color: #888;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE CHARGEMENT ---

@st.cache_data
def load_pipeline_data():
    """Charge les fichiers Pipeline locaux et extrait le nom de l'AM du nom de fichier."""
    pipeline_files = glob.glob("pipline AM*.csv") # Pattern flexible
    df_list = []
    
    for filename in pipeline_files:
        try:
            # Extraction du nom de l'AM (ex: "pipline AM.xlsx - IMANE.csv" -> "IMANE")
            # On suppose que le format est constant, on prend la partie après le dernier tiret ou avant le .csv
            base_name = os.path.basename(filename)
            am_name = "Inconnu"
            if " - " in base_name:
                am_name = base_name.split(" - ")[-1].replace(".csv", "").strip()
            
            temp_df = pd.read_csv(filename)
            temp_df.columns = temp_df.columns.str.strip() # Nettoyage colonnes
            temp_df['AM_Owner'] = am_name # Ajout colonne AM
            df_list.append(temp_df)
        except Exception as e:
            st.warning(f"Erreur lecture {filename}: {e}")
            
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    return pd.DataFrame()

def load_orders_data(uploaded_file):
    """Charge le fichier Admin Earnings uploadé."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {e}")
            return None
    return None

def calculate_kpis(df_merged):
    """Calcule les métriques enrichies."""
    # Dates
    df_merged['order day'] = pd.to_datetime(df_merged['order day'], errors='coerce')
    df_merged['Month'] = df_merged['order day'].dt.to_period('M')
    
    # KPIs de base
    df_merged['is_cancelled'] = df_merged['status'].apply(lambda x: 1 if str(x).lower() != 'delivered' else 0)
    df_merged['GMV'] = pd.to_numeric(df_merged['item total'], errors='coerce').fillna(0)
    df_merged['Delivery Time'] = pd.to_numeric(df_merged['delivery time(M)'], errors='coerce')
    
    return df_merged

# --- MAIN APP ---

st.title("🟣 Yassir Analytics Dashboard")
st.markdown("*Pilotage de la performance Restaurant - Powered by Data*")

# 1. SIDEBAR : UPLOAD & FILTRES GLOBAUX
with st.sidebar:
    st.header("📂 Données")
    st.info("Les fichiers 'Pipeline AM' sont chargés automatiquement depuis le dossier.")
    
    uploaded_file = st.file_uploader("Uploader 'Admin Earnings' (CSV)", type=['csv'])
    
    if uploaded_file is None:
        st.warning("Veuillez uploader le fichier Admin Earnings pour commencer.")
        st.stop()

# Chargement des données
df_pipeline = load_pipeline_data()
df_orders = load_orders_data(uploaded_file)

if df_orders is not None and not df_pipeline.empty:
    
    # Fusion des données
    # On garde toutes les commandes, et on joint les infos Pipeline (AM, Commission, etc.)
    # Clé: Orders[Restaurant ID] <-> Pipeline[ID]
    # Standardisation ID
    if 'ID' in df_pipeline.columns and 'Restaurant ID' in df_orders.columns:
        df_full = pd.merge(
            df_orders,
            df_pipeline[['ID', 'Restaurant Name', 'AM_Owner', 'Commission %', 'Priority', 'Created At', 'MAIN CITY']],
            left_on='Restaurant ID',
            right_on='ID',
            how='left'
        )
        
        # Nettoyage après fusion
        df_full['AM_Owner'] = df_full['AM_Owner'].fillna('Non Assigné')
        df_full['Restaurant Name'] = df_full['Restaurant Name'].fillna(df_full['restaurant name'])
        df_full['City'] = df_full['MAIN CITY'].fillna(df_full['city'])
        
        # Calculs
        df_full = calculate_kpis(df_full)
        
        # FILTRES SIDEBAR SUITE
        with st.sidebar:
            st.divider()
            st.header("🔍 Filtres Temporels")
            min_date = df_full['order day'].min().date()
            max_date = df_full['order day'].max().date()
            date_range = st.date_input("Période", [min_date, max_date], min_value=min_date, max_value=max_date)
            
            if len(date_range) == 2:
                start, end = date_range
                mask = (df_full['order day'].dt.date >= start) & (df_full['order day'].dt.date <= end)
                df_filtered = df_full[mask]
            else:
                df_filtered = df_full
        
        # --- ONGLETS PRINCIPAUX ---
        tab_am, tab_global = st.tabs(["👤 Vue par AM", "🌍 Vue Global Yassir"])
        
        # ==========================================================================================
        # SECTION 1 : VUE PAR AM
        # ==========================================================================================
        with tab_am:
            st.header("Analyse de la Performance par Account Manager")
            
            # Sélecteur AM
            all_ams = sorted(df_filtered['AM_Owner'].unique())
            selected_am = st.selectbox("Selectionner un Account Manager (AM):", ["Tous"] + list(all_ams))
            
            # Filtrage AM
            if selected_am != "Tous":
                df_am = df_filtered[df_filtered['AM_Owner'] == selected_am]
            else:
                df_am = df_filtered
            
            if df_am.empty:
                st.warning("Aucune donnée pour cette sélection.")
            else:
                # 1. KPIs Global AM
                kpi_gmv = df_am['GMV'].sum()
                kpi_orders = len(df_am)
                kpi_aov = kpi_gmv / kpi_orders if kpi_orders > 0 else 0
                kpi_cancel = (df_am['is_cancelled'].sum() / kpi_orders * 100) if kpi_orders > 0 else 0
                kpi_del_time = df_am['Delivery Time'].mean()
                
                # Layout Métriques
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("💰 GMV Total", f"{kpi_gmv:,.0f} DH")
                col2.metric("📦 Commandes", f"{kpi_orders}")
                col3.metric("🛒 AOV (Panier Moyen)", f"{kpi_aov:.0f} DH")
                col4.metric("❌ Taux Annulation", f"{kpi_cancel:.1f}%")
                col5.metric("⏱️ Temps Livraison", f"{kpi_del_time:.0f} min")
                
                st.divider()
                
                # 2. Graphiques d'Évolution (Growth)
                st.subheader(f"📈 Évolution Mensuelle - {selected_am}")
                
                df_monthly = df_am.groupby(df_am['order day'].dt.to_period('M')).agg({
                    'GMV': 'sum',
                    'order id': 'count',
                    'is_cancelled': 'mean'
                }).reset_index()
                df_monthly['order day'] = df_monthly['order day'].astype(str)
                df_monthly['Cancellation Rate'] = df_monthly['is_cancelled'] * 100
                
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    fig_gmv = px.bar(df_monthly, x='order day', y='GMV', title="Croissance GMV (Mensuel)", color_discrete_sequence=['#6c35de'])
                    st.plotly_chart(fig_gmv, use_container_width=True)
                with col_chart2:
                    fig_orders = px.line(df_monthly, x='order day', y='order id', title="Tendance Volume Commandes", markers=True, color_discrete_sequence=['#9b72e6'])
                    st.plotly_chart(fig_orders, use_container_width=True)

                st.divider()

                # 3. TOP 10 / FLOP 10 (Filtrable)
                st.subheader("🏆 Classement Restaurants (Top & Flop)")
                
                # Préparation des données agrégées par restaurant
                df_restos = df_am.groupby(['Restaurant Name', 'City', 'Priority']).agg({
                    'GMV': 'sum',
                    'order id': 'count',
                    'is_cancelled': 'mean',
                    'Delivery Time': 'mean',
                    'Commission %': 'max'
                }).reset_index()
                df_restos['Cancellation %'] = df_restos['is_cancelled'] * 100
                df_restos['AOV'] = df_restos['GMV'] / df_restos['order id']
                
                # Filtre métrique
                metric_options = {
                    "Chiffre d'Affaires (GMV)": 'GMV',
                    "Volume Commandes": 'order id',
                    "Taux d'Annulation": 'Cancellation %',
                    "Panier Moyen (AOV)": 'AOV',
                    "Temps de Livraison": 'Delivery Time'
                }
                selected_metric_label = st.selectbox("Classer les restaurants par :", list(metric_options.keys()))
                selected_metric_col = metric_options[selected_metric_label]
                
                # Logique de tri (Ascendant pour Annulation/Temps, Descendant pour GMV/Commandes)
                ascending_sort = True if selected_metric_col in ['Cancellation %', 'Delivery Time'] else False
                
                col_top, col_flop = st.columns(2)
                
                with col_top:
                    st.markdown("#### 🌟 Top 10 Performers")
                    st.dataframe(
                        df_restos.sort_values(by=selected_metric_col, ascending=ascending_sort).head(10)
                        [['Restaurant Name', 'GMV', 'order id', 'Cancellation %', 'AOV', 'Delivery Time']]
                        .style.format({'GMV': "{:.0f}", 'Cancellation %': "{:.1f}", 'AOV': "{:.0f}", 'Delivery Time': "{:.1f}"})
                        .background_gradient(cmap="Purples", subset=[selected_metric_col])
                    )

                with col_flop:
                    st.markdown("#### ⚠️ Flop 10 (À surveiller)")
                    st.dataframe(
                        df_restos.sort_values(by=selected_metric_col, ascending=not ascending_sort).head(10)
                        [['Restaurant Name', 'GMV', 'order id', 'Cancellation %', 'AOV', 'Delivery Time']]
                        .style.format({'GMV': "{:.0f}", 'Cancellation %': "{:.1f}", 'AOV': "{:.0f}", 'Delivery Time': "{:.1f}"})
                        .background_gradient(cmap="Reds", subset=[selected_metric_col])
                    )
                
                st.divider()
                
                # 4. Vue Détaillée & Filtres Resto
                st.subheader("📋 Vue Détaillée par Restaurant")
                
                restos_list = sorted(df_am['Restaurant Name'].unique())
                selected_restos = st.multiselect("Filtrer par Restaurant(s) :", restos_list)
                
                if selected_restos:
                    df_detail = df_restos[df_restos['Restaurant Name'].isin(selected_restos)]
                else:
                    df_detail = df_restos # Tout afficher par défaut
                
                st.dataframe(
                    df_detail.style.format({'GMV': "{:.0f} DH", 'Commission %': "{:.0f}%", 'Cancellation %': "{:.2f}%", 'AOV': "{:.0f} DH", 'Delivery Time': "{:.1f} min"}),
                    use_container_width=True
                )

        # ==========================================================================================
        # SECTION 2 : VUE GLOBAL YASSIR
        # ==========================================================================================
        with tab_global:
            st.header("🌍 Vue Global Yassir (Tous les AMs)")
            
            # KPIs Macro
            total_gmv = df_filtered['GMV'].sum()
            total_orders = len(df_filtered)
            global_cancel_rate = (df_filtered['is_cancelled'].sum() / total_orders * 100) if total_orders else 0
            
            # Affichage style "Big Numbers"
            col1, col2, col3 = st.columns(3)
            col1.metric("Total GMV Yassir", f"{total_gmv:,.0f} DH", delta="Global")
            col2.metric("Commandes Totales", f"{total_orders:,.0f}")
            col3.metric("Taux Annulation Moyen", f"{global_cancel_rate:.2f}%", delta_color="inverse")
            
            st.divider()
            
            # Analyse croisée par AM
            st.subheader("Performance comparée des Account Managers")
            
            df_by_am = df_filtered.groupby('AM_Owner').agg({
                'GMV': 'sum',
                'order id': 'count',
                'is_cancelled': 'mean',
                'Restaurant Name': 'nunique'
            }).reset_index().rename(columns={'Restaurant Name': 'Nb Portefeuille'})
            
            df_by_am['Cancellation %'] = df_by_am['is_cancelled'] * 100
            
            # Graphique comparatif
            fig_am_comp = px.scatter(
                df_by_am, 
                x='Nb Portefeuille', 
                y='GMV', 
                size='order id', 
                color='AM_Owner',
                hover_name='AM_Owner',
                title="Performance AM : Portefeuille vs GMV (Taille = Vol. Commandes)",
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            st.plotly_chart(fig_am_comp, use_container_width=True)
            
            st.dataframe(df_by_am.style.background_gradient(cmap="Purples", subset=['GMV']), use_container_width=True)
            
            st.markdown("---")
            st.markdown("**Note :** Les données affichées dépendent de la période sélectionnée et des fichiers chargés.")

    else:
        st.error("Erreur de fusion des données. Vérifiez que les fichiers Pipeline contiennent bien une colonne 'ID' et le fichier Admin une colonne 'Restaurant ID'.")

# --- FOOTER ---
st.markdown('<div class="footer">Bounoir Saif eddine - Yassir Analytics</div>', unsafe_allow_html=True)
