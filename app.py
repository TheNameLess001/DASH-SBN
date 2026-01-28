import streamlit as st
import pandas as pd
import plotly.express as px
import glob
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Yassir Analytics", page_icon="🟣", layout="wide")

# --- CSS YASSIR ---
st.markdown("""
    <style>
    :root { --primary-color: #6c35de; }
    .stMetric { background-color: #f8f6ff; border-left: 5px solid #6c35de; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #4b2c92; }
    .stButton>button { background-color: #6c35de; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS ROBUSTES ---

@st.cache_data
def load_pipeline_data():
    # On cherche tous les CSV qui commencent par "pipline"
    pipeline_files = glob.glob("pipline*.csv") 
    df_list = []
    
    if not pipeline_files:
        st.error("⚠️ Aucun fichier Pipeline trouvé ! Vérifiez qu'ils sont bien dans le dossier.")
        return pd.DataFrame()

    for filename in pipeline_files:
        try:
            # Nom AM
            base_name = os.path.basename(filename)
            am_name = "Inconnu"
            if " - " in base_name:
                am_name = base_name.split(" - ")[-1].replace(".csv", "").strip()
            
            temp_df = pd.read_csv(filename)
            
            # --- CORRECTION CRITIQUE : NORMALISATION DES COLONNES ---
            # On met tout en MAJUSCULE et on enlève les espaces pour éviter les erreurs "Id" vs "ID"
            temp_df.columns = temp_df.columns.str.strip().str.upper()
            
            # On renomme pour être sûr d'avoir les bonnes colonnes standard
            # Si on trouve 'ID' ou 'ID ', on garde. Idem pour 'RESTAURANT NAME'
            
            temp_df['AM_OWNER'] = am_name
            df_list.append(temp_df)
            
        except Exception as e:
            st.warning(f"⚠️ Erreur lecture fichier {filename}: {e}")
            
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        return final_df
    return pd.DataFrame()

def load_orders_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Erreur lecture Admin Earnings: {e}")
    return None

# --- MAIN APP ---
st.title("🟣 Yassir Analytics Dashboard (Corrigé)")

# Sidebar
with st.sidebar:
    st.header("📂 Import Données")
    uploaded_file = st.file_uploader("1. Uploader 'Admin Earnings' (CSV)", type=['csv'])
    
    st.info(f"📂 Fichiers Pipeline détectés : {len(glob.glob('pipline*.csv'))}")

# Chargement
df_pipeline = load_pipeline_data()
df_orders = load_orders_data(uploaded_file)

if df_orders is not None and not df_pipeline.empty:
    
    # --- PRÉPARATION ORDERS ---
    # Conversion date
    df_orders['order day'] = pd.to_datetime(df_orders['order day'], errors='coerce')
    
    # --- FUSION INTELLIGENTE ---
    # On s'assure que la clé de jointure est propre
    if 'ID' in df_pipeline.columns:
        join_key_pipeline = 'ID'
    else:
        st.error("❌ Colonne 'ID' introuvable dans le pipeline. Colonnes dispo : " + str(df_pipeline.columns.tolist()))
        st.stop()
        
    # Fusion
    df_full = pd.merge(
        df_orders,
        df_pipeline,
        left_on='Restaurant ID', # Clé dans fichier Admin
        right_on=join_key_pipeline, # Clé dans Pipeline (ID)
        how='left'
    )
    
    # Remplissage des trous (si le resto n'est pas dans le pipeline)
    df_full['AM_OWNER'] = df_full['AM_OWNER'].fillna('Non Assigné')
    # On utilise 'RESTAURANT NAME' du pipeline s'il existe, sinon celui des orders
    if 'RESTAURANT NAME' in df_full.columns:
        df_full['Restaurant Name Final'] = df_full['RESTAURANT NAME'].fillna(df_full['restaurant name'])
    else:
        df_full['Restaurant Name Final'] = df_full['restaurant name']

    # --- CALCULS KPI ---
    df_full['is_cancelled'] = df_full['status'].apply(lambda x: 1 if str(x).lower() != 'delivered' else 0)
    df_full['GMV'] = pd.to_numeric(df_full['item total'], errors='coerce').fillna(0)
    df_full['Delivery Time'] = pd.to_numeric(df_full['delivery time(M)'], errors='coerce')

    # --- FILTRE DATE (IMPORTANT CAR DONNÉES 2026) ---
    with st.sidebar:
        st.divider()
        st.header("📅 Période")
        min_date = df_full['order day'].min().date()
        max_date = df_full['order day'].max().date()
        date_range = st.date_input("Choisir Période", [min_date, max_date])
        
        if len(date_range) == 2:
            mask = (df_full['order day'].dt.date >= date_range[0]) & (df_full['order day'].dt.date <= date_range[1])
            df_filtered = df_full[mask]
        else:
            df_filtered = df_full

    # --- DASHBOARD ---
    
    # DEBUG EXPANDER (Pour vérifier si ça marche)
    with st.expander("🛠️ Vérification des données (Clique ici si c'est vide)"):
        st.write("Nombre total de lignes :", len(df_filtered))
        st.write("Aperçu des données :", df_filtered.head(3))
        st.write("AMs trouvés :", df_filtered['AM_OWNER'].unique())

    tab_am, tab_global = st.tabs(["👤 Vue par AM", "🌍 Vue Global Yassir"])

    # === VUE AM ===
    with tab_am:
        # Sélecteur
        list_ams = sorted(df_filtered['AM_OWNER'].astype(str).unique())
        selected_am = st.selectbox("Choisir l'AM :", ["Tous"] + list_ams)
        
        if selected_am != "Tous":
            df_view = df_filtered[df_filtered['AM_OWNER'] == selected_am]
        else:
            df_view = df_filtered

        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        total_gmv = df_view['GMV'].sum()
        total_orders = len(df_view)
        cancel_rate = (df_view['is_cancelled'].sum() / total_orders * 100) if total_orders > 0 else 0
        aov = total_gmv / total_orders if total_orders > 0 else 0
        
        col1.metric("GMV", f"{total_gmv:,.0f} DH")
        col2.metric("Commandes", f"{total_orders}")
        col3.metric("Taux Annul.", f"{cancel_rate:.2f}%")
        col4.metric("Panier Moyen", f"{aov:.0f} DH")

        st.divider()

        # TOP / FLOP 10
        st.subheader(f"🏆 Top & Flop - {selected_am}")
        
        # Groupement par resto
        # On utilise 'Restaurant Name Final' calculé plus haut
        df_restos = df_view.groupby('Restaurant Name Final').agg({
            'GMV': 'sum',
            'order id': 'count',
            'is_cancelled': 'mean',
            'Delivery Time': 'mean'
        }).reset_index()
        df_restos['Cancellation %'] = df_restos['is_cancelled'] * 100
        
        metric_choice = st.selectbox("Trier par :", ["GMV", "order id", "Cancellation %", "Delivery Time"])
        
        # Sens du tri (Ascendant pour Taux d'annul, Descendant pour GMV)
        asc = True if metric_choice in ["Cancellation %", "Delivery Time"] else False
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🟢 TOP 10")
            st.dataframe(df_restos.sort_values(metric_choice, ascending=asc).head(10).style.background_gradient(cmap="Purples"))
        with c2:
            st.markdown("##### 🔴 FLOP 10")
            st.dataframe(df_restos.sort_values(metric_choice, ascending=not asc).head(10).style.background_gradient(cmap="Reds"))

        # GRAPHIQUE ÉVOLUTION
        st.subheader("📈 Évolution Temporelle")
        df_time = df_view.groupby(df_view['order day'].dt.to_period('M').astype(str)).agg({'GMV':'sum', 'order id':'count'}).reset_index()
        fig = px.bar(df_time, x='order day', y='GMV', title="Évolution GMV par Mois", color_discrete_sequence=['#6c35de'])
        st.plotly_chart(fig, use_container_width=True)

    # === VUE GLOBAL ===
    with tab_global:
        st.header("🌍 Performance Globale")
        st.metric("GMV Total Yassir", f"{df_filtered['GMV'].sum():,.0f} DH")
        
        # Comparaison AM
        st.subheader("Comparatif des AMs")
        df_comp = df_filtered.groupby('AM_OWNER').agg({'GMV':'sum', 'order id':'count', 'is_cancelled':'mean'}).reset_index()
        df_comp['Cancellation %'] = df_comp['is_cancelled'] * 100
        
        fig_comp = px.scatter(df_comp, x='order id', y='GMV', color='AM_OWNER', size='GMV', hover_name='AM_OWNER', title="Matrice Performance AM")
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(df_comp)

elif uploaded_file is None:
    st.info("👈 Veuillez uploader le fichier Admin Earnings dans la barre latérale.")
elif df_pipeline.empty:
    st.error("❌ Aucun fichier Pipeline valide trouvé. Vérifiez les noms des fichiers.")
