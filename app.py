import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Yassir Analytics", page_icon="🟣", layout="wide")

# --- CSS YASSIR ---
st.markdown("""
    <style>
    :root { --primary-color: #6c35de; }
    .stMetric { background-color: #f8f6ff; border-left: 5px solid #6c35de; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    h1, h2, h3 { color: #4b2c92; font-family: 'Source Sans Pro', sans-serif; }
    .stButton>button { background-color: #6c35de; color: white; border-radius: 8px; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { background-color: #6c35de; color: white; }
    .footer { position: fixed; bottom: 10px; right: 10px; color: #888; font-size: 0.8em; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE CHARGEMENT ---

@st.cache_data
def load_pipeline_excel(uploaded_file):
    """
    Lit un fichier Excel avec plusieurs feuilles.
    Chaque feuille correspond à un AM.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    
    try:
        # sheet_name=None permet de lire TOUTES les feuilles dans un dictionnaire
        # keys = noms des feuilles (AM), values = dataframes
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        
        all_data = []
        
        for sheet_name, df_sheet in xls.items():
            # Nettoyage des colonnes (Majuscules + sans espaces)
            df_sheet.columns = df_sheet.columns.str.strip().str.upper()
            
            # On assigne le nom de l'onglet comme nom de l'AM
            df_sheet['AM_OWNER'] = sheet_name
            
            all_data.append(df_sheet)
            
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            return final_df
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture du fichier Excel Pipeline : {e}")
        return pd.DataFrame()

@st.cache_data
def load_orders_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"❌ Erreur lecture Admin Earnings : {e}")
    return None

# --- APPLICATION PRINCIPALE ---

st.title("🟣 Yassir Analytics Dashboard")
st.markdown("**Performance Commerciale & Opérationnelle**")

# --- SIDEBAR : UPLOADS ---
with st.sidebar:
    st.header("📂 Importation des Données")
    
    st.markdown("### 1. Données Commandes")
    orders_file = st.file_uploader("Uploader 'Admin Earnings' (.csv)", type=['csv'])
    
    st.markdown("### 2. Pipeline AMs")
    pipeline_file = st.file_uploader("Uploader 'Pipeline Global' (.xlsx)", type=['xlsx'])
    
    st.info("💡 Le fichier Excel Pipeline doit contenir une feuille par AM.")

# --- TRAITEMENT DES DONNÉES ---

if orders_file is not None and pipeline_file is not None:
    
    # 1. Chargement
    df_orders = load_orders_csv(orders_file)
    df_pipeline = load_pipeline_excel(pipeline_file)
    
    if not df_orders.empty and not df_pipeline.empty:
        
        # 2. Préparation Admin Earnings
        df_orders['order day'] = pd.to_datetime(df_orders['order day'], errors='coerce')
        
        # 3. Fusion (Merge)
        # Vérification de la colonne ID dans le pipeline
        if 'ID' in df_pipeline.columns:
            join_key = 'ID'
        elif 'RESTAURANT ID' in df_pipeline.columns: # Cas où l'AM a renommé la colonne
             join_key = 'RESTAURANT ID'
        else:
            st.error(f"❌ Colonne 'ID' introuvable dans le fichier Excel. Colonnes détectées : {list(df_pipeline.columns)}")
            st.stop()

        # Fusion Left (On garde toutes les commandes)
        df_full = pd.merge(
            df_orders,
            df_pipeline,
            left_on='Restaurant ID',
            right_on=join_key,
            how='left'
        )
        
        # 4. Nettoyage post-fusion
        df_full['AM_OWNER'] = df_full['AM_OWNER'].fillna('Non Assigné / Organique')
        
        # Gestion des noms de restaurants et villes (Priorité Pipeline > Earnings)
        if 'RESTAURANT NAME' in df_full.columns:
            df_full['Restaurant_Final'] = df_full['RESTAURANT NAME'].fillna(df_full['restaurant name'])
        else:
            df_full['Restaurant_Final'] = df_full['restaurant name']
            
        if 'MAIN CITY' in df_full.columns:
            df_full['City_Final'] = df_full['MAIN CITY'].fillna(df_full['city'])
        else:
            df_full['City_Final'] = df_full['city']

        # 5. Calcul des KPIs
        df_full['is_cancelled'] = df_full['status'].apply(lambda x: 1 if str(x).lower() != 'delivered' else 0)
        df_full['GMV'] = pd.to_numeric(df_full['item total'], errors='coerce').fillna(0)
        df_full['Delivery Time'] = pd.to_numeric(df_full['delivery time(M)'], errors='coerce')

        # --- FILTRE DATE GLOBAL ---
        with st.sidebar:
            st.divider()
            st.header("📅 Période d'analyse")
            min_d = df_full['order day'].min().date()
            max_d = df_full['order day'].max().date()
            dates = st.date_input("Sélectionner la plage :", [min_d, max_d])
            
            if len(dates) == 2:
                mask = (df_full['order day'].dt.date >= dates[0]) & (df_full['order day'].dt.date <= dates[1])
                df_filtered = df_full[mask]
            else:
                df_filtered = df_full

        # --- DASHBOARD VISUALIZATION ---
        
        tab_am, tab_global = st.tabs(["👤 Vue par Account Manager", "🌍 Vue Globale Yassir"])

        # ====================================================================
        # ONGLET 1 : VUE PAR AM
        # ====================================================================
        with tab_am:
            # Sélecteur AM
            am_list = ["Tous"] + sorted(df_filtered['AM_OWNER'].astype(str).unique().tolist())
            selected_am = st.selectbox("Choisir un Account Manager :", am_list)
            
            # Filtre Dataframe
            if selected_am != "Tous":
                df_view = df_filtered[df_filtered['AM_OWNER'] == selected_am]
            else:
                df_view = df_filtered

            # --- HEADER KPIs ---
            col1, col2, col3, col4, col5 = st.columns(5)
            
            gmv = df_view['GMV'].sum()
            orders = len(df_view)
            cancel_rate = (df_view['is_cancelled'].sum() / orders * 100) if orders > 0 else 0
            aov = gmv / orders if orders > 0 else 0
            del_time = df_view['Delivery Time'].mean()

            col1.metric("💰 Chiffre d'Affaires", f"{gmv:,.0f} DH")
            col2.metric("📦 Commandes", f"{orders:,.0f}")
            col3.metric("🛒 Panier Moyen (AOV)", f"{aov:.0f} DH")
            col4.metric("❌ Taux d'Annulation", f"{cancel_rate:.2f}%")
            col5.metric("⏱️ Temps Livraison", f"{del_time:.0f} min")

            st.divider()

            # --- GRAPHIQUES ÉVOLUTION (Growth) ---
            st.subheader(f"📈 Croissance & Tendance - {selected_am}")
            
            # Agrégation par mois
            df_growth = df_view.groupby(df_view['order day'].dt.to_period('M').astype(str)).agg({
                'GMV': 'sum',
                'order id': 'count',
                'is_cancelled': 'mean'
            }).reset_index().rename(columns={'order day': 'Mois'})
            
            c1, c2 = st.columns(2)
            with c1:
                fig_gmv = px.bar(df_growth, x='Mois', y='GMV', title="Évolution du GMV (DH)", color_discrete_sequence=['#6c35de'])
                fig_gmv.update_layout(xaxis_title=None)
                st.plotly_chart(fig_gmv, use_container_width=True)
            with c2:
                fig_orders = px.line(df_growth, x='Mois', y='order id', title="Évolution Volume Commandes", markers=True, color_discrete_sequence=['#ff00ff'])
                fig_orders.update_layout(xaxis_title=None)
                st.plotly_chart(fig_orders, use_container_width=True)

            st.divider()

            # --- TOP & FLOP 10 ---
            st.subheader("🏆 Classement Restaurants")
            
            # Agrégation par Resto
            df_rank = df_view.groupby(['Restaurant_Final', 'City_Final']).agg({
                'GMV': 'sum',
                'order id': 'count',
                'is_cancelled': 'mean',
                'Delivery Time': 'mean'
            }).reset_index()
            df_rank['Taux Annulation %'] = (df_rank['is_cancelled'] * 100).round(2)
            df_rank['AOV'] = (df_rank['GMV'] / df_rank['order id']).round(0)
            df_rank['GMV'] = df_rank['GMV'].round(0)
            df_rank['Delivery Time'] = df_rank['Delivery Time'].round(1)

            # Filtre dynamique de tri
            sort_col = st.selectbox("Classer les résultats par :", ["GMV", "Commandes", "Taux Annulation %", "AOV", "Temps Livraison"])
            
            # Mapping pour le nom de colonne technique
            col_map = {
                "GMV": "GMV", 
                "Commandes": "order id", 
                "Taux Annulation %": "Taux Annulation %", 
                "AOV": "AOV",
                "Temps Livraison": "Delivery Time"
            }
            tech_col = col_map[sort_col]
            
            # Sens du tri (Annulation & Temps = Plus petit est mieux, donc Ascendant pour TOP)
            # MAIS pour le tableau "TOP", on veut afficher les "Meilleurs chiffres".
            # Pour GMV : Descendant. Pour Annulation : Ascendant (le plus bas).
            if tech_col in ["Taux Annulation %", "Delivery Time"]:
                ascending_top = True
            else:
                ascending_top = False

            cols_to_show = ['Restaurant_Final', 'City_Final', 'GMV', 'order id', 'Taux Annulation %', 'AOV', 'Delivery Time']

            c_top, c_flop = st.columns(2)
            
            with c_top:
                st.markdown(f"#### 🌟 TOP 10 ({sort_col})")
                st.dataframe(
                    df_rank.sort_values(tech_col, ascending=ascending_top).head(10)[cols_to_show]
                    .style.background_gradient(cmap="Purples", subset=[tech_col]),
                    use_container_width=True
                )
            
            with c_flop:
                st.markdown(f"#### ⚠️ FLOP 10 ({sort_col})")
                st.dataframe(
                    df_rank.sort_values(tech_col, ascending=not ascending_top).head(10)[cols_to_show]
                    .style.background_gradient(cmap="Reds", subset=[tech_col]),
                    use_container_width=True
                )

            # --- DÉTAILS AVEC FILTRES ---
            with st.expander("📋 Voir les détails complets (Filtrables)"):
                selected_restos = st.multiselect("Filtrer par Restaurant :", sorted(df_rank['Restaurant_Final'].unique()))
                if selected_restos:
                    st.dataframe(df_rank[df_rank['Restaurant_Final'].isin(selected_restos)], use_container_width=True)
                else:
                    st.dataframe(df_rank, use_container_width=True)

        # ====================================================================
        # ONGLET 2 : VUE GLOBALE
        # ====================================================================
        with tab_global:
            st.header("🌍 Performance Globale Yassir")
            
            # BIG NUMBERS
            tot_gmv = df_filtered['GMV'].sum()
            tot_ord = len(df_filtered)
            tot_cancel = (df_filtered['is_cancelled'].sum() / tot_ord * 100) if tot_ord > 0 else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("GMV Total", f"{tot_gmv:,.0f} DH")
            k2.metric("Volume Commandes", f"{tot_ord:,.0f}")
            k3.metric("Taux Annulation Global", f"{tot_cancel:.2f}%")
            
            st.divider()
            
            # MATRICE PERFORMANCE AM
            st.subheader("📊 Comparatif des Account Managers")
            
            df_am_perf = df_filtered.groupby('AM_OWNER').agg({
                'GMV': 'sum', 
                'order id': 'count', 
                'is_cancelled': 'mean',
                'Restaurant_Final': 'nunique'
            }).reset_index()
            df_am_perf['Taux Annulation %'] = (df_am_perf['is_cancelled'] * 100).round(2)
            df_am_perf = df_am_perf.rename(columns={'Restaurant_Final': 'Portefeuille (Nb Restos)'})
            
            # Scatter Plot interactif
            fig_perf = px.scatter(
                df_am_perf, 
                x='Portefeuille (Nb Restos)', 
                y='GMV', 
                size='order id', 
                color='AM_OWNER',
                hover_name='AM_OWNER',
                text='AM_OWNER',
                title="Performance AM : Portefeuille vs Chiffre d'Affaires",
                labels={'order id': 'Volume Commandes'},
                height=500
            )
            fig_perf.update_traces(textposition='top center')
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Tableau récapitulatif
            st.dataframe(
                df_am_perf.sort_values('GMV', ascending=False)
                .style.background_gradient(cmap="Purples", subset=['GMV']),
                use_container_width=True
            )

    else:
        st.warning("⚠️ Veuillez uploader les deux fichiers pour voir l'analyse.")

else:
    st.info("👈 En attente des fichiers dans la barre latérale...")

# --- FOOTER ---
st.markdown('<div class="footer">Bounoir Saif eddine - Yassir Analytics Dashboard</div>', unsafe_allow_html=True)
