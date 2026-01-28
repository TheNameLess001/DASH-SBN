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

def get_monthly_evolution_table(df):
    """Génère un tableau récapitulatif par mois avec calculs de progression (Growth)."""
    if df.empty:
        return pd.DataFrame()
        
    # 1. Aggrégation mensuelle
    df_monthly = df.groupby(df['order day'].dt.to_period('M').astype(str)).agg({
        'GMV': 'sum',
        'order id': 'count',
        'is_cancelled': 'mean',
        'Delivery Time': 'mean'
    }).reset_index().rename(columns={'order day': 'Mois'})
    
    # 2. Tri chronologique (Ascendant) pour calculer la progression
    df_monthly = df_monthly.sort_values('Mois', ascending=True)
    
    # 3. Calculs KPIs dérivés
    df_monthly['AOV'] = df_monthly['GMV'] / df_monthly['order id']
    df_monthly['Cancel Rate'] = df_monthly['is_cancelled'] * 100
    
    # 4. Calcul de la CROISSANCE (Prog/Reg) vs Mois Précédent
    # pct_change() calcule la variation en pourcentage
    df_monthly['Growth GMV'] = df_monthly['GMV'].pct_change() * 100
    df_monthly['Growth Orders'] = df_monthly['order id'].pct_change() * 100
    df_monthly['Growth AOV'] = df_monthly['AOV'].pct_change() * 100
    df_monthly['Growth Cancel'] = df_monthly['Cancel Rate'].pct_change() * 100
    df_monthly['Growth Time'] = df_monthly['Delivery Time'].pct_change() * 100
    
    # 5. Tri décroissant (Mois le plus récent en haut) pour l'affichage
    df_monthly = df_monthly.sort_values('Mois', ascending=False)
    
    # 6. Mise en forme (Arrondis)
    df_monthly['GMV'] = df_monthly['GMV'].round(0)
    df_monthly['AOV'] = df_monthly['AOV'].round(0)
    df_monthly['Cancel Rate'] = df_monthly['Cancel Rate'].round(2)
    df_monthly['Delivery Time'] = df_monthly['Delivery Time'].round(1)
    
    # 7. Renommage et Organisation des colonnes
    cols_map = {
        'Mois': 'Mois',
        'GMV': 'CA (GMV)',
        'Growth GMV': 'Prog/Reg CA %',
        'order id': 'Commandes',
        'Growth Orders': 'Prog/Reg Cmd %',
        'AOV': 'Panier Moyen',
        'Growth AOV': 'Prog/Reg AOV %',
        'Cancel Rate': 'Taux Annul %',
        'Growth Cancel': 'Prog/Reg Annul %',
        'Delivery Time': 'Temps Livr.',
        'Growth Time': 'Prog/Reg Temps %'
    }
    
    df_final = df_monthly.rename(columns=cols_map)
    
    # Ordre final : Métrique, puis sa progression
    ordered_cols = [
        'Mois', 
        'CA (GMV)', 'Prog/Reg CA %', 
        'Commandes', 'Prog/Reg Cmd %', 
        'Panier Moyen', 'Prog/Reg AOV %', 
        'Taux Annul %', 'Prog/Reg Annul %', 
        'Temps Livr.', 'Prog/Reg Temps %'
    ]
    
    return df_final[ordered_cols]

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
        if 'ID' in df_pipeline.columns:
            join_key = 'ID'
        elif 'RESTAURANT ID' in df_pipeline.columns:
             join_key = 'RESTAURANT ID'
        else:
            st.error(f"❌ Colonne 'ID' introuvable dans le fichier Excel. Colonnes détectées : {list(df_pipeline.columns)}")
            st.stop()

        # Fusion Left
        df_full = pd.merge(
            df_orders,
            df_pipeline,
            left_on='Restaurant ID',
            right_on=join_key,
            how='left'
        )
        
        # 4. Nettoyage post-fusion
        df_full['AM_OWNER'] = df_full['AM_OWNER'].fillna('Non Assigné / Organique')
        
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

            # --- TABLEAU ÉVOLUTION PAR MOIS (AMÉLIORÉ) ---
            st.subheader("📅 Tableau de Bord Mensuel (KPIs & Croissance)")
            st.markdown("Vue détaillée mois par mois avec progression (Prog/Reg) par rapport au mois précédent.")
            
            with st.expander("Voir le tableau d'évolution complet", expanded=True):
                df_evo_am = get_monthly_evolution_table(df_view)
                
                # Mise en forme conditionnelle pour les colonnes "Prog/Reg"
                st.dataframe(
                    df_evo_am.style
                    .background_gradient(cmap="Purples", subset=['CA (GMV)', 'Commandes'])
                    .format("{:.2f}%", subset=['Prog/Reg CA %', 'Prog/Reg Cmd %', 'Prog/Reg AOV %', 'Prog/Reg Annul %', 'Prog/Reg Temps %'], na_rep="-"),
                    use_container_width=True
                )

            st.divider()
            
            # --- GRAPHIQUES ---
            st.subheader(f"📈 Tendances Graphiques - {selected_am}")
            df_growth = df_view.groupby(df_view['order day'].dt.to_period('M').astype(str)).agg({
                'GMV': 'sum', 'order id': 'count'
            }).reset_index().rename(columns={'order day': 'Mois'})
            
            c1, c2 = st.columns(2)
            with c1:
                fig_gmv = px.bar(df_growth, x='Mois', y='GMV', title="Évolution du GMV", color_discrete_sequence=['#6c35de'])
                st.plotly_chart(fig_gmv, use_container_width=True)
            with c2:
                fig_orders = px.line(df_growth, x='Mois', y='order id', title="Évolution Volume Commandes", markers=True, color_discrete_sequence=['#ff00ff'])
                st.plotly_chart(fig_orders, use_container_width=True)

            st.divider()

            # --- TOP & FLOP 10 ---
            st.subheader("🏆 Classement Restaurants")
            
            df_rank = df_view.groupby(['Restaurant_Final', 'City_Final']).agg({
                'GMV': 'sum', 'order id': 'count', 'is_cancelled': 'mean', 'Delivery Time': 'mean'
            }).reset_index()
            df_rank['Taux Annulation %'] = (df_rank['is_cancelled'] * 100).round(2)
            df_rank['AOV'] = (df_rank['GMV'] / df_rank['order id']).round(0)
            df_rank['GMV'] = df_rank['GMV'].round(0)
            df_rank['Delivery Time'] = df_rank['Delivery Time'].round(1)

            sort_col = st.selectbox("Classer par :", ["GMV", "Commandes", "Taux Annulation %", "AOV", "Temps Livraison"])
            col_map = {"GMV": "GMV", "Commandes": "order id", "Taux Annulation %": "Taux Annulation %", "AOV": "AOV", "Temps Livraison": "Delivery Time"}
            tech_col = col_map[sort_col]
            ascending_top = True if tech_col in ["Taux Annulation %", "Delivery Time"] else False
            cols_to_show = ['Restaurant_Final', 'City_Final', 'GMV', 'order id', 'Taux Annulation %', 'AOV', 'Delivery Time']

            c_top, c_flop = st.columns(2)
            with c_top:
                st.markdown(f"#### 🌟 TOP 10 ({sort_col})")
                st.dataframe(df_rank.sort_values(tech_col, ascending=ascending_top).head(10)[cols_to_show].style.background_gradient(cmap="Purples", subset=[tech_col]), use_container_width=True)
            with c_flop:
                st.markdown(f"#### ⚠️ FLOP 10 ({sort_col})")
                st.dataframe(df_rank.sort_values(tech_col, ascending=not ascending_top).head(10)[cols_to_show].style.background_gradient(cmap="Reds", subset=[tech_col]), use_container_width=True)

            with st.expander("📋 Détails Restaurants (Filtrables)"):
                sel = st.multiselect("Filtrer par Restaurant :", sorted(df_rank['Restaurant_Final'].unique()))
                st.dataframe(df_rank[df_rank['Restaurant_Final'].isin(sel)] if sel else df_rank, use_container_width=True)

        # ====================================================================
        # ONGLET 2 : VUE GLOBALE
        # ====================================================================
        with tab_global:
            st.header("🌍 Performance Globale Yassir")
            
            tot_gmv = df_filtered['GMV'].sum()
            tot_ord = len(df_filtered)
            tot_cancel = (df_filtered['is_cancelled'].sum() / tot_ord * 100) if tot_ord > 0 else 0
            
            k1, k2, k3 = st.columns(3)
            k1.metric("GMV Total", f"{tot_gmv:,.0f} DH")
            k2.metric("Volume Commandes", f"{tot_ord:,.0f}")
            k3.metric("Taux Annulation Global", f"{tot_cancel:.2f}%")
            
            st.divider()

            # --- TABLEAU ÉVOLUTION GLOBAL (AMÉLIORÉ) ---
            st.subheader("📅 Détail Évolution Mensuelle Global (KPIs & Croissance)")
            with st.expander("Voir le tableau d'évolution complet", expanded=True):
                df_evo_global = get_monthly_evolution_table(df_filtered)
                st.dataframe(
                    df_evo_global.style
                    .background_gradient(cmap="Purples", subset=['CA (GMV)', 'Commandes'])
                    .format("{:.2f}%", subset=['Prog/Reg CA %', 'Prog/Reg Cmd %', 'Prog/Reg AOV %', 'Prog/Reg Annul %', 'Prog/Reg Temps %'], na_rep="-"),
                    use_container_width=True
                )
            
            st.divider()
            
            st.subheader("📊 Comparatif des Account Managers")
            df_am_perf = df_filtered.groupby('AM_OWNER').agg({
                'GMV': 'sum', 'order id': 'count', 'is_cancelled': 'mean', 'Restaurant_Final': 'nunique'
            }).reset_index()
            df_am_perf['Taux Annulation %'] = (df_am_perf['is_cancelled'] * 100).round(2)
            df_am_perf = df_am_perf.rename(columns={'Restaurant_Final': 'Portefeuille (Nb Restos)'})
            
            fig_perf = px.scatter(df_am_perf, x='Portefeuille (Nb Restos)', y='GMV', size='order id', color='AM_OWNER', hover_name='AM_OWNER', title="Performance AM : Portefeuille vs Chiffre d'Affaires")
            st.plotly_chart(fig_perf, use_container_width=True)
            st.dataframe(df_am_perf.sort_values('GMV', ascending=False).style.background_gradient(cmap="Purples", subset=['GMV']), use_container_width=True)

    else:
        st.warning("⚠️ Veuillez uploader les deux fichiers pour voir l'analyse.")

else:
    st.info("👈 En attente des fichiers dans la barre latérale...")

# --- FOOTER ---
st.markdown('<div class="footer">Bounoir Saif eddine - Yassir Analytics Dashboard</div>', unsafe_allow_html=True)
