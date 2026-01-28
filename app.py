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
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        xls = pd.read_excel(uploaded_file, sheet_name=None)
        all_data = []
        for sheet_name, df_sheet in xls.items():
            df_sheet.columns = df_sheet.columns.str.strip().str.upper()
            df_sheet['AM_OWNER'] = sheet_name
            all_data.append(df_sheet)
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Erreur Excel Pipeline : {e}")
        return pd.DataFrame()

@st.cache_data
def load_orders_csv(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Erreur CSV Admin : {e}")
    return None

def get_monthly_evolution_table(df):
    """
    Génère un tableau mensuel avec Prog/Reg.
    CORRECTION : Pour Annulation et Temps, une BAISSE est une PROGRESSION (Inversion du signe).
    """
    if df.empty:
        return pd.DataFrame()
        
    # 1. Aggrégation mensuelle
    df_monthly = df.groupby(df['order day'].dt.to_period('M').astype(str)).agg({
        'GMV': 'sum',
        'order id': 'count',
        'is_cancelled': 'mean',
        'Delivery Time': 'mean'
    }).reset_index().rename(columns={'order day': 'Mois'})
    
    # 2. Tri chronologique pour le calcul
    df_monthly = df_monthly.sort_values('Mois', ascending=True)
    
    # 3. KPIs dérivés
    df_monthly['AOV'] = df_monthly['GMV'] / df_monthly['order id']
    df_monthly['Cancel Rate'] = df_monthly['is_cancelled'] * 100
    
    # 4. Calcul CROISSANCE (Prog/Reg)
    # Pour GMV, Commandes, AOV : Hausse = Positif (+)
    df_monthly['Growth GMV'] = df_monthly['GMV'].pct_change() * 100
    df_monthly['Growth Orders'] = df_monthly['order id'].pct_change() * 100
    df_monthly['Growth AOV'] = df_monthly['AOV'].pct_change() * 100
    
    # Pour Annulation et Temps : Baisse = Positif (+), Hausse = Négatif (-)
    df_monthly['Growth Cancel'] = df_monthly['Cancel Rate'].pct_change() * -1 * 100
    df_monthly['Growth Time'] = df_monthly['Delivery Time'].pct_change() * -1 * 100
    
    # 5. Tri décroissant (Mois récent en haut)
    df_monthly = df_monthly.sort_values('Mois', ascending=False)
    
    # 6. Arrondis
    df_monthly['GMV'] = df_monthly['GMV'].round(0)
    df_monthly['AOV'] = df_monthly['AOV'].round(0)
    df_monthly['Cancel Rate'] = df_monthly['Cancel Rate'].round(2)
    df_monthly['Delivery Time'] = df_monthly['Delivery Time'].round(1)
    
    # 7. Renommage
    cols_map = {
        'Mois': 'Mois',
        'GMV': 'CA (GMV)', 'Growth GMV': 'Prog/Reg CA %',
        'order id': 'Commandes', 'Growth Orders': 'Prog/Reg Cmd %',
        'AOV': 'Panier Moyen', 'Growth AOV': 'Prog/Reg AOV %',
        'Cancel Rate': 'Taux Annul %', 'Growth Cancel': 'Prog/Reg Annul %',
        'Delivery Time': 'Temps Livr.', 'Growth Time': 'Prog/Reg Temps %'
    }
    
    df_final = df_monthly.rename(columns=cols_map)
    
    ordered_cols = [
        'Mois', 
        'CA (GMV)', 'Prog/Reg CA %', 
        'Commandes', 'Prog/Reg Cmd %', 
        'Panier Moyen', 'Prog/Reg AOV %', 
        'Taux Annul %', 'Prog/Reg Annul %', 
        'Temps Livr.', 'Prog/Reg Temps %'
    ]
    
    return df_final[ordered_cols]

def display_advanced_kpis(df_input, title_prefix=""):
    """
    Affiche la section Rentabilité & Efficacité (Big Numbers + Tableau détaillé).
    """
    st.markdown(f"### 💎 {title_prefix} Analyse Avancée : Financier (Gain/Perte) & Ops")
    st.markdown("Vision P&L : Chiffre d'Affaires, Gains Réels (Commissions) et Pertes (Manque à gagner).")

    # Calculs Globaux de la section
    total_gmv = df_input['GMV'].sum()
    net_rev_total = df_input['Net Revenue'].sum()       # Gain (Commissions)
    missed_gmv_total = df_input['Missed GMV'].sum()     # Perte Volume
    missed_comm_total = df_input['Missed Comm'].sum()   # Perte Financière (Commissions ratées)
    
    # Coupon Dependency
    total_coupon = df_input['coupon discount'].sum() if 'coupon discount' in df_input.columns else 0
    coupon_dep_rate = (total_coupon / total_gmv * 100) if total_gmv > 0 else 0
    
    # Vitesse Moyenne
    valid_speed = df_input[(df_input['delivery time(M)'] > 0) & (df_input['Distance travel'] > 0)]
    avg_speed = (valid_speed['Distance travel'] / valid_speed['delivery time(M)'] * 60).mean() if not valid_speed.empty else 0

    # Affichage Big Numbers (5 colonnes pour inclure le CA et la Perte Financière)
    k0, k1, k2, k3, k4 = st.columns(5)
    
    k0.metric("💰 CA Total (GMV)", f"{total_gmv:,.0f} DH", help="Volume d'Affaires Global")
    k1.metric("✅ Gain (Commissions)", f"{net_rev_total:,.0f} DH", help="Revenu Net Yassir (GMV x Com%)")
    k2.metric("📉 Perte Financière", f"{missed_comm_total:,.0f} DH", help="Commissions perdues sur annulations")
    k3.metric("🚫 Volume Perdu", f"{missed_gmv_total:,.0f} DH", help="GMV Total annulé (Manque à gagner ecosystème)")
    k4.metric("🎟️ Dépendance Promo", f"{coupon_dep_rate:.1f}%", help="Part du GMV issue des coupons")

    # Tableau Détaillé par Restaurant
    st.markdown("#### 📋 Détail Financier par Restaurant (P&L)")
    
    # Aggrégation
    df_adv = df_input.groupby(['Restaurant_Final', 'City_Final']).agg({
        'GMV': 'sum',
        'Net Revenue': 'sum',
        'Missed GMV': 'sum',
        'Missed Comm': 'sum',
        'coupon discount': 'sum' if 'coupon discount' in df_input.columns else 'count',
        'Distance travel': 'mean',
        'Delivery Time': 'mean'
    }).reset_index()
    
    # Calculs dérivés
    df_adv['Coupon Dep. %'] = (df_adv['coupon discount'] / df_adv['GMV'] * 100).fillna(0)
    df_adv['Vitesse (km/h)'] = (df_adv['Distance travel'] / df_adv['Delivery Time'] * 60).fillna(0)
    
    # Arrondis
    df_adv['GMV'] = df_adv['GMV'].round(0)
    df_adv['Net Revenue'] = df_adv['Net Revenue'].round(0)
    df_adv['Missed GMV'] = df_adv['Missed GMV'].round(0)
    df_adv['Missed Comm'] = df_adv['Missed Comm'].round(0)
    df_adv['Coupon Dep. %'] = df_adv['Coupon Dep. %'].round(1)
    
    # Renommage
    df_show = df_adv.rename(columns={
        'GMV': 'CA (GMV)',
        'Net Revenue': 'Gain (Commissions)',
        'Missed Comm': 'Perte (Commissions)',
        'Missed GMV': 'Vol. Perdu (Annul.)',
        'Coupon Dep. %': 'Effort Promo %'
    })
    
    # Sélection Colonnes
    cols_final = [
        'Restaurant_Final', 'City_Final', 
        'CA (GMV)', 
        'Gain (Commissions)', 
        'Perte (Commissions)', 
        'Vol. Perdu (Annul.)', 
        'Effort Promo %'
    ]
    
    # Tri par défaut : Gain
    df_show = df_show.sort_values('Gain (Commissions)', ascending=False)
    
    st.dataframe(
        df_show[cols_final].style
        .background_gradient(cmap="Greens", subset=['Gain (Commissions)'])
        .background_gradient(cmap="Reds", subset=['Perte (Commissions)', 'Vol. Perdu (Annul.)']),
        use_container_width=True
    )

# --- APP ---

st.title("🟣 Yassir Analytics Dashboard")
st.markdown("**Performance Commerciale & Opérationnelle**")

# SIDEBAR
with st.sidebar:
    st.header("📂 Données")
    orders_file = st.file_uploader("1. Admin Earnings (.csv)", type=['csv'])
    pipeline_file = st.file_uploader("2. Pipeline Global (.xlsx)", type=['xlsx'])

if orders_file and pipeline_file:
    df_orders = load_orders_csv(orders_file)
    df_pipeline = load_pipeline_excel(pipeline_file)
    
    if not df_orders.empty and not df_pipeline.empty:
        df_orders['order day'] = pd.to_datetime(df_orders['order day'], errors='coerce')
        
        # Merge
        join_key = 'ID' if 'ID' in df_pipeline.columns else 'RESTAURANT ID'
        if join_key not in df_pipeline.columns:
            st.error("Colonne ID manquante dans Pipeline")
            st.stop()
            
        df_full = pd.merge(df_orders, df_pipeline, left_on='Restaurant ID', right_on=join_key, how='left')
        
        # Clean
        df_full['AM_OWNER'] = df_full['AM_OWNER'].fillna('Non Assigné')
        df_full['Restaurant_Final'] = df_full['RESTAURANT NAME'].fillna(df_full['restaurant name']) if 'RESTAURANT NAME' in df_full.columns else df_full['restaurant name']
        df_full['City_Final'] = df_full['MAIN CITY'].fillna(df_full['city']) if 'MAIN CITY' in df_full.columns else df_full['city']

        # KPIs Base
        df_full['is_cancelled'] = df_full['status'].apply(lambda x: 1 if str(x).lower() != 'delivered' else 0)
        df_full['GMV'] = pd.to_numeric(df_full['item total'], errors='coerce').fillna(0)
        df_full['Delivery Time'] = pd.to_numeric(df_full['delivery time(M)'], errors='coerce')
        if 'Distance travel' not in df_full.columns: df_full['Distance travel'] = 0
        else: df_full['Distance travel'] = pd.to_numeric(df_full['Distance travel'], errors='coerce').fillna(0)
        if 'coupon discount' not in df_full.columns: df_full['coupon discount'] = 0
        else: df_full['coupon discount'] = pd.to_numeric(df_full['coupon discount'], errors='coerce').fillna(0)

        # --- KPI FINANCIERS ---
        # 1. Commission Cleaning
        if 'COMMISSION %' in df_full.columns:
            df_full['Commission_Clean'] = pd.to_numeric(df_full['COMMISSION %'], errors='coerce').fillna(0)
            df_full['Commission_Clean'] = df_full['Commission_Clean'].apply(lambda x: x/100 if x > 1 else x)
        else:
            df_full['Commission_Clean'] = 0
            
        # 2. Gain (Revenue Net)
        df_full['Net Revenue'] = df_full['GMV'] * df_full['Commission_Clean']
        
        # 3. Perte (Manque à Gagner)
        df_full['Missed GMV'] = df_full['is_cancelled'] * df_full['GMV']
        df_full['Missed Comm'] = df_full['is_cancelled'] * (df_full['GMV'] * df_full['Commission_Clean'])

        # Filtre Date
        with st.sidebar:
            st.divider()
            min_d, max_d = df_full['order day'].min().date(), df_full['order day'].max().date()
            dates = st.date_input("Période", [min_d, max_d])
            mask = (df_full['order day'].dt.date >= dates[0]) & (df_full['order day'].dt.date <= dates[1]) if len(dates)==2 else True
            df_filtered = df_full[mask]

        # TABS
        tab_am, tab_global = st.tabs(["👤 Vue par AM", "🌍 Vue Globale"])

        # --- VUE AM ---
        with tab_am:
            am_list = ["Tous"] + sorted(df_filtered['AM_OWNER'].astype(str).unique().tolist())
            selected_am = st.selectbox("Account Manager :", am_list)
            df_view = df_filtered[df_filtered['AM_OWNER'] == selected_am] if selected_am != "Tous" else df_filtered

            # KPIs Header
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("💰 Chiffre d'Affaires", f"{df_view['GMV'].sum():,.0f} DH")
            c2.metric("📦 Commandes", f"{len(df_view):,.0f}")
            c3.metric("🛒 Panier Moyen", f"{(df_view['GMV'].sum()/len(df_view) if len(df_view)>0 else 0):.0f} DH")
            c4.metric("❌ Taux Annulation", f"{(df_view['is_cancelled'].mean()*100):.2f}%")
            c5.metric("⏱️ Temps Livraison", f"{df_view['Delivery Time'].mean():.0f} min")

            st.divider()

            # Evolution Tabulaire
            st.subheader("📅 Détail Mensuel & Progression")
            with st.expander("Voir le tableau", expanded=True):
                df_evo = get_monthly_evolution_table(df_view)
                st.dataframe(
                    df_evo.style.background_gradient(cmap="Purples", subset=['CA (GMV)'])
                    .format("{:+.2f}%", subset=['Prog/Reg CA %', 'Prog/Reg Cmd %', 'Prog/Reg AOV %', 'Prog/Reg Annul %', 'Prog/Reg Temps %'], na_rep="-"),
                    use_container_width=True
                )
            
            st.divider()

            # Graphs & Top 10
            c_graph1, c_graph2 = st.columns(2)
            growth_data = df_view.groupby(df_view['order day'].dt.to_period('M').astype(str)).agg({'GMV':'sum', 'order id':'count'}).reset_index()
            c_graph1.plotly_chart(px.bar(growth_data, x='order day', y='GMV', title="Evolution GMV", color_discrete_sequence=['#6c35de']), use_container_width=True)
            c_graph2.plotly_chart(px.line(growth_data, x='order day', y='order id', title="Evolution Commandes", markers=True, color_discrete_sequence=['#ff00ff']), use_container_width=True)
            
            # Top 10
            st.subheader("🏆 Classement Restaurants")
            df_rank = df_view.groupby(['Restaurant_Final', 'City_Final']).agg({
                'GMV': 'sum', 'order id': 'count', 'is_cancelled': 'mean', 'Delivery Time': 'mean'
            }).reset_index()
            df_rank['Taux Annul %'] = (df_rank['is_cancelled']*100).round(2)
            
            sort_col = st.selectbox("Trier par :", ["GMV", "Commandes", "Taux Annul %", "Temps Livraison"])
            tech_col = {'GMV':'GMV', 'Commandes':'order id', 'Taux Annul %':'Taux Annul %', 'Temps Livraison':'Delivery Time'}[sort_col]
            asc = True if tech_col in ['Taux Annul %', 'Delivery Time'] else False
            
            c_top, c_flop = st.columns(2)
            c_top.dataframe(df_rank.sort_values(tech_col, ascending=asc).head(10).style.background_gradient(cmap="Purples", subset=[tech_col]), use_container_width=True)
            c_flop.dataframe(df_rank.sort_values(tech_col, ascending=not asc).head(10).style.background_gradient(cmap="Reds", subset=[tech_col]), use_container_width=True)
            
            st.divider()
            
            # --- SECTION BAS DE PAGE : KPIS FINANCIERS ---
            display_advanced_kpis(df_view, title_prefix="(Vue AM)")


        # --- VUE GLOBALE ---
        with tab_global:
            st.header("🌍 Performance Globale Yassir")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("GMV Total", f"{df_filtered['GMV'].sum():,.0f} DH")
            k2.metric("Commandes", f"{len(df_filtered):,.0f}")
            k3.metric("Taux Annul.", f"{(df_filtered['is_cancelled'].mean()*100):.2f}%")
            k4.metric("Perte Sèche (Vol.)", f"{df_filtered['Missed GMV'].sum():,.0f} DH")

            st.subheader("📅 Evolution Mensuelle Globale")
            df_evo_global = get_monthly_evolution_table(df_filtered)
            st.dataframe(
                df_evo_global.style.background_gradient(cmap="Purples", subset=['CA (GMV)'])
                .format("{:+.2f}%", subset=['Prog/Reg CA %', 'Prog/Reg Cmd %', 'Prog/Reg AOV %', 'Prog/Reg Annul %', 'Prog/Reg Temps %'], na_rep="-"),
                use_container_width=True
            )
            
            st.subheader("📊 Performance par AM")
            df_am = df_filtered.groupby('AM_OWNER').agg({'GMV':'sum', 'order id':'count', 'is_cancelled':'mean', 'Missed GMV':'sum'}).reset_index()
            fig = px.scatter(df_am, x='order id', y='GMV', size='Missed GMV', color='AM_OWNER', title="Matrice AM (Taille bulle = Perte Sèche)")
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()

            # --- SECTION BAS DE PAGE : KPIS FINANCIERS ---
            display_advanced_kpis(df_filtered, title_prefix="(Vue Globale)")

    else:
        st.warning("Veuillez uploader les fichiers.")
else:
    st.info("En attente des fichiers...")

st.markdown('<div class="footer">Bounoir Saif eddine - Yassir Analytics Dashboard</div>', unsafe_allow_html=True)
