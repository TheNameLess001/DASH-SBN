import streamlit as st
import pandas as pd
import plotly.express as px
import os
import unicodedata
from datetime import timedelta

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="DASH-SBN | Analytics", page_icon="📊", layout="wide")

# ---------------------------------------------------------
# UTILITAIRES & CHARGEMENT
# ---------------------------------------------------------
def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return ascii_text.lower().strip()

@st.cache_data
def load_pipelines():
    pipelines_norm = {}
    am_files = {'NAJWA': 'NAJWA.csv', 'HOUDA': 'HOUDA.csv', 'CHAIMA': 'CHAIMA.csv'}
    for am, filename in am_files.items():
        if os.path.exists(filename):
            try:
                try:
                    df = pd.read_csv(filename, sep=',')
                    if len(df.columns)<2: raise ValueError
                except:
                    df = pd.read_csv(filename, sep=';')
                
                df.columns = df.columns.str.strip().str.lower()
                col = next((c for c in df.columns if 'restaurant' in c or 'name' in c), df.columns[0])
                pipelines_norm[am] = [normalize_text(x) for x in df[col].dropna().astype(str).tolist()]
            except: pipelines_norm[am] = []
        else: pipelines_norm[am] = []
    return pipelines_norm

@st.cache_data
def load_data(main_file, pipelines_norm):
    if hasattr(main_file, 'seek'): main_file.seek(0)
    try:
        df = pd.read_csv(main_file, sep=',')
        if 'order day' not in df.columns and len(df.columns) < 5: raise ValueError
    except:
        if hasattr(main_file, 'seek'): main_file.seek(0)
        df = pd.read_csv(main_file, sep=';')

    df.columns = df.columns.str.strip()
    
    # Dates
    df['order day'] = df['order day'].astype(str)
    df['order time'] = df['order time'].astype(str)
    
    def parse_dt(d_str):
        for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d'):
            try: return pd.to_datetime(d_str, format=fmt)
            except: continue
        return pd.to_datetime(d_str, errors='coerce')

    df['order_date_obj'] = df['order day'].apply(parse_dt)
    df['order_datetime'] = pd.to_datetime(
        df['order_date_obj'].dt.strftime('%Y-%m-%d') + ' ' + df['order time'], errors='coerce'
    )
    df['date'] = df['order_datetime'].dt.date
    df['year_month'] = df['order_datetime'].dt.to_period('M')

    # Numérique
    for c in ['item total', 'delivery amount', 'delivery time(M)', 'Time Taken']:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Attribution AM
    df['restaurant_norm'] = df['restaurant name'].apply(normalize_text)
    
    def get_am(row):
        r_norm = row['restaurant_norm']
        city = str(row.get('city', '')).lower()
        r_raw = str(row.get('restaurant name', '')).lower()
        
        for am, p_list in pipelines_norm.items():
            if r_norm in p_list: return am
            for p in p_list:
                if len(p)>3 and p in r_norm: return am
        
        if any(x in r_raw for x in ['mcdonald', 'kfc', 'burger king', 'primos', 'papa john']): return 'NAJWA'
        if any(c in city for c in ['rabat', 'sale', 'temara', 'kenitra']): return 'HOUDA'
        return 'CHAIMA'

    df['AM'] = df.apply(get_am, axis=1)

    # Automatisation
    if 'Assigned By' in df.columns:
        df['is_automated'] = df['Assigned By'].astype(str).str.contains('Algorithm|super_app', case=False, regex=True)
    else: df['is_automated'] = False
    
    # Enseigne
    def get_brand(name):
        n = normalize_text(name)
        if 'mcdonald' in n: return "McDonald's"
        if 'kfc' in n: return "KFC"
        if 'burger king' in n: return "Burger King"
        if 'primos' in n: return "Primos"
        return "Autres"
    df['Enseigne_Groupe'] = df['restaurant name'].apply(get_brand)

    return df

# ---------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------
st.title("🚀 DASH-SBN | Performance Analytics")
pipelines_norm = load_pipelines()

with st.sidebar:
    st.header("📂 Données")
    uploaded_file = st.file_uploader("Fichier CSV", type=['csv'])
    
    if not uploaded_file:
        # Fallback local (Optionnel)
        local = "admin-earnings-orders-export_v1.3.1_countryCode=MA&filters=_s_1761955200000_e_1769212799999exp.csv"
        # if os.path.exists(local): uploaded_file = local
        pass

    if uploaded_file:
        df = load_data(uploaded_file, pipelines_norm)
    else:
        st.info("Chargez le fichier CSV.")
        st.stop()
        
    st.divider()
    st.header("🎯 Scope & Filtres")
    
    # 1. SCOPE
    scope_options = ['Global', 'NAJWA', 'HOUDA', 'CHAIMA']
    selected_scope = st.selectbox("Vue (Scope)", scope_options)
    
    # 2. ENSEIGNE
    if selected_scope != 'Global':
        df_scope_preview = df[df['AM'] == selected_scope]
        available_brands = ['Tous'] + sorted(df_scope_preview['Enseigne_Groupe'].unique().tolist())
    else:
        available_brands = ['Tous'] + sorted(df['Enseigne_Groupe'].unique().tolist())
        
    sel_brand = st.selectbox("Enseigne / Groupe", available_brands)
    
    # 3. DATE
    min_d, max_d = df['date'].min(), df['date'].max()
    st.caption(f"Période dispo : {min_d} au {max_d}")

# ---------------------------------------------------------
# PRÉPARATION
# ---------------------------------------------------------
if selected_scope == 'Global':
    df_scope = df.copy()
    current_pipeline = []
    for p_list in pipelines_norm.values(): current_pipeline.extend(p_list)
    current_pipeline = list(set(current_pipeline))
else:
    df_scope = df[df['AM'] == selected_scope]
    current_pipeline = pipelines_norm.get(selected_scope, [])

if sel_brand != 'Tous':
    df_scope = df_scope[df_scope['Enseigne_Groupe'] == sel_brand]

if df_scope.empty:
    st.warning("Aucune donnée.")
    st.stop()

# ---------------------------------------------------------
# 1. KPI MENSUEL (MAIN TABLE)
# ---------------------------------------------------------
st.subheader(f"📊 Performance Mensuelle : {selected_scope}")

monthly_stats = df_scope.groupby('year_month').agg({
    'item total': 'sum',
    'order id': 'count',
    'is_automated': 'sum',
    'status': lambda x: x.astype(str).str.contains('Reject|Cancel', case=False).sum()
}).reset_index().sort_values('year_month')

monthly_stats['Mois'] = monthly_stats['year_month'].dt.strftime('%Y-%m')
monthly_stats['AOV'] = monthly_stats['item total'] / monthly_stats['order id']
monthly_stats['Auto %'] = (monthly_stats['is_automated'] / monthly_stats['order id'] * 100)
monthly_stats['Rejet/Cancel %'] = (monthly_stats['status'] / monthly_stats['order id'] * 100)
monthly_stats['CA Prev'] = monthly_stats['item total'].shift(1)
monthly_stats['Growth %'] = ((monthly_stats['item total'] - monthly_stats['CA Prev']) / monthly_stats['CA Prev'] * 100).fillna(0)

# Inactifs Mensuels
inactifs_list = []
for ym in monthly_stats['year_month']:
    actifs = df_scope[df_scope['year_month'] == ym]['restaurant_norm'].unique().tolist()
    matched = 0
    if len(current_pipeline) > 0:
        for p in current_pipeline:
            if any(p in a for a in actifs): matched += 1
        inactifs_list.append(max(0, len(current_pipeline) - matched))
    else:
        inactifs_list.append(0)
monthly_stats['Inactifs'] = inactifs_list

final_table = monthly_stats[['Mois', 'item total', 'order id', 'AOV', 'Growth %', 'Auto %', 'Rejet/Cancel %', 'Inactifs']].rename(
    columns={'item total': 'CA (MAD)', 'order id': 'Commandes', 'AOV': 'Panier Moy.'}
)

st.dataframe(
    final_table.style.format({
        "CA (MAD)": "{:,.0f}",
        "Commandes": "{:.0f}",
        "Panier Moy.": "{:.1f}",
        "Growth %": "{:+.1f}%",
        "Auto %": "{:.1f}%",
        "Rejet/Cancel %": "{:.1f}%",
        "Inactifs": "{:.0f}"
    }).background_gradient(subset=['Growth %'], cmap="RdYlGn", vmin=-20, vmax=20),
    use_container_width=True, hide_index=True
)

st.divider()

# ---------------------------------------------------------
# 2. TOP & FLOP (SCROLLABLE)
# ---------------------------------------------------------
st.subheader("📈 Tops & 📉 Flops (Scrollable)")

all_months = sorted(df['year_month'].unique())
if len(all_months) >= 2:
    last_month = all_months[-1]
    prev_month = all_months[-2]
    st.info(f"Comparaison : **{last_month}** vs **{prev_month}**")
    
    stats_m = df_scope[df_scope['year_month'] == last_month].groupby('restaurant name')['order id'].count()
    stats_m_1 = df_scope[df_scope['year_month'] == prev_month].groupby('restaurant name')['order id'].count()
    
    comp_df = pd.DataFrame({'Mois Préc': stats_m_1, 'Mois Actuel': stats_m}).fillna(0)
    comp_df['Delta'] = comp_df['Mois Actuel'] - comp_df['Mois Préc']
    
    col_flop, col_top = st.columns(2)
    
    with col_flop:
        st.markdown("### 🚨 Top Régressions")
        flops = comp_df[comp_df['Delta'] < 0].sort_values('Delta', ascending=True)
        if not flops.empty:
            st.dataframe(
                flops[['Mois Préc', 'Mois Actuel', 'Delta']].style.format("{:.0f}").background_gradient(subset=['Delta'], cmap='Reds_r'),
                use_container_width=True, 
                height=300 # SCROLLABLE FIXED HEIGHT
            )
        else: st.success("Rien à signaler.")

    with col_top:
        st.markdown("### 🚀 Top Progressions")
        tops = comp_df[comp_df['Delta'] > 0].sort_values('Delta', ascending=False)
        if not tops.empty:
            st.dataframe(
                tops[['Mois Préc', 'Mois Actuel', 'Delta']].style.format("{:.0f}").background_gradient(subset=['Delta'], cmap='Greens'),
                use_container_width=True, 
                height=300 # SCROLLABLE FIXED HEIGHT
            )
        else: st.info("Rien à signaler.")
else:
    st.warning("Pas assez d'historique pour Top/Flop.")

st.divider()

# ---------------------------------------------------------
# 3. ANALYSE CROISÉE (CANCELLATION x DELIVERY TIME)
# ---------------------------------------------------------
st.subheader("🚫 Analyse des Annulations & Temps de Livraison")

# On prépare les données par restaurant (sur la période sélectionnée globale ou tout l'historique dispo)
# On calcule : Taux d'annulation, Temps moyen de livraison (pour les commandes livrées)
resto_stats = df_scope.groupby('restaurant name').agg({
    'order id': 'count',
    'status': lambda x: x.astype(str).str.contains('Cancel|Reject', case=False).sum(),
    'delivery time(M)': 'mean', # Temps de livraison moyen (pour celles livrées)
    'AM': 'first' # Pour la couleur du graphe
}).reset_index()

resto_stats.columns = ['Restaurant', 'Total Orders', 'Cancelled', 'Avg Delivery Time (min)', 'AM']
resto_stats['Cancellation Rate (%)'] = (resto_stats['Cancelled'] / resto_stats['Total Orders'] * 100).round(1)

# Filtre pour éviter le bruit (ex: on garde seulement ceux > 5 commandes)
resto_stats_clean = resto_stats[resto_stats['Total Orders'] >= 5]

c1, c2 = st.columns([1, 2])

with c1:
    st.markdown("### ⚠️ Top Taux d'Annulation")
    # Table scrollable des pires taux
    top_cancel = resto_stats_clean.sort_values('Cancellation Rate (%)', ascending=False).head(50)
    st.dataframe(
        top_cancel[['Restaurant', 'Total Orders', 'Cancellation Rate (%)']].style.format({
            'Total Orders': "{:.0f}", 
            'Cancellation Rate (%)': "{:.1f}%"
        }).background_gradient(subset=['Cancellation Rate (%)'], cmap='Reds'),
        use_container_width=True,
        height=400
    )

with c2:
    st.markdown("### 📉 Corrélation : Temps de Livraison vs Annulation")
    st.caption("Chaque bulle est un restaurant. Plus la bulle est haute, plus le taux d'annulation est élevé. Plus elle est à droite, plus le resto est lent.")
    
    if not resto_stats_clean.empty:
        fig = px.scatter(
            resto_stats_clean,
            x='Avg Delivery Time (min)',
            y='Cancellation Rate (%)',
            size='Total Orders', # Taille de la bulle = Volume
            color='AM', # Couleur par AM (si vue Global) ou unique
            hover_name='Restaurant',
            title="Impact de la Lenteur sur les Annulations",
            template="plotly_white"
        )
        # Ligne moyenne
        avg_cancel = resto_stats_clean['Cancellation Rate (%)'].mean()
        fig.add_hline(y=avg_cancel, line_dash="dash", line_color="red", annotation_text="Moyenne Annulation")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Pas assez de données pour le graphique.")

# ---------------------------------------------------------
# 4. DATA RAW
# ---------------------------------------------------------
with st.expander("Voir les données brutes"):
    st.dataframe(df_scope)
