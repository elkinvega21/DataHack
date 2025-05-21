import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime, timedelta
import json
import random
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import traceback

st.set_page_config(layout="wide", page_title="Dashboard ERP Gerencial")

CURRENT_DATE_FOR_ANALYSIS = pd.Timestamp('2025-05-20')
START_YEAR = 2021
END_YEAR = 2025

@st.cache_resource
def get_db_engine_cached():
    usuario = "root"
    contraseña = "3112708652El"
    host = "localhost"
    puerto = 3306
    basededatos = "demo_wft"
    try:
        contraseña_escapada = quote_plus(contraseña)
        engine = create_engine(f"mysql+pymysql://{usuario}:{contraseña_escapada}@{host}:{puerto}/{basededatos}")
        with engine.connect() as connection: pass
        return engine
    except Exception as e:
        return None

engine = get_db_engine_cached()

if not engine:
    st.sidebar.error(f"Error de conexión a BD. Se usarán datos simulados si la carga falla.")

@st.cache_data(ttl=3600)
def load_financial_summary_data(_engine_ref, force_simulate_flag=False):
    all_months_list = []
    for yl in range(START_YEAR, END_YEAR + 1):
        lm = 12
        if yl == CURRENT_DATE_FOR_ANALYSIS.year and yl == END_YEAR: lm = CURRENT_DATE_FOR_ANALYSIS.month
        for ml in range(1, lm + 1):
            if yl == CURRENT_DATE_FOR_ANALYSIS.year and ml > CURRENT_DATE_FOR_ANALYSIS.month and yl == END_YEAR: continue
            all_months_list.append(pd.Timestamp(f'{yl}-{ml:02d}-01'))
    df_meses_base = pd.DataFrame({'mes': all_months_list}); df_meses_base['mes'] = pd.to_datetime(df_meses_base['mes'])
    ing_sim = [80_000_000 * (1.1**(fm.year - START_YEAR)) * np.random.uniform(0.8,1.2) for fm in df_meses_base['mes']]
    gg_sim = [ing * np.random.uniform(0.3,0.5) for ing in ing_sim]
    nom_sim = [25_000_000 * (1.08**(fm.year - START_YEAR)) * np.random.uniform(0.9,1.1) for fm in df_meses_base['mes']]
    df_totales = pd.DataFrame({'mes': df_meses_base['mes'], 'ingresos': ing_sim, 'gastos_generales': gg_sim, 'gastos_nomina': nom_sim})
    df_totales['gastos_totales'] = df_totales['gastos_generales'] + df_totales['gastos_nomina']
    df_totales['utilidad_neta'] = df_totales['ingresos'] - df_totales['gastos_totales']
    return df_totales

@st.cache_data(ttl=3600)
def load_cxc_data(_engine_ref, force_simulate_flag=False):
    col_contact_name = "full_name"
    df_cxc_contacts = pd.DataFrame([{'contact_id_pk_contacts': i, col_contact_name: f'Cliente Sim {i}'} for i in range(1, 6)])
    contact_ids_for_sim = df_cxc_contacts['contact_id_pk_contacts'].unique()
    sim_receivables_list = []; sim_motives = ["Servicio Consultoría", "Venta Productos Alfa", "Mantenimiento Anual"]
    for i in range(30):
        days_old = random.choice([15, 25, 35, 50, 75, 85, 100, 150])
        sim_receivables_list.append({'contact_id': random.choice(contact_ids_for_sim),'outstanding_balance': random.uniform(100000, 2000000),'effective_date_for_aging': CURRENT_DATE_FOR_ANALYSIS - timedelta(days=days_old),'motive': random.choice(sim_motives)})
    df_all_receivables = pd.DataFrame(sim_receivables_list)
    if not df_all_receivables.empty:
        df_all_receivables['age_days'] = (CURRENT_DATE_FOR_ANALYSIS - df_all_receivables['effective_date_for_aging']).dt.days
        bins = [-float('inf'), 30, 60, 90, float('inf')]; aging_labels = ['0-30 días', '31-60 días', '61-90 días', '91+ días']
        df_all_receivables['aging_bucket'] = pd.cut(df_all_receivables['age_days'], bins=bins, labels=aging_labels, right=True)
        df_all_receivables['motive'] = df_all_receivables['motive'].fillna("No Especificado")
    return df_all_receivables, df_cxc_contacts

@st.cache_data(ttl=3600)
def load_inventory_valuation_data(_engine_ref, force_simulate_flag=False):
    df_merged_inv = pd.DataFrame()
    sim_inv_data = []; categories_sim = ['Electrónicos (Sim)', 'Ropa (Sim)', 'Hogar (Sim)']
    for i in range(1, 21):
        sim_inv_data.append({'item_id': i, 'name': f'Producto Sim {i}', 'cost_proxy': random.uniform(5000, 150000), 'category_name': random.choice(categories_sim),'qty_on_hand': random.randint(0,100)})
    df_merged_inv = pd.DataFrame(sim_inv_data)
    if not df_merged_inv.empty : df_merged_inv['inventory_value'] = df_merged_inv['qty_on_hand'] * df_merged_inv['cost_proxy']
    return df_merged_inv

MODELO_PATH_PRED = 'modelo_ventas_pred.pkl'
SCALER_X_PATH_PRED = 'escalador_X_pred.pkl'
SCALER_Y_PATH_PRED = 'escalador_y_pred.pkl'
ENCODER_PATH_PRED = 'label_encoder_clientes_pred.pkl'

def formatear_cop_pred(valor):
    s = f"{valor:,.0f}"
    s = s.replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"COP ${s}"

@st.cache_resource
def cargar_o_entrenar_modelo_ventas_nuevo(_engine_ref):
    if all(os.path.exists(p) for p in [MODELO_PATH_PRED, SCALER_X_PATH_PRED, SCALER_Y_PATH_PRED, ENCODER_PATH_PRED]):
        modelo = joblib.load(MODELO_PATH_PRED); scaler_X = joblib.load(SCALER_X_PATH_PRED)
        scaler_y = joblib.load(SCALER_Y_PATH_PRED); label_encoder = joblib.load(ENCODER_PATH_PRED)
        return modelo, scaler_X, scaler_y, label_encoder, True

    st.warning("Modelo no encontrado para predicción de ventas. Entrenando nuevo modelo... Esto puede tardar.")
    if not _engine_ref:
        st.error("No hay conexión a la base de datos para entrenar el modelo de predicción.")
        return None, None, None, None, False
    try:
        df = pd.read_sql("SELECT third_party_id, year, month, credit_movement FROM accounting_account_balances", _engine_ref)
        df = df.dropna(subset=['year', 'month', 'third_party_id'])
        df['year'] = df['year'].astype(str).str.extract(r'(\d{4})', expand=False).astype('Int64')
        df['month'] = df['month'].astype('Int64')
        df = df[(df['year'] > 1900) & (df['month'] >= 1) & (df['month'] <= 12) & (df['year'] <= END_YEAR)]
        df['third_party_id'] = df['third_party_id'].astype(str)
        df['credit_movement'] = pd.to_numeric(df['credit_movement'], errors='coerce').fillna(0)
        df_grouped = df.groupby(['third_party_id', 'year', 'month'], as_index=False)['credit_movement'].sum()
        df_grouped.rename(columns={'credit_movement': 'ventas'}, inplace=True)

        if df_grouped.empty:
            st.error("No hay datos agrupados para 'ventas' para el modelo.")
            return None, None, None, None, False

        if len(df_grouped) < 50:
            st.warning("Datos insuficientes para entrenar el modelo de predicción (menos de 50 registros).")
            return None, None, None, None, False

        label_encoder = LabelEncoder(); df_grouped['cliente_id_encoded'] = label_encoder.fit_transform(df_grouped['third_party_id'])
        X = df_grouped[['cliente_id_encoded', 'year', 'month']]; y = df_grouped['ventas']
        if len(X) < 20:
            st.warning("Datos insuficientes después de la preparación para el set de entrenamiento (menos de 20).")
            return None, None, None, None, False
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler_X = StandardScaler(); X_train_scaled = scaler_X.fit_transform(X_train)
        scaler_y = StandardScaler(); y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        modelo = MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam', alpha=0.001, max_iter=2000, random_state=42, early_stopping=True, n_iter_no_change=30, tol=1e-4, learning_rate_init=0.001)
        with st.spinner("Entrenando modelo MLP Regressor..."):
            modelo.fit(X_train_scaled, y_train_scaled)
        joblib.dump(modelo, MODELO_PATH_PRED); joblib.dump(scaler_X, SCALER_X_PATH_PRED); joblib.dump(scaler_y, SCALER_Y_PATH_PRED); joblib.dump(label_encoder, ENCODER_PATH_PRED)
        return modelo, scaler_X, scaler_y, label_encoder, True
    except Exception as e_train:
        st.error(f"Error durante el entrenamiento del modelo de predicción: {e_train}")
        st.error(traceback.format_exc())
        return None, None, None, None, False

def predecir_ventas_futuras(clientes_ids, anios, meses, le, scaler_X, scaler_y, modelo):
    resultados = []
    for cliente_original_id_str in clientes_ids:
        try:
            cliente_encoded = le.transform([str(cliente_original_id_str)])[0]
        except ValueError:
            st.warning(f"⚠️ Cliente ID '{cliente_original_id_str}' no encontrado en los datos de entrenamiento. Se omitirá.")
            continue
        for anio_p in anios:
            for mes_p in meses:
                X_nuevo = pd.DataFrame([[cliente_encoded, anio_p, mes_p]], columns=['cliente_id_encoded', 'year', 'month'])
                X_nuevo_scaled = scaler_X.transform(X_nuevo)
                y_nuevo_scaled = modelo.predict(X_nuevo_scaled)
                y_nuevo_bruto = scaler_y.inverse_transform(y_nuevo_scaled.reshape(-1, 1)).ravel()[0]
                y_nuevo_final = max(0, y_nuevo_bruto)
                resultados.append({'cliente_id': cliente_original_id_str, 'year': anio_p, 'month': mes_p, 'ventas_predichas': y_nuevo_final})
    return pd.DataFrame(resultados)

st.title("📊 Dashboard ERP Gerencial")
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox("Seleccione un Módulo:", ["Resumen Financiero", "Cuentas por Cobrar", "Valoración de Inventario", "Predicción de Ventas"])
force_simulation_all = st.sidebar.checkbox("Forzar simulación de datos (ignora BD)", False)
if force_simulation_all and app_mode != "Predicción de Ventas": effective_engine = None
else: effective_engine = engine

if app_mode == "Resumen Financiero":
    st.header("Resumen Financiero Mensual y Anual")
    df_financials = load_financial_summary_data(effective_engine, force_simulate_flag=force_simulation_all)
    if not df_financials.empty:
        total_ingresos = df_financials['ingresos'].sum(); total_gastos = df_financials['gastos_totales'].sum(); total_utilidad = df_financials['utilidad_neta'].sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Ingresos Totales (Periodo)", f"${total_ingresos:,.0f}"); col2.metric("Gastos Totales (Periodo)", f"${total_gastos:,.0f}"); col3.metric("Utilidad Neta Total (Periodo)", f"${total_utilidad:,.0f}")
        st.subheader("Ingresos, Gastos y Utilidad Neta Mensuales")
        fig_fin, ax1_fin = plt.subplots(figsize=(10, 5))
        p1, = ax1_fin.plot(df_financials['mes'], df_financials['ingresos'], color='tab:green', label='Ingresos', marker='o')
        p2, = ax1_fin.plot(df_financials['mes'], df_financials['gastos_totales'], color='tab:red', label='Gastos Totales', marker='x', linestyle='--')
        p3, = ax1_fin.plot(df_financials['mes'], df_financials['utilidad_neta'], color='tab:blue', label='Utilidad Neta', marker='.', linestyle=':')
        ax1_fin.set_xlabel('Mes'); ax1_fin.set_ylabel('Monto ($)'); ax1_fin.legend(handles=[p1,p2,p3], loc='best');
        ax1_fin.grid(True, linestyle='--', alpha=0.7); formatter = mtick.FormatStrFormatter('$%1.0f'); ax1_fin.yaxis.set_major_formatter(formatter)
        plt.xticks(rotation=45); fig_fin.tight_layout(); st.pyplot(fig_fin)
        st.subheader("Presupuesto Anual de Ingresos: Esperado vs. Real")
        df_ing_real_anual = df_financials.groupby(df_financials['mes'].dt.year)['ingresos'].sum().reset_index(); df_ing_real_anual.rename(columns={'mes': 'año', 'ingresos': 'ingresos_reales'}, inplace=True)
        pres_anuales = []; primer_año_b = START_YEAR
        ing_primer_año_v = df_financials[df_financials['mes'].dt.year == primer_año_b]['ingresos'].sum() if not df_financials[df_financials['mes'].dt.year == primer_año_b].empty else 0
        base_an = ing_primer_año_v if ing_primer_año_v > 0 else (80000000*12 * (1.10**(START_YEAR-START_YEAR)))
        for y_iter in range(START_YEAR, END_YEAR + 1):
            pres_año = base_an * (1.10 ** (y_iter - primer_año_b)); pres_anuales.append({'año': y_iter, 'presupuesto_ingresos_esperado': pres_año})
        df_pres_anual = pd.DataFrame(pres_anuales)
        df_comp_pres = pd.merge(df_pres_anual, df_ing_real_anual, on='año', how='left').fillna(0); df_comp_pres['diferencia'] = df_comp_pres['ingresos_reales'] - df_comp_pres['presupuesto_ingresos_esperado']
        df_disp_pres = df_comp_pres.copy();
        for c in ['presupuesto_ingresos_esperado', 'ingresos_reales', 'diferencia']: df_disp_pres[c] = df_disp_pres[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_disp_pres.set_index('año'))
    else: st.warning("No hay datos financieros para mostrar.")

elif app_mode == "Cuentas por Cobrar":
    st.header("Análisis de Cuentas por Cobrar (CxC)")
    df_cxc, df_cxc_contacts = load_cxc_data(effective_engine, force_simulate_flag=force_simulation_all)
    if not df_cxc.empty:
        st.subheader("Antigüedad de Saldos por Cobrar por Motivo")
        if 'motive' in df_cxc.columns and 'aging_bucket' in df_cxc.columns:
            df_aging_by_motive = df_cxc.groupby(['aging_bucket', 'motive'], observed=False)['outstanding_balance'].sum().unstack(fill_value=0)
            aging_lbls_cxc = ['0-30 días', '31-60 días', '61-90 días', '91+ días']; df_aging_by_motive = df_aging_by_motive.reindex(pd.Categorical(aging_lbls_cxc, categories=aging_lbls_cxc, ordered=True)).fillna(0)
            if not df_aging_by_motive.empty:
                fig_cxc_aging, ax_cxc_aging = plt.subplots(figsize=(10, 6)); num_motives = len(df_aging_by_motive.columns); cmap = plt.cm.get_cmap('tab20' if num_motives > 10 else 'Set3', num_motives if num_motives > 0 else 1)
                df_aging_by_motive.plot(kind='bar', stacked=True, ax=ax_cxc_aging, colormap=cmap)
                ax_cxc_aging.set_title('CxC por Antigüedad y Motivo'); ax_cxc_aging.set_xlabel('Rango Antigüedad'); ax_cxc_aging.set_ylabel('Saldo Pendiente ($)')
                ax_cxc_aging.tick_params(axis='x', rotation=0, labelsize=10); formatter = mtick.FormatStrFormatter('$%1.0f'); ax_cxc_aging.yaxis.set_major_formatter(formatter)
                ax_cxc_aging.legend(title='Motivo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small'); ax_cxc_aging.grid(axis='y', linestyle='--', alpha=0.7); fig_cxc_aging.tight_layout(rect=[0, 0, 0.82, 1]); st.pyplot(fig_cxc_aging)
        st.subheader("Principales Deudores")
        col_contact_name_cxc_disp = "full_name"; col_contact_id_cxc_disp_key = "contact_id_pk_contacts"
        df_top_d = df_cxc.groupby('contact_id')['outstanding_balance'].sum().reset_index()
        if not df_cxc_contacts.empty and col_contact_id_cxc_disp_key in df_cxc_contacts.columns:
            df_top_d = pd.merge(df_top_d, df_cxc_contacts.rename(columns={col_contact_id_cxc_disp_key: 'contact_id'}), on='contact_id', how='left')
        else:
            df_top_d[col_contact_name_cxc_disp] = df_top_d['contact_id'].astype(str) + " (Nombre no disponible)"

        if col_contact_name_cxc_disp not in df_top_d.columns:
            df_top_d[col_contact_name_cxc_disp] = df_top_d['contact_id'].astype(str) + " (Nombre no disponible)"

        df_top_d[col_contact_name_cxc_disp] = df_top_d[col_contact_name_cxc_disp].fillna(df_top_d['contact_id'].astype(str) + " (ID no enc.)")
        df_top_d_disp = df_top_d.sort_values(by='outstanding_balance', ascending=False).rename(columns={col_contact_name_cxc_disp: 'Cliente', 'outstanding_balance': 'Saldo Pendiente'})
        df_disp_d_table = df_top_d_disp[['Cliente', 'Saldo Pendiente']].head(10).copy(); df_disp_d_table['Saldo Pendiente'] = df_disp_d_table['Saldo Pendiente'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df_disp_d_table.set_index('Cliente'))
    else: st.warning("No hay datos de CxC para mostrar.")

elif app_mode == "Valoración de Inventario":
    st.header("Valoración de Inventario")
    df_inventory = load_inventory_valuation_data(effective_engine, force_simulate_flag=force_simulation_all)
    if not df_inventory.empty and 'inventory_value' in df_inventory.columns:
        total_inv_value = df_inventory['inventory_value'].sum()
        st.metric("Valor Total del Inventario (Proxy)", f"${total_inv_value:,.0f}")
        st.subheader("Valor del Inventario por Categoría")
        df_val_by_cat = df_inventory.groupby('category_name')['inventory_value'].sum().reset_index(); df_val_by_cat = df_val_by_cat.sort_values(by='inventory_value', ascending=False)
        if not df_val_by_cat.empty:
            fig_inv_cat, ax_inv_cat = plt.subplots(figsize=(12, 7))
            bars_cat = ax_inv_cat.bar(df_val_by_cat['category_name'].head(10), df_val_by_cat['inventory_value'].head(10), color='skyblue')
            ax_inv_cat.set_title('Top 10 - Valor Inventario por Categoría'); ax_inv_cat.set_xlabel('Categoría'); ax_inv_cat.set_ylabel('Valor Inventario ($)')
            ax_inv_cat.tick_params(axis='x', labelsize=10)
            for label in ax_inv_cat.get_xticklabels(): label.set_rotation(45); label.set_ha('right'); label.set_rotation_mode('anchor')
            ax_inv_cat.tick_params(axis='y', labelsize=10)
            formatter = mtick.FormatStrFormatter('$%1.0f'); ax_inv_cat.yaxis.set_major_formatter(formatter); ax_inv_cat.grid(axis='y', linestyle='--', alpha=0.7)
            for bar in bars_cat: yval = bar.get_height(); ax_inv_cat.text(bar.get_x() + bar.get_width()/2.0, yval + ax_inv_cat.get_ylim()[1]*0.01, f'${yval:,.0f}', ha='center', va='bottom', fontsize=9)
            fig_inv_cat.tight_layout(); st.pyplot(fig_inv_cat)
    else: st.warning("No hay datos de Valoración de Inventario para mostrar.")

elif app_mode == "Predicción de Ventas":
    st.header("📈 Predicción de Ventas Futuras Por Clientes (COP)")
    modelo_c, scaler_X_c, scaler_y_c, le_cli_c, modelo_ok = cargar_o_entrenar_modelo_ventas_nuevo(engine)
    if modelo_ok:
        try:
            cli_orig_ids = [str(cls) for cls in le_cli_c.classes_]
        except AttributeError:
            st.error("No se pudo obtener lista de clientes del modelo de predicción."); cli_orig_ids = []
        if not cli_orig_ids: st.warning("No hay clientes del modelo para seleccionar.")
        else:
            cli_sel_ids = st.multiselect("Selecciona clientes (ID Original)", options=cli_orig_ids, default=cli_orig_ids[:min(3, len(cli_orig_ids))] if cli_orig_ids else None)
            curr_pred_year = datetime.now().year
            año_pred = st.slider("Selecciona año para predicción", curr_pred_year, curr_pred_year + 5, curr_pred_year + 1)
            meses_disp = list(range(1, 13))
            meses_sel = st.multiselect("Selecciona meses para predicción", options=meses_disp, default=[datetime.now().month] if datetime.now().month in meses_disp else [meses_disp[0]])
            if st.button("Predecir ventas"):
                if not cli_sel_ids or not meses_sel: st.error("Selecciona al menos un cliente y un mes.")
                else:
                    with st.spinner("Realizando predicciones..."):
                        df_preds = predecir_ventas_futuras(cli_sel_ids, [año_pred], meses_sel, le_cli_c, scaler_X_c, scaler_y_c, modelo_c)
                    if not df_preds.empty:
                        df_preds['Ventas Predichas (COP)'] = df_preds['ventas_predichas'].apply(formatear_cop_pred)
                        st.subheader("Resultados de la Predicción")
                        st.dataframe(df_preds[['cliente_id', 'year', 'month', 'Ventas Predichas (COP)']].rename(columns={'cliente_id': 'ID Cliente (Original)', 'year': 'Año', 'month': 'Mes'}).reset_index(drop=True))
    else: st.error("Modelo de predicción no disponible. Por favor, revisa los mensajes de error anteriores o la consola para más detalles.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dashboard Creado con Streamlit")