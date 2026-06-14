# =============================================================================
# app.py — Dashboard EV3: Google Play Store Apps
# Curso: SCY1101 | Alumno: R. Cuadrado
# Ejecutar: python app.py
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Dash & Plotly ─────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go

# ── Scikit-learn (para re-entrenar con el subconjunto filtrado) ───────────────
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# =============================================================================
# 1. CARGA Y PREPARACIÓN GLOBAL DEL DATASET
# =============================================================================

df = pd.read_csv("googleplaystore_limpio.csv")

# ── Columnas necesarias ───────────────────────────────────────────────────────
columnas_modelo = ["Category", "Rating", "Reviews", "Installs",
                   "Type", "Price", "Content Rating", "Genres"]

df_base = df[columnas_modelo].copy()

# ── Variable objetivo (umbral EV2: Rating >= 4.3) ────────────────────────────
df_base["Rating_Alto"] = df_base["Rating"].apply(lambda x: 1 if x >= 4.3 else 0)

# ── Aseguramos tipos numéricos básicos ───────────────────────────────────────
df_base["Reviews"] = pd.to_numeric(df_base["Reviews"], errors="coerce")
df_base["Installs"] = pd.to_numeric(
    df_base["Installs"].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False),
    errors="coerce"
)
df_base["Price"] = pd.to_numeric(
    df_base["Price"].astype(str)
    .str.replace("$", "", regex=False),
    errors="coerce"
)
df_base.dropna(subset=["Reviews", "Installs", "Price", "Rating"], inplace=True)
df_base.reset_index(drop=True, inplace=True)

# ── Métricas FIJAS reales del notebook (no se recalculan con filtros) ─────────
#    Regresión Logística: Accuracy 0.64, Precision 0.70, Recall 0.89, F1 0.78
#    Random Forest:       Accuracy 0.69, Precision 0.73, Recall 0.76, F1 0.74
METRICAS_FIJAS = {
    "Regresión Logística": {
        "Accuracy":  0.64,
        "Precision": 0.70,
        "Recall":    0.89,
        "F1-Score":  0.78,
    },
    "Random Forest": {
        "Accuracy":  0.69,
        "Precision": 0.73,
        "Recall":    0.76,
        "F1-Score":  0.74,
    },
    "RF Optimizado": {
        "Accuracy":  0.69,
        "Precision": 0.80,
        "Recall":    0.71,
        "F1-Score":  0.75,
    },
}

# ── Listas para los filtros globales ─────────────────────────────────────────
categorias = sorted(df_base["Category"].dropna().unique().tolist())
content_ratings = sorted(df_base["Content Rating"].dropna().unique().tolist())

# =============================================================================
# 2. FUNCIÓN AUXILIAR: pipeline de ML sobre un subconjunto
# =============================================================================

variables_numericas   = ["Reviews", "Installs", "Price"]
variables_categoricas = ["Category", "Type", "Content Rating", "Genres"]

def build_pipeline_and_metrics(df_sub: pd.DataFrame):
    """
    Reconstruye el pipeline EV2 sobre df_sub y devuelve:
      - métricas de LR y RF
      - matriz de confusión del RF
      - importancias de variables del RF
      - df con columna Cluster y coordenadas PCA
    """
    df_sub = df_sub.copy().dropna(
        subset=variables_numericas + variables_categoricas + ["Rating_Alto"]
    )

    if len(df_sub) < 50:
        return None  # subconjunto demasiado pequeño

    X = df_sub[variables_numericas + variables_categoricas]
    y = df_sub["Rating_Alto"]

    preprocesador = ColumnTransformer(transformers=[
        ("num", StandardScaler(), variables_numericas),
        ("cat", OneHotEncoder(handle_unknown="ignore"), variables_categoricas),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Regresión Logística
    pipe_lr = Pipeline([
        ("prepro", preprocesador),
        ("modelo", LogisticRegression(max_iter=1000))
    ])
    pipe_lr.fit(X_train, y_train)
    y_pred_lr = pipe_lr.predict(X_test)

    # Random Forest
    pipe_rf = Pipeline([
        ("prepro", preprocesador),
        ("modelo", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    y_pred_rf = pipe_rf.predict(X_test)

    metricas = {
        "Regresión Logística": {
            "Accuracy":  round(accuracy_score(y_test, y_pred_lr), 4),
            "Precision": round(precision_score(y_test, y_pred_lr, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred_lr, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred_lr, zero_division=0), 4),
        },
        "Random Forest": {
            "Accuracy":  round(accuracy_score(y_test, y_pred_rf), 4),
            "Precision": round(precision_score(y_test, y_pred_rf, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred_rf, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred_rf, zero_division=0), 4),
        },
    }

    cm = confusion_matrix(y_test, y_pred_rf)

    # Importancia de variables
    feature_names = pipe_rf.named_steps["prepro"].get_feature_names_out()
    importances   = pipe_rf.named_steps["modelo"].feature_importances_
    feat_series   = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(10)
    )

    # Clustering KMeans (K=3, como en EV2)
    X_proc = preprocesador.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_proc)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_proc.toarray() if hasattr(X_proc, "toarray") else X_proc)

    df_cluster = df_sub.copy().reset_index(drop=True)
    df_cluster["Cluster"]  = clusters.astype(str)
    df_cluster["PCA_1"]    = X_pca[:, 0]
    df_cluster["PCA_2"]    = X_pca[:, 1]

    return {
        "metricas":   metricas,
        "cm":         cm,
        "feat_imp":   feat_series,
        "df_cluster": df_cluster,
    }

# =============================================================================
# 3. ESTILOS
# =============================================================================

COLORES = {
    "primary":   "#1a73e8",
    "secondary": "#34a853",
    "accent":    "#fbbc04",
    "danger":    "#ea4335",
    "bg":        "#f8f9fa",
    "card":      "#ffffff",
    "text":      "#202124",
    "muted":     "#5f6368",
    "border":    "#e0e0e0",
}

STYLE_PAGE = {
    "fontFamily": "'Segoe UI', Arial, sans-serif",
    "backgroundColor": COLORES["bg"],
    "minHeight": "100vh",
    "color": COLORES["text"],
}

STYLE_HEADER = {
    "background": f"linear-gradient(135deg, {COLORES['primary']} 0%, #1557b0 100%)",
    "color": "white",
    "padding": "24px 32px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
}

STYLE_CARD = {
    "backgroundColor": COLORES["card"],
    "borderRadius": "12px",
    "padding": "20px",
    "boxShadow": "0 1px 6px rgba(0,0,0,0.08)",
    "border": f"1px solid {COLORES['border']}",
    "marginBottom": "16px",
}

STYLE_KPI = {
    **STYLE_CARD,
    "textAlign": "center",
    "flex": "1",
    "minWidth": "160px",
    "marginBottom": "0",
}

STYLE_FILTERS = {
    **STYLE_CARD,
    "display": "flex",
    "gap": "24px",
    "flexWrap": "wrap",
    "alignItems": "flex-end",
    "marginBottom": "20px",
}

STYLE_ROW = {
    "display": "flex",
    "gap": "16px",
    "flexWrap": "wrap",
    "marginBottom": "16px",
}

def kpi_card(title, value_id, color=COLORES["primary"], subtitle=""):
    return html.Div([
        html.P(title, style={
            "fontSize": "12px", "fontWeight": "600", "color": COLORES["muted"],
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "margin": "0 0 6px 0",
        }),
        html.H2(id=value_id, style={
            "fontSize": "28px", "fontWeight": "700",
            "color": color, "margin": "0",
        }),
        html.P(subtitle, style={
            "fontSize": "11px", "color": COLORES["muted"], "margin": "4px 0 0 0",
        }),
    ], style=STYLE_KPI)


def metric_block_static(label, value, color=COLORES["primary"]):
    return html.Div([
        html.P(label, style={
            "fontSize": "11px", "color": COLORES["muted"],
            "textTransform": "uppercase", "margin": "0 0 4px 0",
            "fontWeight": "600",
        }),
        html.H3(f"{value:.2%}", style={
            "fontSize": "24px", "fontWeight": "700",
            "color": color, "margin": "0",
        }),
    ], style={
        "textAlign": "center", "flex": "1", "minWidth": "100px",
        "padding": "12px", "borderRadius": "8px",
        "backgroundColor": f"{color}10",
        "border": f"1px solid {color}30",
    })

# =============================================================================
# 4. LAYOUT
# =============================================================================

app = dash.Dash(__name__, title="Google Play Store Dashboard")
server = app.server  # para despliegue con gunicorn

# ── Filtros globales ──────────────────────────────────────────────────────────
filtros_layout = html.Div([
    html.Div([
        html.Label("Categoría", style={"fontWeight": "600", "fontSize": "13px",
                                        "marginBottom": "6px", "display": "block"}),
        dcc.Dropdown(
            id="dd-category",
            options=[{"label": "Todas las categorías", "value": "ALL"}] +
                    [{"label": c, "value": c} for c in categorias],
            value="ALL",
            clearable=False,
            style={"minWidth": "240px"},
        ),
    ], style={"flex": "1", "minWidth": "200px"}),
    html.Div([
        html.Label("Clasificación de contenido", style={
            "fontWeight": "600", "fontSize": "13px",
            "marginBottom": "6px", "display": "block",
        }),
        dcc.Dropdown(
            id="dd-content-rating",
            options=[{"label": "Todas", "value": "ALL"}] +
                    [{"label": cr, "value": cr} for cr in content_ratings],
            value="ALL",
            clearable=False,
            style={"minWidth": "220px"},
        ),
    ], style={"flex": "1", "minWidth": "180px"}),
    html.Div(id="lbl-filtered", style={
        "fontSize": "13px", "color": COLORES["muted"],
        "alignSelf": "flex-end", "paddingBottom": "6px",
    }),
], style=STYLE_FILTERS)

# ── Tab Gerencial ─────────────────────────────────────────────────────────────
tab_gerencial = dcc.Tab(label="📊 Vista Gerencial", value="gerencial",
    style={"fontWeight": "600", "fontSize": "14px"},
    selected_style={
        "fontWeight": "700", "fontSize": "14px",
        "color": COLORES["primary"],
        "borderTop": f"3px solid {COLORES['primary']}",
    },
    children=[
        html.Div([
            # KPIs fila
            html.Div([
                kpi_card("Total de Apps", "kpi-total", COLORES["primary"],
                         "en la selección actual"),
                kpi_card("Calificación Promedio", "kpi-rating", COLORES["secondary"],
                         "Rating medio del subconjunto"),
                kpi_card("Apps con Rating Alto", "kpi-success", COLORES["accent"],
                         "Rating ≥ 4.3"),
                kpi_card("% Gratuitas", "kpi-free", COLORES["danger"],
                         "Apps de tipo Free"),
            ], style={**STYLE_ROW, "marginBottom": "20px"}),

            # Gráfico 1: Top 10 categorías + Distribución Free/Paid
            html.Div([
                html.Div([
                    html.H3("Top 10 Categorías por Instalaciones",
                            style={"fontSize": "15px", "fontWeight": "600",
                                   "marginBottom": "8px", "color": COLORES["text"]}),
                    dcc.RadioItems(
                        id="radio-top10",
                        options=[
                            {"label": "  Instalaciones", "value": "Installs"},
                            {"label": "  Reviews", "value": "Reviews"},
                        ],
                        value="Installs",
                        inline=True,
                        style={"fontSize": "13px", "marginBottom": "8px"},
                        labelStyle={"marginRight": "16px"},
                    ),
                    dcc.Graph(id="graph-top10", config={"displayModeBar": False}),
                ], style={**STYLE_CARD, "flex": "2", "minWidth": "320px",
                          "marginBottom": "0"}),

                html.Div([
                    html.H3("Distribución Free vs Paid",
                            style={"fontSize": "15px", "fontWeight": "600",
                                   "marginBottom": "8px", "color": COLORES["text"]}),
                    dcc.Graph(id="graph-freepaid", config={"displayModeBar": False}),
                ], style={**STYLE_CARD, "flex": "1", "minWidth": "240px",
                          "marginBottom": "0"}),
            ], style=STYLE_ROW),

            # Gráfico 2: Instalaciones por categoría (barras apiladas Free/Paid)
            html.Div([
                html.H3("Instalaciones promedio por Categoría (Free vs Paid)",
                        style={"fontSize": "15px", "fontWeight": "600",
                               "marginBottom": "8px", "color": COLORES["text"]}),
                dcc.Graph(id="graph-installs-cat", config={"displayModeBar": False}),
            ], style=STYLE_CARD),
        ], style={"padding": "24px"}),
    ]
)

# ── Tab Técnica ───────────────────────────────────────────────────────────────
tab_tecnica = dcc.Tab(label="🤖 Vista Técnica (Data Science)", value="tecnica",
    style={"fontWeight": "600", "fontSize": "14px"},
    selected_style={
        "fontWeight": "700", "fontSize": "14px",
        "color": COLORES["secondary"],
        "borderTop": f"3px solid {COLORES['secondary']}",
    },
    children=[
        html.Div([
            # Banner informativo
            html.Div([
                html.Span("ℹ️  ", style={"fontSize": "16px"}),
                html.Span(
                    "Los KPIs de métricas (fila superior) reflejan los valores reales obtenidos "
                    "en la EV2 sobre el dataset completo. Los gráficos inferiores (Matriz de Confusión, "
                    "Importancia de Variables y Clustering) se recalculan dinámicamente con el "
                    "subconjunto filtrado.",
                    style={"fontSize": "13px", "color": COLORES["muted"]},
                ),
            ], style={
                "backgroundColor": "#e8f0fe", "borderRadius": "8px",
                "padding": "12px 16px", "marginBottom": "20px",
                "border": f"1px solid {COLORES['primary']}30",
            }),

            # KPIs de modelos — FIJOS del notebook
            html.Div([
                # Logistic Regression
                html.Div([
                    html.H4("Regresión Logística",
                            style={"fontSize": "14px", "fontWeight": "700",
                                   "color": COLORES["primary"], "marginBottom": "12px",
                                   "textAlign": "center"}),
                    html.Div([
                        metric_block_static("Accuracy",  METRICAS_FIJAS["Regresión Logística"]["Accuracy"],  COLORES["primary"]),
                        metric_block_static("Precision", METRICAS_FIJAS["Regresión Logística"]["Precision"], COLORES["secondary"]),
                        metric_block_static("Recall",    METRICAS_FIJAS["Regresión Logística"]["Recall"],    COLORES["accent"]),
                        metric_block_static("F1-Score",  METRICAS_FIJAS["Regresión Logística"]["F1-Score"],  COLORES["danger"]),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={**STYLE_CARD, "flex": "1", "minWidth": "280px",
                          "marginBottom": "0"}),

                # Random Forest
                html.Div([
                    html.H4("Random Forest (Mejor Modelo)",
                            style={"fontSize": "14px", "fontWeight": "700",
                                   "color": COLORES["secondary"], "marginBottom": "12px",
                                   "textAlign": "center"}),
                    html.Div([
                        metric_block_static("Accuracy",  METRICAS_FIJAS["Random Forest"]["Accuracy"],  COLORES["primary"]),
                        metric_block_static("Precision", METRICAS_FIJAS["Random Forest"]["Precision"], COLORES["secondary"]),
                        metric_block_static("Recall",    METRICAS_FIJAS["Random Forest"]["Recall"],    COLORES["accent"]),
                        metric_block_static("F1-Score",  METRICAS_FIJAS["Random Forest"]["F1-Score"],  COLORES["danger"]),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={**STYLE_CARD, "flex": "1", "minWidth": "280px",
                          "marginBottom": "0"}),
            ], style={**STYLE_ROW, "marginBottom": "20px"}),

            # Matriz de Confusión + Feature Importance
            html.Div([
                html.Div([
                    html.H3("Matriz de Confusión — Random Forest",
                            style={"fontSize": "15px", "fontWeight": "600",
                                   "marginBottom": "4px", "color": COLORES["text"]}),
                    html.P("Calculada sobre el subconjunto filtrado",
                           style={"fontSize": "12px", "color": COLORES["muted"],
                                  "marginBottom": "8px"}),
                    dcc.Graph(id="graph-cm", config={"displayModeBar": False}),
                ], style={**STYLE_CARD, "flex": "1", "minWidth": "280px",
                          "marginBottom": "0"}),

                html.Div([
                    html.H3("Top 10 Variables Más Importantes",
                            style={"fontSize": "15px", "fontWeight": "600",
                                   "marginBottom": "4px", "color": COLORES["text"]}),
                    html.P("Random Forest — Subconjunto filtrado",
                           style={"fontSize": "12px", "color": COLORES["muted"],
                                  "marginBottom": "8px"}),
                    dcc.Graph(id="graph-feat-imp", config={"displayModeBar": False}),
                ], style={**STYLE_CARD, "flex": "1", "minWidth": "280px",
                          "marginBottom": "0"}),
            ], style=STYLE_ROW),

            # Clustering
            html.Div([
                html.H3("Segmentación KMeans (K=3) — Proyección PCA",
                        style={"fontSize": "15px", "fontWeight": "600",
                               "marginBottom": "4px", "color": COLORES["text"]}),
                html.P(
                    "Cada punto representa una app. Los colores indican el cluster asignado. "
                    "Los ejes corresponden a los dos primeros componentes principales (PCA).",
                    style={"fontSize": "12px", "color": COLORES["muted"],
                           "marginBottom": "8px"},
                ),
                dcc.Graph(id="graph-cluster", config={"displayModeBar": False}),
            ], style=STYLE_CARD),

            # Comparación dinámica de métricas (barras)
            html.Div([
                html.H3("Comparación Dinámica de Modelos — Subconjunto Filtrado",
                        style={"fontSize": "15px", "fontWeight": "600",
                               "marginBottom": "4px", "color": COLORES["text"]}),
                html.P("Métricas recalculadas sobre el subconjunto seleccionado por los filtros",
                       style={"fontSize": "12px", "color": COLORES["muted"],
                              "marginBottom": "8px"}),
                dcc.Graph(id="graph-metrics-compare", config={"displayModeBar": False}),
            ], style=STYLE_CARD),

        ], style={"padding": "24px"}),
    ]
)

# ── Layout principal ──────────────────────────────────────────────────────────
app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.H1("Google Play Store Analytics",
                    style={"margin": "0", "fontSize": "24px", "fontWeight": "700"}),
            html.P("Dashboard de análisis estratégico y rendimiento de Machine Learning",
                   style={"margin": "4px 0 0 0", "fontSize": "13px", "opacity": "0.85"}),
        ]),
        html.Div([
            html.Span("EV3 · SCY1101", style={
                "backgroundColor": "rgba(255,255,255,0.2)",
                "borderRadius": "20px", "padding": "6px 14px",
                "fontSize": "12px", "fontWeight": "600",
            }),
        ]),
    ], style={**STYLE_HEADER,
               "display": "flex", "justifyContent": "space-between",
               "alignItems": "center"}),

    # Cuerpo
    html.Div([
        # Filtros globales
        html.Div([
            html.H4("Filtros Globales",
                    style={"fontSize": "13px", "fontWeight": "700",
                           "textTransform": "uppercase", "color": COLORES["muted"],
                           "marginBottom": "12px", "letterSpacing": "0.5px"}),
            filtros_layout,
        ], style={"padding": "20px 32px 0 32px"}),

        # Tabs
        html.Div([
            dcc.Tabs(id="tabs", value="gerencial",
                     style={"borderBottom": f"1px solid {COLORES['border']}"},
                     children=[tab_gerencial, tab_tecnica]),
        ], style={"padding": "0 32px"}),
    ]),
], style=STYLE_PAGE)

# =============================================================================
# 5. CALLBACKS
# =============================================================================

def filter_df(cat_val, cr_val):
    """Devuelve el DataFrame filtrado según los dropdowns."""
    dff = df_base.copy()
    if cat_val and cat_val != "ALL":
        dff = dff[dff["Category"] == cat_val]
    if cr_val and cr_val != "ALL":
        dff = dff[dff["Content Rating"] == cr_val]
    return dff


# ── Contador de filas filtradas ───────────────────────────────────────────────
@app.callback(
    Output("lbl-filtered", "children"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_label(cat, cr):
    dff = filter_df(cat, cr)
    return f"Mostrando {len(dff):,} apps"


# ── KPIs gerenciales ──────────────────────────────────────────────────────────
@app.callback(
    Output("kpi-total",   "children"),
    Output("kpi-rating",  "children"),
    Output("kpi-success", "children"),
    Output("kpi-free",    "children"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_kpis_gerencial(cat, cr):
    dff = filter_df(cat, cr)
    total   = len(dff)
    avg_rat = dff["Rating"].mean() if total > 0 else 0
    pct_alt = (dff["Rating_Alto"].sum() / total * 100) if total > 0 else 0
    pct_fr  = (
        (dff["Type"].str.upper() == "FREE").sum() / total * 100
        if total > 0 else 0
    )
    return (
        f"{total:,}",
        f"{avg_rat:.2f} ⭐",
        f"{pct_alt:.1f}%",
        f"{pct_fr:.1f}%",
    )


# ── Top 10 categorías ─────────────────────────────────────────────────────────
@app.callback(
    Output("graph-top10", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
    Input("radio-top10", "value"),
)
def update_top10(cat, cr, metric):
    dff = filter_df(cat, cr)
    top = (
        dff.groupby("Category")[metric]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    label = "Instalaciones" if metric == "Installs" else "Reviews"
    fig = px.bar(
        top, x=metric, y="Category", orientation="h",
        color=metric,
        color_continuous_scale=[[0, "#c5d8fb"], [1, COLORES["primary"]]],
        labels={metric: label, "Category": "Categoría"},
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False,
        margin=dict(l=0, r=20, t=10, b=10),
        yaxis={"categoryorder": "total ascending"},
        height=360,
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# ── Free vs Paid (pie) ────────────────────────────────────────────────────────
@app.callback(
    Output("graph-freepaid", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_freepaid(cat, cr):
    dff = filter_df(cat, cr)
    counts = dff["Type"].value_counts().reset_index()
    counts.columns = ["Type", "Count"]
    fig = px.pie(
        counts, names="Type", values="Count",
        color="Type",
        color_discrete_map={
            "Free": COLORES["secondary"],
            "Paid": COLORES["danger"],
        },
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label",
                      pull=[0.03, 0.03])
    fig.update_layout(
        showlegend=True, legend=dict(orientation="h", y=-0.05),
        margin=dict(l=0, r=0, t=10, b=20),
        paper_bgcolor="white",
        height=360,
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# ── Instalaciones por categoría (barras agrupadas) ───────────────────────────
@app.callback(
    Output("graph-installs-cat", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_installs_cat(cat, cr):
    dff = filter_df(cat, cr)
    grp = (
        dff.groupby(["Category", "Type"])["Installs"]
        .mean()
        .reset_index()
    )
    top_cats = (
        dff.groupby("Category")["Installs"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .index.tolist()
    )
    grp = grp[grp["Category"].isin(top_cats)]
    fig = px.bar(
        grp, x="Category", y="Installs", color="Type", barmode="group",
        color_discrete_map={
            "Free": COLORES["secondary"],
            "Paid": COLORES["danger"],
        },
        labels={"Installs": "Instalaciones promedio", "Category": "Categoría"},
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_tickangle=-35,
        margin=dict(l=0, r=0, t=10, b=80),
        height=380,
        legend=dict(title="Tipo"),
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# ── Matriz de Confusión ───────────────────────────────────────────────────────
@app.callback(
    Output("graph-cm", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_cm(cat, cr):
    dff = filter_df(cat, cr)
    result = build_pipeline_and_metrics(dff)
    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes para entrenar el modelo",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=COLORES["muted"]))
        fig.update_layout(paper_bgcolor="white", height=300)
        return fig

    cm = result["cm"]
    labels = ["No Alto (0)", "Alto (1)"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale=[[0, "#e8f0fe"], [1, COLORES["primary"]]],
        labels=dict(x="Predicción", y="Real", color="Cantidad"),
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        height=300,
        font=dict(family="Segoe UI, Arial"),
        coloraxis_showscale=False,
        xaxis_title="Predicción",
        yaxis_title="Real",
    )
    fig.update_traces(
        textfont=dict(size=18, color="white"),
        hovertemplate="Real: %{y}<br>Predicción: %{x}<br>Cantidad: %{z}<extra></extra>",
    )
    return fig


# ── Feature Importance ────────────────────────────────────────────────────────
@app.callback(
    Output("graph-feat-imp", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_feat_imp(cat, cr):
    dff = filter_df(cat, cr)
    result = build_pipeline_and_metrics(dff)
    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=COLORES["muted"]))
        fig.update_layout(paper_bgcolor="white", height=300)
        return fig

    feat = result["feat_imp"].reset_index()
    feat.columns = ["Variable", "Importancia"]

    # Limpiar prefijos de sklearn (num__, cat__x0_…)
    feat["Variable"] = (
        feat["Variable"]
        .str.replace(r"^num__", "", regex=True)
        .str.replace(r"^cat__[^_]+_", "", regex=True)
    )

    fig = px.bar(
        feat, x="Importancia", y="Variable", orientation="h",
        color="Importancia",
        color_continuous_scale=[[0, "#b7e4c7"], [1, COLORES["secondary"]]],
        text="Importancia",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False,
        yaxis={"categoryorder": "total ascending"},
        margin=dict(l=0, r=20, t=10, b=10),
        height=300,
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# ── Clustering ────────────────────────────────────────────────────────────────
@app.callback(
    Output("graph-cluster", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_cluster(cat, cr):
    dff = filter_df(cat, cr)
    result = build_pipeline_and_metrics(dff)
    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes para clustering",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=COLORES["muted"]))
        fig.update_layout(paper_bgcolor="white", height=350)
        return fig

    df_cl = result["df_cluster"]
    fig = px.scatter(
        df_cl, x="PCA_1", y="PCA_2",
        color="Cluster",
        color_discrete_map={
            "0": COLORES["primary"],
            "1": COLORES["secondary"],
            "2": COLORES["accent"],
        },
        hover_data={"Category": True, "Rating": True,
                    "Installs": True, "PCA_1": False, "PCA_2": False},
        labels={"PCA_1": "Componente Principal 1",
                "PCA_2": "Componente Principal 2",
                "Cluster": "Segmento"},
        opacity=0.65,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        plot_bgcolor="#f8f9fa", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(title="Segmento KMeans"),
        height=400,
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# ── Comparación dinámica de modelos ──────────────────────────────────────────
@app.callback(
    Output("graph-metrics-compare", "figure"),
    Input("dd-category", "value"),
    Input("dd-content-rating", "value"),
)
def update_metrics_compare(cat, cr):
    dff = filter_df(cat, cr)
    result = build_pipeline_and_metrics(dff)
    if result is None:
        fig = go.Figure()
        fig.add_annotation(text="Datos insuficientes",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color=COLORES["muted"]))
        fig.update_layout(paper_bgcolor="white", height=300)
        return fig

    metricas = result["metricas"]
    rows = []
    for model_name, vals in metricas.items():
        for metric, val in vals.items():
            rows.append({"Modelo": model_name, "Métrica": metric, "Valor": val})

    df_m = pd.DataFrame(rows)
    fig = px.bar(
        df_m, x="Métrica", y="Valor", color="Modelo", barmode="group",
        color_discrete_map={
            "Regresión Logística": COLORES["primary"],
            "Random Forest":       COLORES["secondary"],
        },
        text="Valor",
        range_y=[0, 1.05],
    )
    fig.update_traces(texttemplate="%{text:.2%}", textposition="outside")
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=10),
        legend=dict(title="Modelo"),
        height=320,
        font=dict(family="Segoe UI, Arial"),
    )
    return fig


# =============================================================================
# 6. MAIN
# =============================================================================
if __name__ == "__main__":
    app.run(debug=True, port=8050)
