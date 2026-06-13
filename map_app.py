import json
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

# ── начальные данные ──────────────────────────────────────────────────────────

CITY_COORDS = {
    "Москва":       (55.7558, 37.6176),
    "Воронеж":      (51.6683, 39.1844),
    "Губкин":       (51.2833, 37.5500),
    "Калининград":  (54.7104, 20.4522),
    "Тамбов":       (52.7212, 41.4523),
    "Ижевск":       (56.8527, 53.2114),
    "Волгоград":    (48.7080, 44.5133),
    "Пятигорск":    (44.0398, 43.0617),
}

SERVICE_COLORS = {
    "Волонтёры образовательных программ": "#FF6B6B",
}

INITIAL_DATA = {
    "services": ["Волонтёры образовательных программ"],
    "volunteers": [
        {"handle": "@lizabeta_b",       "name": "Лиза",   "cities": ["Воронеж", "Губкин"],      "service": "Волонтёры образовательных программ"},
        {"handle": "@p.nekr",           "name": "П.",     "cities": ["Калининград"],             "service": "Волонтёры образовательных программ"},
        {"handle": "@peaid",            "name": "П.",     "cities": ["Тамбов", "Ижевск"],        "service": "Волонтёры образовательных программ"},
        {"handle": "@a_z1609",          "name": "А.",     "cities": ["Волгоград", "Пятигорск"],  "service": "Волонтёры образовательных программ"},
        {"handle": "@daria_volkova328", "name": "Дарья",  "cities": ["Москва"],                  "service": "Волонтёры образовательных программ"},
    ],
}

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9",
]

# ── вспомогательные функции ───────────────────────────────────────────────────

def get_color_map(services):
    cmap = {}
    for i, s in enumerate(services):
        cmap[s] = PALETTE[i % len(PALETTE)]
    return cmap


def build_figure(data):
    color_map = get_color_map(data["services"])

    # группируем точки по службе
    traces_by_service = {}
    for v in data["volunteers"]:
        svc = v["service"]
        if svc not in traces_by_service:
            traces_by_service[svc] = {"lats": [], "lons": [], "texts": [], "customdata": []}
        for city in v["cities"]:
            coords = CITY_COORDS.get(city)
            if coords:
                lat, lon = coords
                traces_by_service[svc]["lats"].append(lat)
                traces_by_service[svc]["lons"].append(lon)
                traces_by_service[svc]["texts"].append(f"{v['handle']}<br>{city}")
                traces_by_service[svc]["customdata"].append(city)

    fig = go.Figure()

    for svc, pts in traces_by_service.items():
        color = color_map.get(svc, "#888")
        fig.add_trace(go.Scattergeo(
            lat=pts["lats"],
            lon=pts["lons"],
            text=pts["texts"],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=11, color="#333"),
            marker=dict(
                size=14,
                color=color,
                line=dict(width=2, color="white"),
                symbol="circle",
            ),
            hovertemplate="<b>%{text}</b><extra></extra>",
            name=svc,
        ))

    fig.update_layout(
        geo=dict(
            scope="asia",
            resolution=50,
            showland=True,
            landcolor="#F0EEE9",
            showocean=True,
            oceancolor="#D6EAF8",
            showcountries=True,
            countrycolor="#CCCCCC",
            showrivers=True,
            rivercolor="#AED6F1",
            showlakes=True,
            lakecolor="#D6EAF8",
            center=dict(lat=62, lon=80),
            projection_scale=2.5,
            lataxis_range=[40, 80],
            lonaxis_range=[18, 170],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            x=0.01, y=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc",
            borderwidth=1,
            font=dict(size=12),
        ),
        paper_bgcolor="#FAFAFA",
        plot_bgcolor="#FAFAFA",
        height=560,
    )
    return fig


def build_volunteer_list(data):
    color_map = get_color_map(data["services"])
    items = []
    for v in data["volunteers"]:
        color = color_map.get(v["service"], "#888")
        items.append(
            html.Div([
                html.Span(
                    v["handle"],
                    style={"fontWeight": "bold", "color": color, "fontSize": "14px"},
                ),
                html.Span(
                    " · " + ", ".join(v["cities"]),
                    style={"color": "#555", "fontSize": "13px"},
                ),
                html.Br(),
                html.Span(
                    v["service"],
                    style={
                        "fontSize": "11px",
                        "color": "white",
                        "background": color,
                        "padding": "1px 8px",
                        "borderRadius": "10px",
                        "display": "inline-block",
                        "marginTop": "3px",
                    },
                ),
            ], style={
                "padding": "10px 12px",
                "marginBottom": "8px",
                "background": "white",
                "borderRadius": "10px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                "borderLeft": f"4px solid {color}",
            })
        )
    return items


# ── layout ────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Форум ШУМ 2026 — Карта команды")
server = app.server

def _input_style():
    return {
        "width": "100%",
        "marginBottom": "8px",
        "padding": "8px 10px",
        "border": "1px solid #ddd",
        "borderRadius": "8px",
        "fontSize": "13px",
        "boxSizing": "border-box",
    }


def _btn_style(color):
    return {
        "width": "100%",
        "padding": "9px",
        "background": color,
        "color": "white",
        "border": "none",
        "borderRadius": "8px",
        "cursor": "pointer",
        "fontSize": "13px",
        "fontWeight": "600",
    }


def _card_style():
    return {
        "background": "white",
        "borderRadius": "12px",
        "padding": "16px",
        "marginBottom": "12px",
        "boxShadow": "0 4px 16px rgba(0,0,0,0.10)",
    }


app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Arial, sans-serif",
        "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "minHeight": "100vh",
        "padding": "20px",
    },
    children=[
        # хранилище данных
        dcc.Store(id="data-store", data=INITIAL_DATA),

        # заголовок
        html.Div([
            html.H1(
                "Форум ШУМ · 2026",
                style={"color": "white", "margin": "0", "fontSize": "28px", "fontWeight": "700"},
            ),
            html.P(
                "Волонтёры образовательных программ · География команды",
                style={"color": "rgba(255,255,255,0.85)", "margin": "4px 0 0", "fontSize": "14px"},
            ),
        ], style={"textAlign": "center", "marginBottom": "20px"}),

        # основной контейнер
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
            children=[

                # ── карта ────────────────────────────────────────────────────
                html.Div(
                    style={
                        "flex": "1",
                        "background": "white",
                        "borderRadius": "16px",
                        "padding": "16px",
                        "boxShadow": "0 8px 32px rgba(0,0,0,0.15)",
                    },
                    children=[
                        dcc.Graph(id="map-graph", config={"scrollZoom": True}),
                    ]
                ),

                # ── правая панель ─────────────────────────────────────────────
                html.Div(
                    style={"width": "300px", "flexShrink": "0"},
                    children=[

                        # список волонтёров
                        html.Div([
                            html.H3("Участники", style={"color": "white", "margin": "0 0 12px", "fontSize": "16px"}),
                            html.Div(id="volunteer-list"),
                        ], style={"marginBottom": "16px"}),

                        # форма добавления волонтёра
                        html.Div([
                            html.H3(
                                "Добавить участника",
                                style={"margin": "0 0 12px", "fontSize": "15px", "color": "#4a4a6a"},
                            ),
                            dcc.Input(id="in-handle", placeholder="@telegram", debounce=False,
                                      style=_input_style()),
                            dcc.Input(id="in-name", placeholder="Имя", debounce=False,
                                      style=_input_style()),
                            dcc.Input(id="in-cities", placeholder="Город1, Город2", debounce=False,
                                      style=_input_style()),
                            dcc.Dropdown(
                                id="in-service",
                                placeholder="Служба",
                                style={"marginBottom": "8px", "fontSize": "13px"},
                            ),
                            html.Button(
                                "Добавить",
                                id="btn-add-volunteer",
                                n_clicks=0,
                                style=_btn_style("#667eea"),
                            ),
                            html.Div(id="msg-volunteer", style={"fontSize": "12px", "color": "#e74c3c", "marginTop": "4px"}),
                        ], style=_card_style()),

                        # форма добавления города
                        html.Div([
                            html.H3(
                                "Добавить город",
                                style={"margin": "0 0 12px", "fontSize": "15px", "color": "#4a4a6a"},
                            ),
                            dcc.Input(id="in-city-name", placeholder="Название города", debounce=False,
                                      style=_input_style()),
                            dcc.Input(id="in-city-lat", placeholder="Широта (55.75)", debounce=False,
                                      style=_input_style()),
                            dcc.Input(id="in-city-lon", placeholder="Долгота (37.62)", debounce=False,
                                      style=_input_style()),
                            html.Button(
                                "Добавить город",
                                id="btn-add-city",
                                n_clicks=0,
                                style=_btn_style("#4ECDC4"),
                            ),
                            html.Div(id="msg-city", style={"fontSize": "12px", "color": "#e74c3c", "marginTop": "4px"}),
                        ], style=_card_style()),

                        # форма добавления службы
                        html.Div([
                            html.H3(
                                "Добавить службу",
                                style={"margin": "0 0 12px", "fontSize": "15px", "color": "#4a4a6a"},
                            ),
                            dcc.Input(id="in-service-name", placeholder="Название службы", debounce=False,
                                      style=_input_style()),
                            html.Button(
                                "Добавить службу",
                                id="btn-add-service",
                                n_clicks=0,
                                style=_btn_style("#96CEB4"),
                            ),
                            html.Div(id="msg-service", style={"fontSize": "12px", "color": "#e74c3c", "marginTop": "4px"}),
                        ], style=_card_style()),
                    ]
                ),
            ]
        ),
    ]
)


# ── callbacks ─────────────────────────────────────────────────────────────────

# обновляем dropdown служб при изменении хранилища
@app.callback(
    Output("in-service", "options"),
    Input("data-store", "data"),
)
def update_service_options(data):
    return [{"label": s, "value": s} for s in data["services"]]


# добавление службы
@app.callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("msg-service", "children"),
    Output("in-service-name", "value"),
    Input("btn-add-service", "n_clicks"),
    State("in-service-name", "value"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def add_service(n, name, data):
    if not name or not name.strip():
        return data, "Введите название службы", ""
    name = name.strip()
    if name in data["services"]:
        return data, "Такая служба уже есть", ""
    data = dict(data)
    data["services"] = data["services"] + [name]
    return data, "", ""


# добавление города (только в CITY_COORDS через глобальный словарь)
@app.callback(
    Output("msg-city", "children"),
    Output("in-city-name", "value"),
    Output("in-city-lat", "value"),
    Output("in-city-lon", "value"),
    Input("btn-add-city", "n_clicks"),
    State("in-city-name", "value"),
    State("in-city-lat", "value"),
    State("in-city-lon", "value"),
    prevent_initial_call=True,
)
def add_city(n, city_name, lat, lon):
    if not city_name or not lat or not lon:
        return "Заполните все поля", city_name, lat, lon
    try:
        lat_f, lon_f = float(lat), float(lon)
    except ValueError:
        return "Широта и долгота — числа", city_name, lat, lon
    city_name = city_name.strip()
    CITY_COORDS[city_name] = (lat_f, lon_f)
    return f"✓ {city_name} добавлен", "", "", ""


# добавление волонтёра
@app.callback(
    Output("data-store", "data", allow_duplicate=True),
    Output("msg-volunteer", "children"),
    Output("in-handle", "value"),
    Output("in-name", "value"),
    Output("in-cities", "value"),
    Output("in-service", "value"),
    Input("btn-add-volunteer", "n_clicks"),
    State("in-handle", "value"),
    State("in-name", "value"),
    State("in-cities", "value"),
    State("in-service", "value"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def add_volunteer(n, handle, name, cities_str, service, data):
    if not handle or not cities_str or not service:
        return data, "Заполните handle, города и службу", handle, name, cities_str, service
    handle = handle.strip()
    cities = [c.strip() for c in cities_str.split(",") if c.strip()]
    unknown = [c for c in cities if c not in CITY_COORDS]
    if unknown:
        return data, f"Города не найдены: {', '.join(unknown)}. Сначала добавьте их.", handle, name, cities_str, service
    entry = {
        "handle": handle,
        "name": (name or "").strip(),
        "cities": cities,
        "service": service,
    }
    data = dict(data)
    data["volunteers"] = data["volunteers"] + [entry]
    return data, "", "", "", "", None


# обновление карты и списка
@app.callback(
    Output("map-graph", "figure"),
    Output("volunteer-list", "children"),
    Input("data-store", "data"),
)
def refresh(data):
    return build_figure(data), build_volunteer_list(data)


if __name__ == "__main__":
    app.run(debug=True, port=8051)
