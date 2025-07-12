import plotly.graph_objects as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


# 1. ОСНОВНАЯ ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ГРАФИКА

def create_bundle_figure(bundle_type, p_x, p_y, p_angle, dist, show_curve, num_lines=20):
    fig = go.Figure()
    r = 1.0  # Радиус абсолюта

    # Словарь для перевода названий на русский
    bundle_titles = {
        'elliptic': 'Эллиптический',
        'parabolic': 'Параболический',
        'hyperbolic': 'Гиперболический'
    }

    # Отрисовка абсолюта
    fig.add_shape(
        type="circle", xref="x", yref="y",
        x0=-r, y0=-r, x1=r, y1=r,
        line_color="RoyalBlue", fillcolor="LightSkyBlue",
        opacity=0.2, line_width=2
    )
    

    # Эллиптический пучок (пересекающиеся) 
    if bundle_type == 'elliptic':
        center_point = np.array([p_x, p_y])
        if np.linalg.norm(center_point) >= r:
            center_point = center_point / np.linalg.norm(center_point) * (r - 0.01)

        for i in range(num_lines):
            angle = i * np.pi / num_lines
            direction = np.array([np.cos(angle), np.sin(angle)])
            a, b, c = 1.0, 2*np.dot(center_point, direction), np.dot(center_point,center_point) - r**2
            discriminant = b**2 - 4*a*c
            if discriminant < 0: continue
            t1, t2 = (-b + np.sqrt(discriminant))/(2*a), (-b - np.sqrt(discriminant))/(2*a)
            p_start, p_end = center_point + t1*direction, center_point + t2*direction
            fig.add_trace(go.Scatter(x=[p_start[0], p_end[0]], y=[p_start[1], p_end[1]], mode='lines', line=dict(color='firebrick', width=1.5)))
        fig.add_trace(go.Scatter(x=[center_point[0]], y=[center_point[1]], mode='markers', marker=dict(color='black', size=8, symbol='circle')))

        if show_curve:
            # Ортогональная кривая - гиперболическая окружность (в модели Клейна - эллипс)
            dist_center = np.linalg.norm(center_point)
            squash_factor = np.sqrt(max(1.0 - dist_center**2, 1e-9))
            radius_euclidean = 0.4 # фиксированный радиус для примера
            
            radius_parallel = radius_euclidean * squash_factor
            radius_perp = radius_euclidean

            t = np.linspace(0, 2*np.pi, 100)
            x_ellipse_std = radius_perp * np.cos(t)
            y_ellipse_std = radius_parallel * np.sin(t)
            
            angle_center = np.arctan2(center_point[1], center_point[0])
            x_ellipse = x_ellipse_std*np.cos(angle_center) - y_ellipse_std*np.sin(angle_center) + center_point[0]
            y_ellipse = x_ellipse_std*np.sin(angle_center) + y_ellipse_std*np.cos(angle_center) + center_point[1]
            fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', line=dict(color='blue', width=3, dash='solid'), name='Гиперб. окружность'))


    #Параболический пучок (параллельные)
    elif bundle_type == 'parabolic':
        angle_rad = np.deg2rad(p_angle)
        ideal_point = np.array([r * np.cos(angle_rad), r * np.sin(angle_rad)])
        for i in range(num_lines + 1):
            end_angle = angle_rad + np.pi/2 + (np.pi * (i + 0.5) / (num_lines+1))
            end_point = np.array([r * np.cos(end_angle), r * np.sin(end_angle)])
            fig.add_trace(go.Scatter(x=[ideal_point[0], end_point[0]], y=[ideal_point[1], end_point[1]], mode='lines', line=dict(color='seagreen', width=1.5)))
        fig.add_trace(go.Scatter(x=[ideal_point[0]], y=[ideal_point[1]], mode='markers', marker=dict(color='black', size=8, symbol='diamond')))
        
        if show_curve:
            # Ортогональная кривая - орицикл (в модели Клейна - окружность, касающаяся абсолюта)
            r_horo = 0.5 # Фиксированный радиус для примера
            center_horo = ideal_point * (1 - r_horo / r)
            t = np.linspace(0, 2*np.pi, 100)
            x_horo = center_horo[0] + r_horo * np.cos(t)
            y_horo = center_horo[1] + r_horo * np.sin(t)
            fig.add_trace(go.Scatter(x=x_horo, y=y_horo, mode='lines', line=dict(color='blue', width=3), name='Орицикл'))

    # Гиперболический пучок (расходящиеся)
    elif bundle_type == 'hyperbolic':
        angle_rad = np.deg2rad(p_angle)
        midpoint = dist * np.array([np.cos(angle_rad), np.sin(angle_rad)])
        direction = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
        half_chord_len = np.sqrt(max(r**2 - dist**2, 0))
        p1, p2 = midpoint - half_chord_len * direction, midpoint + half_chord_len * direction
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode='lines', line=dict(color='black', width=3, dash='dash'), name='Общий перпендикуляр'))

        denominator = p1[0]*p2[1] - p2[0]*p1[1]
        if np.abs(denominator) < 1e-9:
             for i in range(num_lines//2):
                angle = angle_rad + np.pi/2 + (i * np.pi / num_lines)
                end1, end2 = r*np.array([np.cos(angle),np.sin(angle)]), -r*np.array([np.cos(angle),np.sin(angle)])
                fig.add_trace(go.Scatter(x=[end1[0], end2[0]], y=[end1[1], end2[1]], mode='lines', line=dict(color='darkorange', width=1.5)))
        else:
            pole_x, pole_y = r**2*(p2[1]-p1[1])/denominator, r**2*(p1[0]-p2[0])/denominator
            pole = np.array([pole_x, pole_y])
            dist_pole = np.linalg.norm(pole)
            if dist_pole > r:
                angle_to_pole, half_angle_of_view = np.arctan2(pole[1],pole[0]), np.arcsin(r/dist_pole)
                start_angle, end_angle = angle_to_pole - half_angle_of_view, angle_to_pole + half_angle_of_view
                for angle in np.linspace(start_angle, end_angle, num_lines):
                    line_dir = np.array([np.cos(angle), np.sin(angle)])
                    a,b,c = 1.0, 2*np.dot(pole,line_dir), np.dot(pole,pole)-r**2
                    discriminant = b**2 - 4*a*c
                    if discriminant<0 and np.isclose(discriminant,0): discriminant = 0
                    if discriminant < 0: continue
                    t1,t2 = (-b+np.sqrt(discriminant))/(2*a), (-b-np.sqrt(discriminant))/(2*a)
                    p_start, p_end = pole+t1*line_dir, pole+t2*line_dir
                    fig.add_trace(go.Scatter(x=[p_start[0], p_end[0]], y=[p_start[1], p_end[1]], mode='lines', line=dict(color='darkorange', width=1.5)))
        
        if show_curve:
            # Ортогональная кривая - эквидистанта
            # Для простоты воспользуемся дугой окружности, проходящей через те же точки
            bulge = 0.3 # Фиксированный "изгиб" для примера
            center = midpoint + bulge * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            radius = np.linalg.norm(p1 - center)
            start_angle = np.arctan2(p1[1]-center[1], p1[0]-center[0])
            end_angle = np.arctan2(p2[1]-center[1], p2[0]-center[0])
            # правильное направление дуги
            if np.abs(start_angle - end_angle) > np.pi:
                if start_angle < end_angle: start_angle += 2*np.pi
                else: end_angle += 2*np.pi
            
            t = np.linspace(start_angle, end_angle, 100)
            x_arc, y_arc = center[0] + radius*np.cos(t), center[1] + radius*np.sin(t)
            fig.add_trace(go.Scatter(x=x_arc, y=y_arc, mode='lines', line=dict(color='blue', width=3), name='Эквидистанта'))


    # Настройки вида
    fig.update_layout(
        title=f'Модель Кэли-Клейна: {bundle_titles.get(bundle_type, "")} пучок',
        xaxis=dict(range=[-1.1, 1.1], visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[-1.1, 1.1], visible=False),
        showlegend=False,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    return fig

# 2. СОЗДАНИЕ ПРИЛОЖЕНИЯ DASH

app = dash.Dash(__name__)

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'width': '90%', 'margin': 'auto'}, children=[
    html.H1("Пучки прямых в модели Кэли-Клейна", style={'textAlign': 'center'}),
    html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
        html.Div(dcc.Graph(id='klein-bundle-graph'), style={'width': '60%'}),
        html.Div(style={'width': '35%', 'paddingLeft': '5%'}, children=[
            html.H3("Тип пучка"),
            dcc.RadioItems(
                id='bundle-type-selector',
                options=[
                    {'label': ' Эллиптический (пересекающиеся)', 'value': 'elliptic'},
                    {'label': ' Параболический (параллельные)', 'value': 'parabolic'},
                    {'label': ' Гиперболический (расходящиеся)', 'value': 'hyperbolic'},
                ], value='elliptic', labelStyle={'display': 'block', 'marginBottom': '10px'}
            ),
            html.Hr(),
            html.Div(id='elliptic-controls', children=[
                html.Label("Центр X:"), dcc.Slider(id='p_x-slider', min=-0.9, max=0.9, step=0.05, value=0.2),
                html.Label("Центр Y:"), dcc.Slider(id='p_y-slider', min=-0.9, max=0.9, step=0.05, value=0.3),
            ]),
            html.Div(id='parabolic-controls', style={'display': 'none'}, children=[
                html.Label("Угол идеальной точки (градусы):"),
                dcc.Slider(id='p_angle-parabolic-slider', min=0, max=360, step=1, value=45),
            ]),
            html.Div(id='hyperbolic-controls', style={'display': 'none'}, children=[
                html.Label("Угол общего перпендикуляра:"),
                dcc.Slider(id='p_angle-hyperbolic-slider', min=0, max=180, step=1, value=30),
                html.Label("Смещение перпендикуляра от центра:"),
                dcc.Slider(id='dist-hyperbolic-slider', min=0, max=0.95, step=0.05, value=0.5),
            ]),
            html.Hr(),
            html.Button('Построить/скрыть ортогональную кривую', id='toggle-curve-button', n_clicks=0, style={'width': '100%', 'padding': '10px'}),
            dcc.Store(id='show-curve-store', data=False)
        ])
    ])
])

# 3. ЛОГИКА ОБНОВЛЕНИЯ ГРАФИКА И ИНТЕРФЕЙСА
# (чтобы "камера" не возвращалась в исходное положение при изменении картинки)
@app.callback(
    Output('show-curve-store', 'data'),
    [Input('toggle-curve-button', 'n_clicks')],
    [State('show-curve-store', 'data')]
)
def toggle_curve(n_clicks, show_curve):
    if n_clicks > 0:
        return not show_curve
    return show_curve

@app.callback(
    Output('klein-bundle-graph', 'figure'),
    [Input('bundle-type-selector', 'value'), Input('p_x-slider', 'value'), Input('p_y-slider', 'value'),
     Input('p_angle-parabolic-slider', 'value'), Input('p_angle-hyperbolic-slider', 'value'), Input('dist-hyperbolic-slider', 'value'),
     Input('show-curve-store', 'data')]
)
def update_graph(bundle_type, p_x, p_y, p_angle_para, p_angle_hyper, dist_hyper, show_curve):
    if bundle_type == 'parabolic': return create_bundle_figure(bundle_type, 0, 0, p_angle_para, 0, show_curve)
    elif bundle_type == 'hyperbolic': return create_bundle_figure(bundle_type, 0, 0, p_angle_hyper, dist_hyper, show_curve)
    else: return create_bundle_figure(bundle_type, p_x, p_y, 0, 0, show_curve)

@app.callback(
    [Output('elliptic-controls', 'style'), Output('parabolic-controls', 'style'), Output('hyperbolic-controls', 'style')],
    [Input('bundle-type-selector', 'value')]
)
def update_controls_visibility(bundle_type):
    if bundle_type == 'parabolic': return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif bundle_type == 'hyperbolic': return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
    else: return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}


# 4. ЗАПУСК ПРИЛОЖЕНИЯ

if __name__ == '__main__':
    app.run(debug=True)
