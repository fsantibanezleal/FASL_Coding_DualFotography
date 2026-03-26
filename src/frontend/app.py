"""Dash web application for interactive dual photography.

Provides a browser-based interface for:
- Running synthetic dual photography simulations
- Visualizing primal, dual, and relighted images
- Analyzing transport matrix properties (SVD spectrum)
- Interactive parameter control (scene type, resolution, SVD rank)
- Real-time relighting with custom patterns

Launch with: python -m src.frontend.app
"""

from __future__ import annotations

import io
import base64
import traceback

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, ctx
from dash.exceptions import PreventUpdate
from PIL import Image

from src.core.transport import TransportMatrix
from src.simulation.renderer import VirtualRenderer
from src.simulation.scene import SceneType

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Dual Photography Lab",
    suppress_callback_exceptions=True,
)
server = app.server  # Expose for WSGI deployment

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
SCENE_OPTIONS = [
    {"label": "Box + Wall (occlusion)", "value": "box_and_wall"},
    {"label": "Cornell Box (classic)", "value": "cornell_box"},
    {"label": "Gallery (multi-depth)", "value": "gallery"},
    {"label": "Staircase (parallax)", "value": "staircase"},
    {"label": "Mirror Room (specular)", "value": "mirror_room"},
    {"label": "Sphere on Plane", "value": "sphere_on_plane"},
    {"label": "Corner Room", "value": "corner_room"},
    {"label": "Two Angled Planes", "value": "two_planes"},
    {"label": "Cylinder with Text", "value": "cylinder_text"},
    {"label": "Flat Textured Wall", "value": "flat_textured"},
]

RESOLUTION_OPTIONS = [
    {"label": "16 x 16 (fast)", "value": "16"},
    {"label": "24 x 24", "value": "24"},
    {"label": "32 x 32", "value": "32"},
    {"label": "48 x 48 (detailed)", "value": "48"},
    {"label": "64 x 64 (slow)", "value": "64"},
]

RELIGHT_OPTIONS = [
    {"label": "Uniform White", "value": "white"},
    {"label": "Left Half", "value": "left"},
    {"label": "Right Half", "value": "right"},
    {"label": "Top Half", "value": "top"},
    {"label": "Bottom Half", "value": "bottom"},
    {"label": "Center Spot", "value": "spot"},
    {"label": "Horizontal Stripes", "value": "h_stripes"},
    {"label": "Vertical Stripes", "value": "v_stripes"},
    {"label": "Diagonal", "value": "diagonal"},
    {"label": "Random", "value": "random"},
]


def _numpy_to_b64_img(arr: np.ndarray) -> str:
    """Convert a 2D numpy array to a base64-encoded PNG for display.

    Args:
        arr: 2D array with values in [0, max]. Will be normalized to [0, 255].

    Returns:
        Base64-encoded PNG data URI string suitable for html.Img src attribute.
    """
    arr = arr.astype(np.float64)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-10:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr_uint8, mode="L")
    # Upscale for visibility using nearest-neighbor
    scale = max(1, 256 // max(img.size))
    if scale > 1:
        img = img.resize(
            (img.size[0] * scale, img.size[1] * scale), Image.NEAREST
        )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _make_relight_pattern(name: str, h: int, w: int) -> np.ndarray:
    """Generate a named relighting pattern.

    Args:
        name: Pattern preset name from RELIGHT_OPTIONS.
        h: Pattern height in pixels.
        w: Pattern width in pixels.

    Returns:
        Pattern array of shape (h, w) with float64 values in [0, 1].
    """
    pattern = np.zeros((h, w), dtype=np.float64)

    if name == "white":
        pattern[:] = 1.0
    elif name == "left":
        pattern[:, : w // 2] = 1.0
    elif name == "right":
        pattern[:, w // 2:] = 1.0
    elif name == "top":
        pattern[: h // 2, :] = 1.0
    elif name == "bottom":
        pattern[h // 2:, :] = 1.0
    elif name == "spot":
        cy, cx = h // 2, w // 2
        r = max(1, min(h, w) // 4)
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
        pattern[mask] = 1.0
    elif name == "h_stripes":
        pattern[::2, :] = 1.0
    elif name == "v_stripes":
        pattern[:, ::2] = 1.0
    elif name == "diagonal":
        for row in range(h):
            for col in range(w):
                if (row + col) % 2 == 0:
                    pattern[row, col] = 1.0
    elif name == "random":
        pattern = np.random.default_rng(123).random((h, w))
    else:
        pattern[:] = 1.0

    return pattern


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _control_panel() -> dbc.Card:
    """Build the left-side control panel with all simulation parameters."""
    return dbc.Card(
        dbc.CardBody([
            html.H4("Simulation Parameters", className="card-title mb-3"),

            dbc.Label("Scene Type", html_for="scene-type"),
            dbc.Select(
                id="scene-type",
                options=SCENE_OPTIONS,
                value="box_and_wall",
                className="mb-3",
            ),

            dbc.Label("Resolution", html_for="resolution"),
            dbc.Select(
                id="resolution",
                options=RESOLUTION_OPTIONS,
                value="32",
                className="mb-3",
            ),

            dbc.Label("Surface Albedo", html_for="albedo"),
            dbc.Select(
                id="albedo",
                options=[
                    {"label": f"{v:.1f}", "value": str(v)}
                    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ],
                value="0.8",
                className="mb-3",
            ),

            dbc.Label("SVD Rank (truncation)", html_for="svd-rank"),
            dbc.Select(
                id="svd-rank",
                options=[{"label": "Full (no truncation)", "value": "0"}],
                value="0",
                className="mb-3",
            ),

            dbc.Label("Light Bounces (indirect)", html_for="n-bounces"),
            dbc.Select(
                id="n-bounces",
                options=[
                    {"label": "0 (direct only)", "value": "0"},
                    {"label": "1 bounce", "value": "1"},
                    {"label": "2 bounces", "value": "2"},
                    {"label": "3 bounces", "value": "3"},
                ],
                value="0",
                className="mb-3",
            ),

            html.Hr(),

            html.H5("Camera / Projector Position", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Projector X", html_for="proj-x"),
                    dbc.Select(
                        id="proj-x",
                        options=[
                            {"label": str(v), "value": str(v)}
                            for v in [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
                        ],
                        value="-1.5",
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Label("Camera X", html_for="cam-x"),
                    dbc.Select(
                        id="cam-x",
                        options=[
                            {"label": str(v), "value": str(v)}
                            for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
                        ],
                        value="1.5",
                    ),
                ], width=6),
            ], className="mb-3"),

            html.Hr(),

            dbc.Button(
                "Run Simulation",
                id="run-btn",
                color="primary",
                size="lg",
                className="w-100 mb-2",
                n_clicks=0,
            ),
            html.Div(id="status-msg", className="text-center mt-2"),
        ]),
        className="h-100",
    )


def _placeholder_img() -> str:
    """Generate a placeholder gray image for initial state."""
    arr = np.full((8, 8), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    img = img.resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _empty_figure() -> go.Figure:
    """Create an empty placeholder figure."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=20, b=40),
        height=280,
        xaxis=dict(title="Index", visible=True),
        yaxis=dict(title="Singular Value", visible=True),
        annotations=[dict(
            text="Run a simulation first",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray"),
        )],
    )
    return fig


def _results_panel() -> html.Div:
    """Build the right-side results visualization panel."""
    placeholder = _placeholder_img()
    return html.Div([
        # Row 1: Primal + Dual images
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Primal Image (Camera View)", className="text-center"),
                    html.P(
                        "Scene as seen by the camera under uniform projector illumination.",
                        className="text-muted text-center small",
                    ),
                    html.Div(
                        html.Img(id="primal-img", src=placeholder,
                                 style={"maxWidth": "100%", "imageRendering": "pixelated"}),
                        className="text-center",
                    ),
                ])),
            ], md=6),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Dual Image (Projector View)", className="text-center"),
                    html.P(
                        "Scene as seen from the projector position via T-transpose (Helmholtz reciprocity).",
                        className="text-muted text-center small",
                    ),
                    html.Div(
                        html.Img(id="dual-img", src=placeholder,
                                 style={"maxWidth": "100%", "imageRendering": "pixelated"}),
                        className="text-center",
                    ),
                ])),
            ], md=6),
        ], className="mb-3"),

        # Row 2: SVD spectrum + Analysis
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Singular Value Spectrum", className="text-center"),
                    dcc.Graph(id="svd-graph", figure=_empty_figure(),
                              config={"displayModeBar": False}),
                ])),
            ], md=6),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Transport Matrix Analysis", className="text-center"),
                    html.Div(
                        id="analysis-info",
                        children=[html.P("Run a simulation to see analysis.",
                                         className="text-muted")],
                    ),
                ])),
            ], md=6),
        ], className="mb-3"),

        # Row 3: Relighting
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5("Relighting", className="text-center"),
                    html.P(
                        "Select a pattern and click Relight to see the scene under new illumination.",
                        className="text-muted text-center small mb-3",
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Relighting Pattern", html_for="relight-pattern"),
                            dbc.Select(
                                id="relight-pattern",
                                options=RELIGHT_OPTIONS,
                                value="white",
                                className="mb-2",
                            ),
                            dbc.Button(
                                "Relight",
                                id="relight-btn",
                                color="success",
                                className="w-100 mt-2",
                                n_clicks=0,
                            ),
                        ], md=4),
                        dbc.Col([
                            html.P("Pattern", className="text-center text-muted small mb-1"),
                            html.Div(
                                html.Img(id="relight-pattern-img", src=placeholder,
                                         style={"maxWidth": "100%", "imageRendering": "pixelated"}),
                                className="text-center",
                            ),
                        ], md=4),
                        dbc.Col([
                            html.P("Relighted Result", className="text-center text-muted small mb-1"),
                            html.Div(
                                html.Img(id="relight-result-img", src=placeholder,
                                         style={"maxWidth": "100%", "imageRendering": "pixelated"}),
                                className="text-center",
                            ),
                        ], md=4),
                    ]),
                ])),
            ]),
        ], className="mb-3"),
    ])


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.H2("Dual Photography Lab", className="text-center my-3"),
            width=12,
        ),
    ]),
    dbc.Row([
        dbc.Col(
            html.P(
                "Interactive exploration of dual photography via light transport "
                "matrix acquisition and Helmholtz reciprocity. "
                "Based on Sen et al. (SIGGRAPH 2005).",
                className="text-muted text-center mb-3",
            ),
            width=12,
        ),
    ]),
    dbc.Row([
        dbc.Col(_control_panel(), md=3),
        dbc.Col(_results_panel(), md=9),
    ]),
    # Hidden stores
    dcc.Store(id="transport-store", data=None),
], fluid=True, className="py-3")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("svd-rank", "options"),
    Input("resolution", "value"),
)
def update_svd_options(resolution_str: str):
    """Update SVD rank dropdown options based on selected resolution.

    Generates rank options as powers of 2 up to resolution^2,
    plus a 'Full' option for no truncation.
    """
    resolution = int(resolution_str)
    max_rank = resolution * resolution
    options = [{"label": "Full (no truncation)", "value": "0"}]
    rank = 1
    while rank < max_rank:
        options.append({"label": f"Rank {rank}", "value": str(rank)})
        rank *= 2
    if max_rank not in [1 << i for i in range(20)]:
        options.append({"label": f"Rank {max_rank}", "value": str(max_rank)})
    return options


@app.callback(
    Output("primal-img", "src"),
    Output("dual-img", "src"),
    Output("svd-graph", "figure"),
    Output("analysis-info", "children"),
    Output("status-msg", "children"),
    Output("transport-store", "data"),
    Input("run-btn", "n_clicks"),
    State("scene-type", "value"),
    State("resolution", "value"),
    State("albedo", "value"),
    State("svd-rank", "value"),
    State("n-bounces", "value"),
    State("proj-x", "value"),
    State("cam-x", "value"),
    prevent_initial_call=True,
)
def run_simulation(
    n_clicks, scene_type, resolution_str, albedo_str, svd_rank_str,
    n_bounces_str, proj_x_str, cam_x_str,
):
    """Execute a dual photography simulation and update all visualizations.

    This is the main callback that:
    1. Creates a synthetic scene with the selected parameters
    2. Computes the transport matrix via ray-casting
    3. Generates primal (camera view) and dual (projector view) images
    4. Performs SVD analysis of the transport matrix
    5. Updates all UI components with results
    """
    if not n_clicks:
        raise PreventUpdate

    try:
        resolution = int(resolution_str)
        albedo = float(albedo_str)
        svd_rank = int(svd_rank_str)
        proj_x = float(proj_x_str)
        cam_x = float(cam_x_str)
        n_bounces = int(n_bounces_str)

        scene = SceneType(scene_type)
        proj_shape = (resolution, resolution)
        cam_shape = (resolution, resolution)

        renderer = VirtualRenderer(
            proj_shape=proj_shape,
            cam_shape=cam_shape,
            albedo=albedo,
            n_bounces=n_bounces,
        )

        rank = svd_rank if 0 < svd_rank < resolution * resolution else None
        proj_pos = np.array([proj_x, 0.5, 2.0])
        cam_pos = np.array([cam_x, 0.5, 2.0])

        result = renderer.run_simulation(
            scene_type=scene,
            proj_pos=proj_pos,
            cam_pos=cam_pos,
            svd_rank=rank,
        )

        # Build base64 images
        primal_src = _numpy_to_b64_img(result.primal_image)
        dual_src = _numpy_to_b64_img(result.dual_image)

        # SVD spectrum plot
        sv = result.analysis["singular_values"]
        cumulative = result.analysis["energy_cumulative"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(sv) + 1)),
            y=sv.tolist(),
            name="Singular Values",
            marker_color="#636EFA",
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=(cumulative * float(sv.max())).tolist(),
            name="Cumulative Energy (scaled)",
            line=dict(color="#EF553B", width=2),
            yaxis="y2",
        ))
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=40, t=20, b=40),
            height=280,
            legend=dict(orientation="h", y=-0.25),
            yaxis=dict(title="Singular Value"),
            yaxis2=dict(
                title="Cumulative Energy",
                overlaying="y",
                side="right",
                range=[0, float(sv.max()) * 1.05],
            ),
            xaxis=dict(title="Index"),
        )

        # Analysis summary
        cond = result.analysis["condition_number"]
        cond_str = f"{cond:.2f}" if cond < 1e10 else f"{cond:.2e}"
        analysis_children = [
            html.P([html.Strong("Matrix size: "),
                    f"{result.transport.T.shape[0]} x {result.transport.T.shape[1]}"]),
            html.P([html.Strong("Condition number: "), cond_str]),
            html.P([html.Strong("Rank for 90% energy: "),
                    str(result.analysis["effective_rank_90"])]),
            html.P([html.Strong("Rank for 99% energy: "),
                    str(result.analysis["effective_rank_99"])]),
            html.P([html.Strong("Total singular values: "), str(len(sv))]),
        ]
        if rank:
            analysis_children.append(
                dbc.Alert(f"Using SVD rank: {rank}", color="warning", className="mt-2 py-1")
            )

        status = dbc.Alert("Simulation complete!", color="success", className="py-1 mb-0")

        # Store transport matrix for relighting (as nested lists for JSON serialization)
        store_data = {
            "T": result.transport.T.tolist(),
            "proj_shape": list(proj_shape),
            "cam_shape": list(cam_shape),
        }

        return primal_src, dual_src, fig, analysis_children, status, store_data

    except Exception as e:
        error_msg = dbc.Alert(f"Error: {e}", color="danger", className="py-1 mb-0")
        raise PreventUpdate from e


@app.callback(
    Output("relight-pattern-img", "src"),
    Output("relight-result-img", "src"),
    Input("relight-btn", "n_clicks"),
    State("relight-pattern", "value"),
    State("transport-store", "data"),
    prevent_initial_call=True,
)
def run_relighting(n_clicks, pattern_name, store_data):
    """Relight the scene with a selected illumination pattern.

    Uses the stored transport matrix from the last simulation to compute
    what the camera would see under new illumination without re-capturing.
    """
    if not n_clicks or store_data is None:
        raise PreventUpdate

    T = np.array(store_data["T"])
    proj_shape = tuple(store_data["proj_shape"])
    cam_shape = tuple(store_data["cam_shape"])
    h, w = proj_shape

    pattern = _make_relight_pattern(pattern_name, h, w)

    tm = TransportMatrix(T=T, cam_shape=cam_shape, proj_shape=proj_shape)
    relighted = tm.forward(pattern)

    return _numpy_to_b64_img(pattern), _numpy_to_b64_img(relighted)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the Dash application."""
    print("=" * 60)
    print("  Dual Photography Lab")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
