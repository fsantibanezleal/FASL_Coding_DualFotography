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
import time
import traceback
from datetime import datetime

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


def _log_entry(message: str) -> str:
    """Create a timestamped log entry string."""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"


def _render_log(entries: list) -> list:
    """Render log entries as html.P children for the activity-log div."""
    if not entries:
        return [html.P("Ready.", className="text-muted small mb-0")]
    return [
        html.P(entry, className="small mb-0", style={"color": "#adb5bd"})
        for entry in entries
    ]


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

    Uses percentile-based normalization to handle outliers and low-contrast
    images robustly. The 1st and 99th percentiles define the display range,
    preventing a few bright pixels from washing out the rest.

    Args:
        arr: 2D array with values in [0, max].

    Returns:
        Base64-encoded PNG data URI string suitable for html.Img src attribute.
    """
    arr = arr.astype(np.float64)
    # Percentile-based normalization: robust to outliers
    vmin = float(np.percentile(arr, 1))
    vmax = float(np.percentile(arr, 99))
    if vmax - vmin < 1e-10:
        vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-10:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.full_like(arr, 0.5)  # Constant -> mid-gray, not black
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
# Help / usage content (condensed from docs/user_guide.md)
# ---------------------------------------------------------------------------
HELP_MARKDOWN = """
### Quick Start

1. Pick a **Scene Type** (Box + Wall is the classic demo).
2. Choose a **Resolution** (32x32 is a good balance).
3. Click **Run Simulation**.
4. Once done, pick a **Relighting Pattern** and click **Relight**.

### Controls

| Control | What it does |
|---|---|
| **Scene Type** | 3D scene geometry. Box + Wall shows occlusion well. |
| **Resolution** | N x N pixels. Higher = more detail, slower. |
| **Surface Albedo** | Reflectance; 0.1 dark, 1.0 white. |
| **SVD Rank** | Truncation rank for the dual image. Full = no compression. |
| **Light Bounces** | 0 = direct only; >=1 enables indirect light. |
| **Projector/Camera X** | Horizontal positions. Bigger gap = more parallax. |

### What you are seeing

- **Primal image**: camera view under uniform projector illumination, `c = T.1`.
- **Dual image**: projector view via Helmholtz reciprocity, `p = T^T.1`.
- **SVD spectrum**: singular values of T; a steep drop means low-rank transport.
- **Relighting**: reuses the captured T to synthesize novel illuminations with no re-capture.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `?` or `/` | Open this Help dialog |
| `Esc` | Close the Help dialog |
| `r` | Focus Run Simulation button |
| `l` | Focus Relight button |

Full documentation lives in `docs/user_guide.md` in the repo.
"""


# Clientside JS: bind keyboard shortcuts that set a hidden dcc.Input value.
# Dash then reacts via a normal callback on that Input. Kept tiny on purpose.
KEYBOARD_JS = """
if (!window.__dp_kbd_bound) {
  window.__dp_kbd_bound = true;
  document.addEventListener('keydown', function(e) {
    var target = e.target || {};
    var tag = (target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    var el = document.getElementById('kbd-shim');
    if (!el) return;
    if (e.key === '?' || e.key === '/') {
      el.value = 'help:' + Date.now();
      el.dispatchEvent(new Event('input', { bubbles: true }));
      e.preventDefault();
    } else if (e.key === 'Escape') {
      el.value = 'close:' + Date.now();
      el.dispatchEvent(new Event('input', { bubbles: true }));
    } else if (e.key === 'r' || e.key === 'R') {
      var rb = document.getElementById('run-btn'); if (rb) rb.focus();
    } else if (e.key === 'l' || e.key === 'L') {
      var lb = document.getElementById('relight-btn'); if (lb) lb.focus();
    }
  });
}
"""


# ---------------------------------------------------------------------------
# Layout helpers for new components (Help modal, status bar, tooltips)
# ---------------------------------------------------------------------------
def _help_modal() -> dbc.Modal:
    """Build the Help modal displayed on `?` key or Help button click."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Dual Photography Lab - Help")),
            dbc.ModalBody(dcc.Markdown(HELP_MARKDOWN)),
            dbc.ModalFooter(
                dbc.Button("Close", id="help-close-btn", className="ms-auto", n_clicks=0)
            ),
        ],
        id="help-modal",
        is_open=False,
        size="lg",
        scrollable=True,
    )


def _status_bar() -> dbc.Card:
    """Build the run-status bar shown above the controls.

    Reflects the last completed simulation: scene, resolution, bounces, and
    elapsed seconds. Driven by `status-store` written from `run_simulation`.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span("Status: ", className="text-muted small"),
                        html.Span(
                            "Idle - no simulation run yet.",
                            id="status-bar-text",
                            className="small",
                        ),
                    ],
                    className="d-flex align-items-center justify-content-between",
                )
            ],
            className="py-2",
        ),
        className="mb-3",
        style={"backgroundColor": "rgba(0,0,0,0.25)"},
    )


# Controls that receive a tooltip. Order drives generation in _tooltips().
TOOLTIP_TARGETS: list[tuple[str, str]] = [
    ("scene-type", "3D scene geometry. Box + Wall is the canonical demo."),
    ("resolution", "Image resolution (N x N). Higher is slower but crisper."),
    ("albedo", "Surface reflectance. 0.1 is dark, 1.0 is white."),
    ("svd-rank", "SVD truncation rank for the dual image. 'Full' = no truncation."),
    ("n-bounces", "Number of indirect light bounces. 0 disables inter-reflections."),
    ("proj-x", "Projector X position. Larger spread = more parallax."),
    ("cam-x", "Camera X position. Larger spread = more parallax."),
    ("run-btn", "Run a full dual photography simulation."),
    ("relight-pattern", "Illumination pattern for relighting."),
    ("relight-btn", "Relight the captured scene with the chosen pattern."),
    ("help-open-btn", "Open the Help dialog (or press '?')."),
]


def _tooltips() -> list:
    """Build `dbc.Tooltip`s bound to every interactive control."""
    return [
        dbc.Tooltip(text, target=target_id, placement="right")
        for target_id, text in TOOLTIP_TARGETS
    ]


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
            width=10,
        ),
        dbc.Col(
            dbc.Button(
                "Help (?)",
                id="help-open-btn",
                color="secondary",
                outline=True,
                className="float-end my-3",
                n_clicks=0,
            ),
            width=2,
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
    # Status bar (last simulation summary)
    dbc.Row([dbc.Col(_status_bar(), width=12)]),
    dbc.Row([
        dbc.Col(_control_panel(), md=3),
        dbc.Col(_results_panel(), md=9),
    ]),
    # Activity log panel
    dbc.Row([
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H5("Activity Log", className="text-center mb-2"),
                html.Div(
                    id="activity-log",
                    children=[html.P("Ready.", className="text-muted small mb-0")],
                    style={
                        "maxHeight": "160px",
                        "overflowY": "auto",
                        "fontFamily": "monospace",
                        "fontSize": "0.8rem",
                        "backgroundColor": "rgba(0,0,0,0.3)",
                        "padding": "8px",
                        "borderRadius": "4px",
                    },
                ),
            ])),
        ], width=12),
    ], className="mb-3"),
    # Help modal + tooltips on all controls
    _help_modal(),
    *_tooltips(),
    # Hidden keyboard shim: clientside JS writes a timestamped value here
    # (e.g. "help:172..." or "close:172...") that a callback observes.
    dcc.Input(id="kbd-shim", type="hidden", value=""),
    # Hidden stores
    dcc.Store(id="transport-store", data=None),
    dcc.Store(id="log-store", data=[]),
    dcc.Store(id="status-store", data={
        "scene": None, "resolution": None, "bounces": None, "elapsed_s": None,
    }),
], fluid=True, className="py-3")


# Install keyboard-shim JS once at load time.
app.clientside_callback(
    f"function(trigger) {{ {KEYBOARD_JS} return window.dash_clientside.no_update; }}",
    Output("kbd-shim", "style"),
    Input("kbd-shim", "id"),
)


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
    Output("log-store", "data", allow_duplicate=True),
    Output("activity-log", "children", allow_duplicate=True),
    Output("status-store", "data"),
    Input("run-btn", "n_clicks"),
    State("scene-type", "value"),
    State("resolution", "value"),
    State("albedo", "value"),
    State("svd-rank", "value"),
    State("n-bounces", "value"),
    State("proj-x", "value"),
    State("cam-x", "value"),
    State("log-store", "data"),
    prevent_initial_call=True,
)
def run_simulation(
    n_clicks, scene_type, resolution_str, albedo_str, svd_rank_str,
    n_bounces_str, proj_x_str, cam_x_str, log_data,
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

    log_entries = list(log_data or [])
    # Clear and show processing
    log_entries.append(_log_entry("Processing..."))

    t_start = time.perf_counter()
    try:
        resolution = int(resolution_str)
        albedo = float(albedo_str)
        svd_rank = int(svd_rank_str)
        proj_x = float(proj_x_str)
        cam_x = float(cam_x_str)
        n_bounces = int(n_bounces_str)

        log_entries.append(_log_entry(
            f"Simulation started: scene={scene_type}, res={resolution}, "
            f"albedo={albedo}, bounces={n_bounces}"
        ))

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

        # Extended analysis: sparsity, reciprocity, frequency
        sparsity = result.transport.sparsity()
        recip_err = result.transport.reciprocity_error()
        freq = result.transport.frequency_analysis()

        cond = result.analysis["condition_number"]
        cond_str = f"{cond:.2f}" if cond < 1e10 else f"{cond:.2e}"
        analysis_children = [
            html.P([html.Strong("Matrix: "),
                    f"{result.transport.T.shape[0]}x{result.transport.T.shape[1]}"]),
            html.P([html.Strong("Condition: "), cond_str]),
            html.P([html.Strong("Rank 90%/99%: "),
                    f"{result.analysis['effective_rank_90']} / {result.analysis['effective_rank_99']}"]),
            html.P([html.Strong("Sparsity: "),
                    f"{sparsity['nnz_fraction']*100:.1f}% non-zero"]),
            html.P([html.Strong("Reciprocity error: "),
                    f"{recip_err:.4f}",
                    html.Span(" (0=symmetric)", className="text-muted small")]),
            html.P([html.Strong("Freq. DC/HF: "),
                    f"{freq['dc_fraction']*100:.0f}% / {freq['high_freq_fraction']*100:.0f}%"]),
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

        elapsed_s = time.perf_counter() - t_start
        log_entries.append(_log_entry(
            f"Transport matrix computed: {result.transport.T.shape[0]}x"
            f"{result.transport.T.shape[1]}, cond={cond_str}"
        ))
        log_entries.append(_log_entry(f"Simulation complete ({elapsed_s:.2f}s)"))

        status_data = {
            "scene": scene_type,
            "resolution": resolution,
            "bounces": n_bounces,
            "elapsed_s": round(elapsed_s, 2),
        }

        return (
            primal_src, dual_src, fig, analysis_children, status, store_data,
            log_entries, _render_log(log_entries), status_data,
        )

    except Exception as e:
        log_entries.append(_log_entry(f"Error: {e}"))
        error_msg = dbc.Alert(f"Error: {e}", color="danger", className="py-1 mb-0")
        raise PreventUpdate from e


@app.callback(
    Output("relight-pattern-img", "src"),
    Output("relight-result-img", "src"),
    Output("log-store", "data", allow_duplicate=True),
    Output("activity-log", "children", allow_duplicate=True),
    Input("relight-btn", "n_clicks"),
    State("relight-pattern", "value"),
    State("transport-store", "data"),
    State("log-store", "data"),
    prevent_initial_call=True,
)
def run_relighting(n_clicks, pattern_name, store_data, log_data):
    """Relight the scene with a selected illumination pattern.

    Uses the stored transport matrix from the last simulation to compute
    what the camera would see under new illumination without re-capturing.
    """
    if not n_clicks or store_data is None:
        raise PreventUpdate

    log_entries = list(log_data or [])
    log_entries.append(_log_entry(f"Relighting: pattern={pattern_name}"))

    T = np.array(store_data["T"])
    proj_shape = tuple(store_data["proj_shape"])
    cam_shape = tuple(store_data["cam_shape"])
    h, w = proj_shape

    pattern = _make_relight_pattern(pattern_name, h, w)

    tm = TransportMatrix(T=T, cam_shape=cam_shape, proj_shape=proj_shape)
    relighted = tm.forward(pattern)

    log_entries.append(_log_entry(f"Relighting complete: {pattern_name}"))

    return (
        _numpy_to_b64_img(pattern), _numpy_to_b64_img(relighted),
        log_entries, _render_log(log_entries),
    )


@app.callback(
    Output("help-modal", "is_open"),
    Input("help-open-btn", "n_clicks"),
    Input("help-close-btn", "n_clicks"),
    Input("kbd-shim", "value"),
    State("help-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_help_modal(_open_clicks, _close_clicks, kbd_value, is_open):
    """Open/close the Help modal on button clicks or `?`/`Esc` keys."""
    trigger = ctx.triggered_id
    if trigger == "help-open-btn":
        return True
    if trigger == "help-close-btn":
        return False
    if trigger == "kbd-shim" and isinstance(kbd_value, str):
        if kbd_value.startswith("help:"):
            return True
        if kbd_value.startswith("close:"):
            return False
    raise PreventUpdate


@app.callback(
    Output("status-bar-text", "children"),
    Input("status-store", "data"),
)
def update_status_bar(status_data):
    """Render the status bar from the last-run status-store payload."""
    if not status_data or status_data.get("scene") is None:
        return "Idle - no simulation run yet."
    scene = status_data.get("scene")
    res = status_data.get("resolution")
    bounces = status_data.get("bounces")
    elapsed = status_data.get("elapsed_s")
    return (
        f"Last run: scene={scene}, resolution={res}x{res}, "
        f"bounces={bounces}, elapsed={elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    """Entry point for the Dash application."""
    print("=" * 60)
    print("  Dual Photography Lab")
    print("  Open http://127.0.0.1:8004 in your browser")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=8004)


if __name__ == "__main__":
    main()
