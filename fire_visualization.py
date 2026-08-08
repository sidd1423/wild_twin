"""
Animated 3D visualization of the fire spread simulation over the
Silver Peak terrain, using PyVista.

Controls (interactive window):
    space  - pause / resume the animation
    r      - reset the simulation to the ignition point
    q      - quit

Run:
    python fire_visualize.py
"""

import time

import numpy as np
import pyvista as pv
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap

from fire_risk import get_fire_risk
from weather import weather_variables
from fire_sim import FireSimulation, UNBURNED, BURNING, BURNED

IGNITION_ROW = 419
IGNITION_COL = 393

STEPS_PER_TICK = 3        # simulation steps advanced per timer tick
TICK_DURATION_MS = 120    # real time between ticks
MAX_TICKS = 4000          # safety cap so the timer doesn't run forever

# Camera stays zoomed to a window this many meters wide/tall around the
# ignition point, so a handful of burning 2 m cells are actually visible
# instead of being lost in the full 2000 m x 2000 m terrain.
VIEW_HALF_WIDTH_M = 250

PRINT_EVERY_N_TICKS = 5   # console progress so you can confirm it IS running


# =========================
# LOAD DEM + BUILD TERRAIN
# =========================

DEM_FILE = "layers/silver_peak_dem.tif"

with rasterio.open(DEM_FILE) as src:
    elevation = src.read(1).astype(np.float32)
    dem_profile = src.profile

rows, cols = elevation.shape
resolution = abs(dem_profile["transform"][0])

x = np.arange(cols, dtype=np.float32) * resolution
y = np.arange(rows, dtype=np.float32) * resolution
xx, yy = np.meshgrid(x, y)

terrain = pv.StructuredGrid(
    xx.astype(np.float32),
    yy.astype(np.float32),
    elevation.astype(np.float32),
)


# =========================
# HELPERS
# =========================

def match_raster(source_file, reference_file, resampling=Resampling.bilinear):
    with rasterio.open(reference_file) as ref:
        ref_array = ref.read(1)
        destination = np.zeros(ref_array.shape, dtype=np.float32)

        with rasterio.open(source_file) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=resampling,
            )

    return destination


# =========================
# SLOPE + ASPECT
# =========================

dzdx = np.gradient(elevation, axis=1) / resolution
dzdy = np.gradient(elevation, axis=0) / resolution

slope = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360


# =========================
# FUEL TYPE
# =========================
# Land-cover codes are CATEGORICAL, so they must be resampled with
# nearest-neighbor, never bilinear (bilinear blends adjacent codes into
# non-integer values that don't match any class below, silently
# defaulting those cells to 0 fuel -- this was zeroing out roughly half
# the map near every class boundary).

fuel_raw = match_raster("layers/silver_peak_fuel.tif", DEM_FILE, resampling=Resampling.nearest)
fuel = np.zeros_like(fuel_raw, dtype=np.float32)

# Native codes present in this raster: 0, 1, 5, 10, 11, 12, 13.
# 0/1 look like non-fuel (rock/water) -- left at 0. Adjust the mapping
# below if you find out what your land-cover legend actually assigns
# to each code.
fuel[fuel_raw == 5] = 0.3
fuel[fuel_raw == 10] = 0.2
fuel[fuel_raw == 11] = 0.5
fuel[fuel_raw == 12] = 0.8
fuel[fuel_raw == 13] = 1.0


# =========================
# CANOPY / DENSITY / DRYNESS
# =========================

with rasterio.open("layers/silver_peak_canopy.tif") as canopy_src:
    canopy = canopy_src.read(1).astype(np.float32)
canopy_norm = np.clip(canopy / canopy.max(), 0, 1)

with rasterio.open("layers/silver_peak_density.tif") as density_src:
    density = density_src.read(1).astype(np.float32)
density = np.clip(density, 0, 1)

dryness = match_raster("layers/silver_peak_dryness.tif", DEM_FILE)
dryness = np.clip(dryness, 0, 1)


# =========================
# FIRE RISK + WEATHER
# =========================

risk = get_fire_risk(weather_variables, dryness, density, canopy_norm, slope, aspect, fuel)
temperature, humidity, wind_speed, wind_direction, rain = weather_variables()

print(
    f"Weather: {temperature}C, {humidity}% RH, "
    f"wind {wind_speed} km/h from {wind_direction}deg, rain {rain} mm"
)


# =========================
# FIRE SIMULATION
# =========================

def make_simulation():
    return FireSimulation(
        risk=risk,
        fuel=fuel,
        dryness=dryness,
        density=density,
        canopy=canopy_norm,
        slope=slope,
        aspect=aspect,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        seed_row=IGNITION_ROW,
        seed_col=IGNITION_COL,
    )


sim = make_simulation()
terrain["State"] = sim.state.astype(np.float32).ravel(order="F")


# =========================
# PYVISTA SETUP
# =========================

# 0 = unburned (green), 1 = burning (orange-red), 2 = burned (dark gray)
fire_cmap = ListedColormap(["#2e8b57", "#ff4500", "#333333"])

plotter = pv.Plotter()

actor = plotter.add_mesh(
    terrain,
    scalars="State",
    cmap=fire_cmap,
    clim=[UNBURNED, BURNED],
    show_scalar_bar=False,
    ambient=0.35,   # keep hue distinguishable in shadow without going fully
                     # flat -- fully unlit removed all relief shading, which
                     # is what made the mountains disappear
)

plotter.add_text(
    "Fire Spread Simulation\nspace: pause/resume   r: reset   q: quit",
    font_size=10,
)

# Zoom + tilt the camera to an oblique angle over the ignition point,
# instead of the default full-terrain view (too far to see a few 2 m
# burning cells) or a straight top-down view (which hides all relief
# since you're looking straight down the height axis).
ignition_x = IGNITION_COL * resolution
ignition_y = IGNITION_ROW * resolution
ignition_z = float(elevation[IGNITION_ROW, IGNITION_COL])

# Oblique viewing angle (45 deg azimuth, 35 deg elevation above the
# horizon) so terrain relief is still visible, instead of a top-down
# view that flattens all height cues away.
AZIMUTH_DEG = 225
ELEVATION_DEG = 35
CAMERA_DISTANCE_M = VIEW_HALF_WIDTH_M * 3.5

az = np.radians(AZIMUTH_DEG)
el = np.radians(ELEVATION_DEG)

camera_x = ignition_x + CAMERA_DISTANCE_M * np.cos(el) * np.cos(az)
camera_y = ignition_y + CAMERA_DISTANCE_M * np.cos(el) * np.sin(az)
camera_z = ignition_z + CAMERA_DISTANCE_M * np.sin(el)

plotter.camera_position = [
    (camera_x, camera_y, camera_z),
    (ignition_x, ignition_y, ignition_z),
    (0, 0, 1),
]
plotter.camera.view_angle = 40

state = {"running": True, "tick": 0}


def advance():
    if not state["running"]:
        return

    for _ in range(STEPS_PER_TICK):
        sim.step()

    terrain["State"] = sim.state.astype(np.float32).ravel(order="F")
    state["tick"] += 1

    if state["tick"] <= 3 or state["tick"] % PRINT_EVERY_N_TICKS == 0:
        burning = int((sim.state == BURNING).sum())
        burned = int((sim.state == BURNED).sum())
        print(f"tick {state['tick']}: burning={burning} burned={burned}", flush=True)

    plotter.render()


def toggle_pause():
    state["running"] = not state["running"]


def reset():
    global sim
    sim = make_simulation()
    terrain["State"] = sim.state.astype(np.float32).ravel(order="F")
    state["tick"] = 0
    plotter.render()


plotter.add_key_event("space", toggle_pause)
plotter.add_key_event("r", reset)

# Manual render loop instead of add_timer_event(): VTK's built-in
# repeating timer can silently stop firing on some window
# managers/backends (especially once the window loses focus), which
# looks exactly like "the sim froze" even though nothing in fire_sim.py
# is wrong. Driving the loop ourselves with interactive_update=True is
# more reliable.
print("[checkpoint] about to call plotter.show()...", flush=True)
plotter.show(interactive_update=True, auto_close=False)
print("[checkpoint] show() returned control -- entering animation loop", flush=True)

tick_count = 0
while not plotter._closed:
    advance()
    plotter.update()
    time.sleep(TICK_DURATION_MS / 1000.0)
    tick_count += 1
    if tick_count >= MAX_TICKS:
        break

plotter.close()