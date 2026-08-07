import pyvista as pv
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt

from fire_risk import get_fire_risk
from weather import weather_variables
from fire_sim import FireSimulation

IGNITION_ROW = 500
IGNITION_COL = 500


# =========================
# LOAD DEM
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
    elevation.astype(np.float32)
)
terrain.texture_map_to_plane(inplace=True)


# =========================
# HELPERS
# =========================

def raster_to_texture(array, cmap):
    array = np.nan_to_num(array)
    norm = (array - array.min()) / (array.max() - array.min() + 1e-9)
    img = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    img = np.flipud(img)

    return pv.numpy_to_texture(img)


def match_raster(source_file, reference_file):
    with rasterio.open(reference_file) as ref:
        ref_array = ref.read(1)
        destination = np.zeros(ref_array.shape, dtype=np.float32)

        with rasterio.open(source_file) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear
            )

    return destination


# =========================
# SATELLITE
# =========================

with rasterio.open("layers/silver_peak_texture.tif") as src:
    sat = src.read([1, 2, 3]).astype(np.float32)

for i in range(3):
    band = sat[i]
    sat[i] = (band - band.min()) / (band.max() - band.min() + 1e-9)

sat = np.transpose(sat, (1, 2, 0))
sat = (sat * 255).astype(np.uint8)
sat = np.flipud(sat)
sat_texture = pv.numpy_to_texture(sat)


# =========================
# SLOPE + ASPECT
# =========================

dzdx = np.gradient(elevation, axis=1) / resolution
dzdy = np.gradient(elevation, axis=0) / resolution

slope = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360


# =========================
# FUEL TYPE
# =========================

fuel_raw = match_raster("layers/silver_peak_fuel.tif", "layers/silver_peak_dem.tif")
fuel = np.zeros_like(fuel_raw, dtype=np.float32)

fuel[fuel_raw == 10] = 0.2
fuel[fuel_raw == 11] = 0.5
fuel[fuel_raw == 12] = 0.8
fuel[fuel_raw == 13] = 1.0


# =========================
# CANOPY
# =========================

with rasterio.open("layers/silver_peak_canopy.tif") as src:
    canopy = src.read(1).astype(np.float32)

canopy_norm = np.clip(canopy / canopy.max(), 0, 1)


# =========================
# DENSITY
# =========================

with rasterio.open("layers/silver_peak_density.tif") as src:
    density = src.read(1).astype(np.float32)

density = np.clip(density, 0, 1)


# =========================
# DRYNESS
# =========================

dryness = match_raster("layers/silver_peak_dryness.tif", "layers/silver_peak_dem.tif")
dryness = np.clip(dryness, 0, 1)


# =========================
# FIRE RISK GENERATION
# =========================

risk = get_fire_risk(weather_variables, dryness, density, canopy_norm, slope, aspect, fuel)
risk_texture = raster_to_texture(risk, plt.cm.RdYlGn_r)


# =========================
# CREATE TEXTURES
# =========================

textures = {
    "satellite":
        sat_texture,
    "slope":
        raster_to_texture(
            slope,
            plt.cm.inferno
        ),
    "aspect":
        raster_to_texture(
            aspect,
            plt.cm.hsv
        ),
    "fuel":
        raster_to_texture(
            fuel,
            plt.cm.YlOrRd
        ),
    "canopy":
        raster_to_texture(
            canopy_norm,
            plt.cm.viridis
        ),
    "density":
        raster_to_texture(
            density,
            plt.cm.Greens
        ),
    "dryness":
        raster_to_texture(
            dryness,
            plt.cm.Reds
        ),
    "risk":
        risk_texture
}


# =========================
# SCALAR DATA
# =========================

terrain["Slope"] = slope.ravel(order="F")
terrain["Aspect"] = aspect.ravel(order="F")
terrain["Fuel"] = fuel.ravel(order="F")
terrain["Canopy"] = canopy_norm.ravel(order="F")
terrain["Density"] = density.ravel(order="F")
terrain["Dryness"] = dryness.ravel(order="F")
terrain["Risk"] = risk.ravel(order="F")


# =========================
# PYVISTA
# =========================

plotter = pv.Plotter()

mesh = terrain.copy()
current_actor = None

layers = {
    "slope": {
        "scalars": "Slope",
        "cmap": "inferno",
        "clim": [0, 90],
        "title": "Slope (degrees)"
    },
    "aspect": {
        "scalars": "Aspect",
        "cmap": "hsv",
        "clim": [0, 360],
        "title": "Aspect (degrees)"
    },
    "fuel": {
        "scalars": "Fuel",
        "cmap": "YlOrRd",
        "clim": [0, 1],
        "title": "Fuel Hazard"
    },
    "canopy": {
        "scalars": "Canopy",
        "cmap": "viridis",
        "clim": [0, 1],
        "title": "Canopy Height"
    },
    "density": {
        "scalars": "Density",
        "cmap": "Greens",
        "clim": [0, 1],
        "title": "Fuel Density"
    },
    "dryness": {
        "scalars": "Dryness",
        "cmap": "Reds",
        "clim": [0, 1],
        "title": "Dryness"
    },
    "risk": {
        "scalars": "Risk",
        "cmap": "RdYlGn_r",
        "clim": [0, 1],
        "title": "Fire Risk"
    }
}


def show_layer(name):
    global current_actor

    if current_actor is not None:
        plotter.remove_actor(current_actor)

    if len(plotter.scalar_bars) > 0:
        plotter.remove_scalar_bar()

    if name == "satellite":
        current_actor = plotter.add_mesh(terrain.copy(), texture=sat_texture)

    else:
        data = layers[name]
        current_actor = plotter.add_mesh(
            terrain.copy(),
            scalars=data["scalars"],
            cmap=data["cmap"],
            clim=data["clim"],
            scalar_bar_args={
                "title": data["title"],
                "vertical": True,
                "width": 0.08,
                "height": 0.7
            }
        )

    plotter.render()


keys = {"1": "satellite", "2": "slope", "3": "aspect", "4": "fuel", "5": "canopy", "6": "density", "7": "dryness", "8": "risk"}

for key, layer in keys.items():
    plotter.add_key_event(key, lambda l=layer: show_layer(l))

plotter.add_text(
    "1 Satellite \n2 Slope \n3 Aspect \n4 Fuel \n5 Canopy Height \n6 Density \n7 Dryness \n8 Fire Risk",
    font_size=12
)
show_layer("satellite")

plotter.show()
