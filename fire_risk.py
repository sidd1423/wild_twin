import numpy as np
import rasterio


# =========================
# SETTINGS
# =========================
UPDATE_FIRE_RISK = True
RISK_FILE = "layers/silver_peak_fire_risk.tif"


# =========================
# FIRE RISK CALCULATION
# =========================
def wind_aspect_score(wind_direction, aspect):
    difference = np.abs(wind_direction - aspect)
    difference = np.minimum(difference, 360 - difference)
    score = np.cos(np.radians(difference))

    return (score + 1) / 2


def stability_factor(temperature, humidity, wind_speed):

    temp = np.clip((temperature - 15) / 20, 0, 1)
    humidity = np.clip(1 - humidity / 100, 0, 1)
    wind = np.clip(wind_speed / 30, 0, 1)

    stability = (
        0.5 * temp +
        0.3 * humidity +
        0.2 * wind
    )

    return np.clip(stability, 0, 1)


def create_fire_risk_map(temperature, humidity, wind_speed, wind_direction, rain, dryness, density, canopy, slope, aspect, fuel):
    temp_score = np.clip((temperature - 10) / 30, 0, 1)
    humidity_score = np.clip(1 - humidity / 100, 0, 1)
    wind_score = np.clip(wind_speed / 50, 0, 1)
    rain_score = np.clip(1 - rain / 10, 0, 1)
    slope_score = np.clip(slope / 45, 0, 1)
    wind_alignment_score = wind_aspect_score(wind_direction, aspect)
    stability_score = stability_factor(temperature, humidity, wind_speed)

    weather_score = 0.35 * temp_score + 0.30 * humidity_score + 0.20 * wind_score + 0.15 * rain_score
    weather_score *= 0.7 + 0.6 * stability_score
    fuel_score = 0.35 * dryness + 0.25 * density + 0.20 * canopy + 0.20 * fuel
    terrain_score = 0.5 * slope_score + 0.5 * wind_alignment_score

    risk = 0.45 * fuel_score + 0.35 * weather_score + 0.20 * terrain_score

    return np.clip(risk, 0, 1)


# =========================
# SAVE RASTER
# =========================
def save_risk_map(risk, template_file):
    with rasterio.open(template_file) as src:
        profile = src.profile.copy()

    profile.update(dtype="float32", count=1, compress="lzw")

    with rasterio.open(RISK_FILE, "w", **profile) as dst:
        dst.write(risk.astype(np.float32), 1)


# =========================
# LOAD OR UPDATE
# =========================
def get_fire_risk(weather, dryness, density, canopy, slope, aspect, fuel):
    if UPDATE_FIRE_RISK:
        temperature, humidity, wind_speed, wind_direction, rain = weather()

        risk = create_fire_risk_map(temperature, humidity, wind_speed, wind_direction, rain, dryness, density, canopy, slope, aspect, fuel)
        save_risk_map(risk, "layers/silver_peak_dem.tif")

    else:
        with rasterio.open(RISK_FILE) as src:
            risk = src.read(1)

    return risk
