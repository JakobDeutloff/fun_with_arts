# %%
import pyarts
import numpy as np
import xarray as xr
import FluxSimulator as fsm
import matplotlib.pyplot as plt

# %% import example atmosphere
atmosphere = xr.open_dataset("atms.nc")

# %% convert xarray to ARTS gridded field
atms_grd = pyarts.arts.ArrayOfGriddedField4()
for lat in atmosphere.lat:
    for lon in atmosphere.lon:
        profile = atmosphere.sel(lat=lat, lon=lon)
        profile_grd = fsm.generate_gridded_field_from_profiles(
            profile["pressure"].values,
            profile["temperature"].values,
            gases={
                "H2O": profile["H2O"],
                "CO2": profile["CO2"],
                "O3": profile["O3"],
                "N2": profile["N2"],
                "O2": profile["O2"],
            },
        )
        atms_grd.append(profile_grd)

# %% setup flux simulator
gases = ["H2O", "CO2", "O2", "N2", "O3"]  # gases to include in the simulation
f_grid = np.linspace(1, 3e3, 200)  # frequency grid in cm^-1
f_grid_freq = pyarts.arts.convert.kaycm2freq(f_grid)  # frequency grid in Hz
surface_reflectivity_lw = 0.05  # surface reflectivity
setup_name = "test"  # name of the simulation
species = [  # species to include in the simulation
    "H2O, H2O-SelfContCKDMT350, H2O-ForeignContCKDMT350",
    "O2-*-1e12-1e99,O2-CIAfunCKDMT100",
    "N2, N2-CIAfunCKDMT252, N2-CIArotCKDMT252",
    "CO2, CO2-CKDMT252",
    "O3",
    "O3-XFIT",
]

LW_flux_simulator = fsm.FluxSimulator(setup_name)
LW_flux_simulator.ws.f_grid = f_grid_freq
LW_flux_simulator.set_species(species)


# %% generate lookup table if it doesn't exist for the given setup name
LW_flux_simulator.get_lookuptableBatch(atms_grd)


# %% initialize flux datasets
fluxes_spectral = xr.Dataset(
    {
        "flux_upward": (
            ("lat", "lon", "pressure", "f_grid"),
            np.zeros(
                (
                    len(atmosphere.lat),
                    len(atmosphere.lon),
                    len(atmosphere.pressure),
                    len(f_grid),
                )
            ),
        ),
        "flux_downward": (
            ("lat", "lon", "pressure", "f_grid"),
            np.zeros(
                (
                    len(atmosphere.lat),
                    len(atmosphere.lon),
                    len(atmosphere.pressure),
                    len(f_grid),
                )
            ),
        ),
    },
    coords={"lat": atmosphere.lat, "lon": atmosphere.lon, "f_grid": f_grid},
)

fluxes_integrated = xr.Dataset(
    {
        "flux_upward": (
            ("lat", "lon", "pressure"),
            np.zeros(
                (len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure))
            ),
        ),
        "flux_downward": (
            ("lat", "lon", "pressure"),
            np.zeros(
                (len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure))
            ),
        ),
        "heating_rate": (
            ("lat", "lon", "pressure"),
            np.zeros(
                (len(atmosphere.lat), len(atmosphere.lon), len(atmosphere.pressure))
            ),
        ),
    },
    coords={"lat": atmosphere.lat, "lon": atmosphere.lon},
)

# %% calculate fluxes
for i, lat in enumerate(atmosphere.lat):
    for j, lon in enumerate(atmosphere.lon):
        idx = i * len(atmosphere.lat) + j
        print(idx)
        results_lw = LW_flux_simulator.flux_simulator_single_profile(
            atms_grd[idx],
            atmosphere.sel(lat=lat, lon=lon).isel(pressure=0)["temperature"].values,
            np.max(
                [
                    -318,
                    atmosphere.sel(lat=lat, lon=lon)
                    .isel(pressure=0)["geometric_height"]
                    .values,
                ]
            ),
            surface_reflectivity_lw,
            geographical_position=[lat, lon],
        )

        fluxes_spectral["flux_upward"].loc[lat, lon] = results_lw[
            "spectral_flux_clearsky_up"
        ].T
        fluxes_spectral["flux_downward"].loc[lat, lon] = results_lw[
            "spectral_flux_clearsky_down"
        ].T
        fluxes_integrated["flux_upward"].loc[lat, lon] = results_lw["flux_clearsky_up"]
        fluxes_integrated["flux_downward"].loc[lat, lon] = results_lw[
            "flux_clearsky_down"
        ]
        fluxes_integrated["heating_rate"].loc[lat, lon] = results_lw[
            "heating_rate_clearsky"
        ].T


# %% make simple control plots which rely on xarray plotting
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fluxes_spectral.isel(lat=0, lon=0)["flux_upward"].plot(ax=ax[0], x="f_grid")
fluxes_spectral.isel(lat=0, lon=0)["flux_downward"].plot(ax=ax[1], x="f_grid")

# %%
