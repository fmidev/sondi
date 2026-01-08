"""Parser for FMI WFS sounding data (multipointcoverage format).

Example URL:
https://opendata.fmi.fi/wfs?request=getFeature&storedquery_id=fmi::observations::weather::sounding::multipointcoverage&fmisid=101104&starttime=2025-08-24T12:00:00Z&endtime=2025-08-24T12:00:00Z
"""

import re
from io import BytesIO
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import xarray as xr
import requests


# XML namespaces used in FMI WFS responses
NAMESPACES = {
    'wfs': 'http://www.opengis.net/wfs/2.0',
    'gml': 'http://www.opengis.net/gml/3.2',
    'om': 'http://www.opengis.net/om/2.0',
    'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0',
    'gmlcov': 'http://www.opengis.net/gmlcov/1.0',
    'swe': 'http://www.opengis.net/swe/2.0',
    'sams': 'http://www.opengis.net/samplingSpatial/2.0',
    'xlink': 'http://www.w3.org/1999/xlink',
}

# Parameter names in the sounding data (order matters - matches data columns)
SOUNDING_PARAMS = [
    'pressure',       # PAP_PT1S_AVG - Atmospheric pressure [hPa]
    'wind_speed',     # WSP_PT1S_AVG - Wind speed [m/s]
    'wind_dir',       # WDP_PT1S_AVG - Wind direction [degrees]
    'temperature',    # TAP_PT1S_AVG - Air temperature [°C]
    'dewpoint',       # TDP_PT1S_AVG - Dew point temperature [°C]
    'humidity',       # RHP_PT1S_AVG - Relative humidity [%]
    'ascent_rate',    # UAP_PT1S_AVG - Ascent rate [m/s]
]


def fetch_sounding(url: str) -> xr.Dataset:
    """Fetch and parse sounding data from FMI WFS service.

    Args:
        url: Full WFS request URL for sounding data.

    Returns:
        xarray Dataset with sounding profile data, indexed by altitude.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return parse_sounding_xml(response.content)


def parse_sounding_xml(xml_content: bytes) -> xr.Dataset:
    """Parse FMI WFS sounding XML response into xarray Dataset.

    Args:
        xml_content: Raw XML bytes from WFS response.

    Returns:
        xarray Dataset with variables:
        - pressure (hPa)
        - temperature (°C)
        - dewpoint (°C)
        - humidity (%)
        - wind_speed (m/s)
        - wind_dir (degrees)
        - mixing_ratio (g/kg)
        - latitude (degrees)
        - longitude (degrees)

        Coordinates:
        - altitude (m above sea level)
        - time (datetime64)
    """
    root = ET.fromstring(xml_content)

    # Extract positions: lat, lon, altitude, unix_timestamp
    positions_elem = root.find('.//gmlcov:positions', NAMESPACES)
    if positions_elem is None or positions_elem.text is None:
        raise ValueError("No position data found in sounding XML")

    positions_text = positions_elem.text.strip()
    positions_values = [float(x) for x in positions_text.split()]

    # Reshape: each position has 4 values (lat, lon, alt, time)
    n_points = len(positions_values) // 4
    positions = np.array(positions_values).reshape(n_points, 4)

    latitudes = positions[:, 0]
    longitudes = positions[:, 1]
    altitudes = positions[:, 2]
    timestamps = positions[:, 3].astype('int64')

    # Extract measurement values
    values_elem = root.find('.//gml:doubleOrNilReasonTupleList', NAMESPACES)
    if values_elem is None or values_elem.text is None:
        raise ValueError("No measurement data found in sounding XML")

    values_text = values_elem.text.strip()
    values_list = [float(x) for x in values_text.split()]

    # Each observation has 7 parameters
    n_params = len(SOUNDING_PARAMS)
    # Trim any trailing values that don't fit complete records
    expected_count = n_points * n_params
    values_list = values_list[:expected_count]
    values = np.array(values_list).reshape(n_points, n_params)

    # Convert unix timestamps to datetime
    times = pd.to_datetime(timestamps, unit='s', utc=True)

    # Build Dataset
    ds = xr.Dataset(
        data_vars={
            SOUNDING_PARAMS[i]: ('altitude', values[:, i])
            for i in range(n_params)
        },
        coords={
            'altitude': altitudes,
            'time': ('altitude', times),
            'latitude': ('altitude', latitudes),
            'longitude': ('altitude', longitudes),
        },
    )

    # Add metadata
    ds['altitude'].attrs = {'units': 'm', 'long_name': 'Altitude above sea level'}
    ds['pressure'].attrs = {'units': 'hPa', 'long_name': 'Atmospheric pressure'}
    ds['temperature'].attrs = {'units': '°C', 'long_name': 'Air temperature'}
    ds['dewpoint'].attrs = {'units': '°C', 'long_name': 'Dew point temperature'}
    ds['humidity'].attrs = {'units': '%', 'long_name': 'Relative humidity'}
    ds['wind_speed'].attrs = {'units': 'm/s', 'long_name': 'Wind speed'}
    ds['wind_dir'].attrs = {'units': 'degrees', 'long_name': 'Wind direction'}
    ds['ascent_rate'].attrs = {'units': 'm/s', 'long_name': 'Balloon ascent rate'}

    return ds


def find_freezing_level(ds: xr.Dataset) -> dict:
    """Find the altitude where temperature crosses 0°C.

    Args:
        ds: Sounding dataset with temperature and altitude.

    Returns:
        Dictionary with:
        - freezing_level_m: Interpolated altitude of 0°C [m]
        - pressure_hPa: Pressure at freezing level [hPa]
        - method: Interpolation method used
    """
    temp = ds['temperature'].values
    alt = ds['altitude'].values
    pres = ds['pressure'].values

    # Find where temperature crosses from positive to negative
    # (ascending through atmosphere, temperature generally decreases)
    for i in range(len(temp) - 1):
        if temp[i] >= 0 and temp[i + 1] < 0:
            # Linear interpolation to find exact crossing
            t1, t2 = temp[i], temp[i + 1]
            a1, a2 = alt[i], alt[i + 1]
            p1, p2 = pres[i], pres[i + 1]

            # Interpolate altitude where T = 0
            frac = (0 - t1) / (t2 - t1)
            freezing_alt = a1 + frac * (a2 - a1)
            freezing_pres = p1 + frac * (p2 - p1)

            return {
                'freezing_level_m': round(freezing_alt, 1),
                'pressure_hPa': round(freezing_pres, 1),
                'lower_point': {'altitude_m': a1, 'temperature_C': t1, 'pressure_hPa': p1},
                'upper_point': {'altitude_m': a2, 'temperature_C': t2, 'pressure_hPa': p2},
                'method': 'linear_interpolation',
            }

    raise ValueError("No freezing level found in sounding data")


if __name__ == '__main__':
    # Example usage
    url = (
        "https://opendata.fmi.fi/wfs?request=getFeature"
        "&storedquery_id=fmi::observations::weather::sounding::multipointcoverage"
        "&fmisid=101104&starttime=2025-08-24T12:00:00Z&endtime=2025-08-24T12:00:00Z"
    )

    ds = fetch_sounding(url)
    print("Sounding Dataset:")
    print(ds)
    print()

    freezing = find_freezing_level(ds)
    print(f"Freezing level: {freezing['freezing_level_m']} m")
    print(f"Pressure at freezing level: {freezing['pressure_hPa']} hPa")
    print(f"Lower point: {freezing['lower_point']}")
    print(f"Upper point: {freezing['upper_point']}")
