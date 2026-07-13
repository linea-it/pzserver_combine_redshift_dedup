from __future__ import annotations

"""Combine Redshift Catalogs - catalog preparation.

Loads a raw spectroscopic catalog, validates/normalizes schema, derives
standardized fields (IDs, homogenized flags, footprint flags), optionally imports
HATS and generates margin cache, and writes a prepared Parquet artifact.

Public API:
    - prepare_catalog
"""

# -----------------------
# Standard library
# -----------------------
import ast as _ast
import difflib
import hashlib
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Sequence

# -----------------------
# Third-party
# -----------------------
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import get_client as _get_client
from dask.distributed import wait

dask.config.set({"dataframe.shuffle.method": "tasks"})
os.environ.setdefault("DASK_DISTRIBUTED__SHUFFLE__METHOD", "tasks")

# -----------------------
# Project (lsdb/hats/CRC)
# -----------------------
import hats  # noqa: F401
from deduplication import (
    _INSTRUMENT_TYPE_TO_INCLUDE_KEY,
    _OBJECT_TYPE_TO_INCLUDE_KEY,
    validate_instrument_type_inclusion,
    validate_object_type_inclusion,
)
from product_handle import (
    ProductHandle,
)
from product_handle import (
    build_collection_with_retry as _build_collection_with_retry,
)
from specz_homogenization import (
    JADES_LETTER_TO_SCORE,
    VIMOS_FLAG_TO_SCORE,
    _homogenize,
    _honor_user_homogenized_mapping,
)
from utils import ensure_crc_logger

if TYPE_CHECKING:
    from dask.distributed import Client  # noqa: F401

# -----------------------
# Arrow-backed dtypes (pandas >= 2.x)
# -----------------------
import pyarrow as pa

USE_ARROW_TYPES = True

if USE_ARROW_TYPES:
    DTYPE_STR = pd.ArrowDtype(pa.string())
    DTYPE_FLOAT = pd.ArrowDtype(pa.float64())
    DTYPE_INT = pd.ArrowDtype(pa.int64())
    DTYPE_BOOL = pd.ArrowDtype(pa.bool_())
    DTYPE_INT8 = pd.ArrowDtype(pa.int8())
else:
    DTYPE_STR = "string"
    DTYPE_FLOAT = "Float64"
    DTYPE_INT = "Int64"
    DTYPE_BOOL = "boolean"
    DTYPE_INT8 = "Int8"

HOMOGENIZED_COLUMNS = (
    "z_flag_homogenized",
    "instrument_type_homogenized",
    "object_type_homogenized",
)

# -----------------------
# Module exports & constants
# -----------------------
__all__ = ["prepare_catalog"]

LOGGER_NAME = "crc.specz"

DP1_REGIONS = [
    (6.02, -72.08, 2.5),  # 47 Tuc
    (37.86, 6.98, 2.5),  # Rubin SV 38 7
    (40.00, -34.45, 2.5),  # Fornax dSph
    (53.13, -28.10, 2.5),  # ECDFS
    (59.10, -48.73, 2.5),  # EDFS
    (95.00, -25.00, 2.5),  # Rubin SV 95 -25
    (106.23, -10.51, 2.5),  # Seagull
]
RUBIN_BORDER_RA = np.array(
    [
        75.23437500,
        76.64062500,
        78.04687500,
        79.45312500,
        80.85937500,
        82.26562500,
        83.67187500,
        85.07812500,
        86.48437500,
        87.89062500,
        89.29687500,
        90.70312500,
        92.10937500,
        93.51562500,
        94.92187500,
        96.32812500,
        97.73437500,
        99.14062500,
        100.54687500,
        101.95312500,
        103.35937500,
        104.76562500,
        68.90625000,
        70.31250000,
        71.71875000,
        73.12500000,
        74.53125000,
        75.93750000,
        77.34375000,
        78.75000000,
        80.15625000,
        81.56250000,
        82.96875000,
        84.37500000,
        85.78125000,
        87.18750000,
        88.59375000,
        90.00000000,
        91.40625000,
        92.81250000,
        94.21875000,
        95.62500000,
        97.03125000,
        98.43750000,
        99.84375000,
        101.25000000,
        102.65625000,
        104.06250000,
        105.46875000,
        106.87500000,
        108.28125000,
        109.68750000,
        111.09375000,
        63.98437500,
        65.39062500,
        66.79687500,
        68.20312500,
        69.60937500,
        71.01562500,
        72.42187500,
        73.82812500,
        106.17187500,
        107.57812500,
        108.98437500,
        110.39062500,
        111.79687500,
        113.20312500,
        114.60937500,
        116.01562500,
        60.46875000,
        61.87500000,
        63.28125000,
        64.68750000,
        66.09375000,
        67.50000000,
        112.50000000,
        113.90625000,
        115.31250000,
        116.71875000,
        118.12500000,
        119.53125000,
        56.95312500,
        58.35937500,
        59.76562500,
        61.17187500,
        62.57812500,
        117.42187500,
        118.82812500,
        120.23437500,
        121.64062500,
        123.04687500,
        53.43750000,
        54.84375000,
        56.25000000,
        57.65625000,
        59.06250000,
        120.93750000,
        122.34375000,
        123.75000000,
        125.15625000,
        126.56250000,
        49.92187500,
        51.32812500,
        52.73437500,
        54.14062500,
        55.54687500,
        124.45312500,
        125.85937500,
        127.26562500,
        128.67187500,
        130.07812500,
        47.81250000,
        49.21875000,
        50.62500000,
        52.03125000,
        127.96875000,
        129.37500000,
        130.78125000,
        132.18750000,
        44.29687500,
        45.70312500,
        47.10937500,
        48.51562500,
        131.48437500,
        132.89062500,
        134.29687500,
        135.70312500,
        42.18750000,
        43.59375000,
        45.00000000,
        46.40625000,
        133.59375000,
        135.00000000,
        136.40625000,
        137.81250000,
        40.07812500,
        41.48437500,
        42.89062500,
        137.10937500,
        138.51562500,
        139.92187500,
        37.96875000,
        39.37500000,
        40.78125000,
        139.21875000,
        140.62500000,
        142.03125000,
        35.85937500,
        37.26562500,
        38.67187500,
        141.32812500,
        142.73437500,
        144.14062500,
        33.75000000,
        35.15625000,
        36.56250000,
        143.43750000,
        144.84375000,
        146.25000000,
        31.64062500,
        33.04687500,
        34.45312500,
        145.54687500,
        146.95312500,
        148.35937500,
        29.53125000,
        30.93750000,
        32.34375000,
        147.65625000,
        149.06250000,
        150.46875000,
        28.82812500,
        30.23437500,
        149.76562500,
        151.17187500,
        26.71875000,
        28.12500000,
        151.87500000,
        153.28125000,
        185.62500000,
        187.03125000,
        24.60937500,
        26.01562500,
        27.42187500,
        152.57812500,
        153.98437500,
        155.39062500,
        183.51562500,
        184.92187500,
        186.32812500,
        187.73437500,
        189.14062500,
        22.50000000,
        23.90625000,
        25.31250000,
        154.68750000,
        156.09375000,
        157.50000000,
        182.81250000,
        184.21875000,
        188.43750000,
        189.84375000,
        191.25000000,
        21.79687500,
        23.20312500,
        156.79687500,
        158.20312500,
        182.10937500,
        190.54687500,
        191.95312500,
        19.68750000,
        21.09375000,
        158.90625000,
        160.31250000,
        181.40625000,
        192.65625000,
        17.57812500,
        18.98437500,
        20.39062500,
        159.60937500,
        161.01562500,
        162.42187500,
        180.70312500,
        193.35937500,
        16.87500000,
        18.28125000,
        161.71875000,
        163.12500000,
        180.00000000,
        194.06250000,
        14.76562500,
        16.17187500,
        163.82812500,
        165.23437500,
        179.29687500,
        193.35937500,
        14.06250000,
        15.46875000,
        164.53125000,
        165.93750000,
        178.59375000,
        194.06250000,
        11.95312500,
        13.35937500,
        166.64062500,
        168.04687500,
        179.29687500,
        194.76562500,
        9.84375000,
        11.25000000,
        12.65625000,
        167.34375000,
        168.75000000,
        170.15625000,
        178.59375000,
        194.06250000,
        9.14062500,
        10.54687500,
        169.45312500,
        170.85937500,
        172.26562500,
        173.67187500,
        175.07812500,
        176.48437500,
        177.89062500,
        194.76562500,
        196.17187500,
        197.57812500,
        198.98437500,
        200.39062500,
        201.79687500,
        203.20312500,
        204.60937500,
        206.01562500,
        207.42187500,
        208.82812500,
        210.23437500,
        211.64062500,
        213.04687500,
        214.45312500,
        215.85937500,
        217.26562500,
        218.67187500,
        220.07812500,
        221.48437500,
        222.89062500,
        224.29687500,
        225.70312500,
        227.10937500,
        228.51562500,
        229.92187500,
        231.32812500,
        232.73437500,
        234.14062500,
        235.54687500,
        236.95312500,
        238.35937500,
        239.76562500,
        241.17187500,
        242.57812500,
        243.98437500,
        245.39062500,
        246.79687500,
        248.20312500,
        249.60937500,
        251.01562500,
        252.42187500,
        253.82812500,
        255.23437500,
        256.64062500,
        258.04687500,
        259.45312500,
        260.85937500,
        262.26562500,
        263.67187500,
        265.07812500,
        266.48437500,
        267.89062500,
        269.29687500,
        270.70312500,
        272.10937500,
        273.51562500,
        274.92187500,
        276.32812500,
        277.73437500,
        279.14062500,
        280.54687500,
        281.95312500,
        283.35937500,
        284.76562500,
        286.17187500,
        287.57812500,
        288.98437500,
        290.39062500,
        291.79687500,
        293.20312500,
        294.60937500,
        296.01562500,
        297.42187500,
        298.82812500,
        300.23437500,
        301.64062500,
        7.03125000,
        8.43750000,
        171.56250000,
        172.96875000,
        174.37500000,
        175.78125000,
        177.18750000,
        195.46875000,
        196.87500000,
        198.28125000,
        199.68750000,
        201.09375000,
        202.50000000,
        203.90625000,
        205.31250000,
        206.71875000,
        208.12500000,
        209.53125000,
        210.93750000,
        212.34375000,
        213.75000000,
        215.15625000,
        216.56250000,
        217.96875000,
        219.37500000,
        220.78125000,
        222.18750000,
        223.59375000,
        225.00000000,
        226.40625000,
        227.81250000,
        229.21875000,
        230.62500000,
        232.03125000,
        233.43750000,
        234.84375000,
        236.25000000,
        237.65625000,
        239.06250000,
        240.46875000,
        241.87500000,
        243.28125000,
        244.68750000,
        246.09375000,
        247.50000000,
        248.90625000,
        250.31250000,
        251.71875000,
        253.12500000,
        254.53125000,
        255.93750000,
        257.34375000,
        258.75000000,
        260.15625000,
        261.56250000,
        262.96875000,
        264.37500000,
        265.78125000,
        267.18750000,
        268.59375000,
        270.00000000,
        271.40625000,
        272.81250000,
        274.21875000,
        275.62500000,
        277.03125000,
        278.43750000,
        279.84375000,
        281.25000000,
        282.65625000,
        284.06250000,
        285.46875000,
        286.87500000,
        288.28125000,
        289.68750000,
        291.09375000,
        292.50000000,
        293.90625000,
        295.31250000,
        296.71875000,
        298.12500000,
        299.53125000,
        300.93750000,
        6.32812500,
        7.73437500,
        300.23437500,
        4.21875000,
        5.62500000,
        300.93750000,
        3.51562500,
        4.92187500,
        300.23437500,
        1.40625000,
        2.81250000,
        299.53125000,
        0.70312500,
        2.10937500,
        298.82812500,
        0.00000000,
        299.53125000,
        358.59375000,
        298.82812500,
        357.89062500,
        359.29687500,
        298.12500000,
        355.78125000,
        357.18750000,
        298.82812500,
        355.07812500,
        356.48437500,
        298.12500000,
        352.96875000,
        354.37500000,
        297.42187500,
        352.26562500,
        353.67187500,
        298.12500000,
        350.15625000,
        351.56250000,
        297.42187500,
        349.45312500,
        350.85937500,
        298.12500000,
        347.34375000,
        348.75000000,
        297.42187500,
        346.64062500,
        348.04687500,
        298.12500000,
        344.53125000,
        345.93750000,
        297.42187500,
        343.82812500,
        345.23437500,
        298.12500000,
        299.53125000,
        300.93750000,
        302.34375000,
        303.75000000,
        305.15625000,
        306.56250000,
        307.96875000,
        309.37500000,
        310.78125000,
        312.18750000,
        313.59375000,
        315.00000000,
        316.40625000,
        317.81250000,
        319.21875000,
        320.62500000,
        322.03125000,
        323.43750000,
        324.84375000,
        326.25000000,
        327.65625000,
        329.06250000,
        330.46875000,
        331.87500000,
        333.28125000,
        334.68750000,
        336.09375000,
        337.50000000,
        338.90625000,
        340.31250000,
        341.71875000,
        343.12500000,
        298.82812500,
        300.23437500,
        301.64062500,
        303.04687500,
        304.45312500,
        305.85937500,
        307.26562500,
        308.67187500,
        310.07812500,
        311.48437500,
        312.89062500,
        314.29687500,
        315.70312500,
        317.10937500,
        318.51562500,
        319.92187500,
        321.32812500,
        322.73437500,
        324.14062500,
        325.54687500,
        326.95312500,
        328.35937500,
        329.76562500,
        331.17187500,
        332.57812500,
        333.98437500,
        335.39062500,
        336.79687500,
        338.20312500,
        339.60937500,
        341.01562500,
        342.42187500,
    ],
    dtype=float,
)

RUBIN_BORDER_DEC = np.array(
    [
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.79716830,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        32.08995126,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        31.38816646,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.69158768,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        30.00000000,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        29.31319896,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        28.63098984,
        27.95318688,
        27.95318688,
        27.95318688,
        27.95318688,
        27.95318688,
        27.95318688,
        27.95318688,
        27.95318688,
        27.27961274,
        27.27961274,
        27.27961274,
        27.27961274,
        27.27961274,
        27.27961274,
        27.27961274,
        27.27961274,
        26.61009781,
        26.61009781,
        26.61009781,
        26.61009781,
        26.61009781,
        26.61009781,
        26.61009781,
        26.61009781,
        25.94447977,
        25.94447977,
        25.94447977,
        25.94447977,
        25.94447977,
        25.94447977,
        25.28260304,
        25.28260304,
        25.28260304,
        25.28260304,
        25.28260304,
        25.28260304,
        24.62431835,
        24.62431835,
        24.62431835,
        24.62431835,
        24.62431835,
        24.62431835,
        23.96948232,
        23.96948232,
        23.96948232,
        23.96948232,
        23.96948232,
        23.96948232,
        23.31795707,
        23.31795707,
        23.31795707,
        23.31795707,
        23.31795707,
        23.31795707,
        22.66960987,
        22.66960987,
        22.66960987,
        22.66960987,
        22.66960987,
        22.66960987,
        22.02431284,
        22.02431284,
        22.02431284,
        22.02431284,
        21.38194258,
        21.38194258,
        21.38194258,
        21.38194258,
        21.38194258,
        21.38194258,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.74237995,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        20.10550979,
        19.47122063,
        19.47122063,
        19.47122063,
        19.47122063,
        19.47122063,
        19.47122063,
        19.47122063,
        18.83940455,
        18.83940455,
        18.83940455,
        18.83940455,
        18.83940455,
        18.83940455,
        18.20995686,
        18.20995686,
        18.20995686,
        18.20995686,
        18.20995686,
        18.20995686,
        18.20995686,
        18.20995686,
        17.58277601,
        17.58277601,
        17.58277601,
        17.58277601,
        17.58277601,
        17.58277601,
        16.95776330,
        16.95776330,
        16.95776330,
        16.95776330,
        16.95776330,
        16.95776330,
        16.33482278,
        16.33482278,
        16.33482278,
        16.33482278,
        16.33482278,
        16.33482278,
        15.71386105,
        15.71386105,
        15.71386105,
        15.71386105,
        15.71386105,
        15.71386105,
        15.09478710,
        15.09478710,
        15.09478710,
        15.09478710,
        15.09478710,
        15.09478710,
        15.09478710,
        15.09478710,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        14.47751219,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.86194967,
        13.24801491,
        13.24801491,
        13.24801491,
        12.63562509,
        12.63562509,
        12.63562509,
        12.02469918,
        12.02469918,
        12.02469918,
        11.41515774,
        11.41515774,
        11.41515774,
        10.80692287,
        10.80692287,
        10.80692287,
        10.19991809,
        10.19991809,
        10.19991809,
        9.59406823,
        9.59406823,
        9.59406823,
        8.98929935,
        8.98929935,
        8.98929935,
        8.38553865,
        8.38553865,
        8.38553865,
        7.78271439,
        7.78271439,
        7.78271439,
        7.18075578,
        7.18075578,
        7.18075578,
        6.57959294,
        6.57959294,
        6.57959294,
        5.97915680,
        5.97915680,
        5.97915680,
        5.37937899,
        5.37937899,
        5.37937899,
        4.78019185,
        4.78019185,
        4.78019185,
        4.18152827,
        4.18152827,
        4.18152827,
        3.58332170,
        3.58332170,
        3.58332170,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.98550601,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
        2.38801546,
    ],
    dtype=float,
)


# -----------------------
# Centralized logging
# -----------------------
def _get_logger() -> logging.Logger:
    """Return a child logger that propagates to the root 'crc' logger.

    Returns:
        logging.Logger: Base logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def _phase_logger(
    base_logger: logging.Logger, phase: str, product: str | None = None
) -> logging.LoggerAdapter:
    """Return a LoggerAdapter injecting phase (and product if provided).

    Args:
        base_logger: Base logger.
        phase: Phase label.
        product: Optional product identifier.

    Returns:
        logging.LoggerAdapter: Logger with extra context.
    """
    extra = {"phase": phase}
    if product:
        extra["product"] = product
    return logging.LoggerAdapter(base_logger, extra)


# -----------------------
# YAML mapping validation
# -----------------------
def _looks_like_pipeline_output(columns: list[str]) -> bool:
    """Return True when columns match this pipeline's output schema."""
    have = set(map(str, columns))
    required = {"CRD_ID", "ra", "dec", "z", "source"}
    markers = {
        "tie_result",
        "compared_to",
        "group_id",
        "z_flag_homogenized",
        "instrument_type_homogenized",
        "is_in_DP1_fields",
        "is_in_rubin_footprint",
    }
    return required.issubset(have) and bool(markers & have)


def _prefer_pipeline_output_id_mapping(
    entry: dict, columns: list[str], logger: logging.Logger
) -> dict:
    """Prefer original ``id`` over previous ``CRD_ID`` for pipeline outputs."""
    if not _looks_like_pipeline_output(columns):
        return entry

    have = set(map(str, columns))
    source_id = "id" if "id" in have else "CRD_ID"
    columns_cfg = dict(entry.get("columns") or {})
    current = columns_cfg.get("id")

    if current != source_id:
        logger.info(
            "%s Detected previous pipeline output; using '%s' as input id "
            "instead of configured '%s'.",
            entry["internal_name"],
            source_id,
            current,
        )
        columns_cfg["id"] = source_id
        entry = {**entry, "columns": columns_cfg}

    return entry


def _validate_and_rename(
    df: dd.DataFrame, entry: dict, logger: logging.Logger
) -> dd.DataFrame:
    """Validate YAML column mapping and apply conflict-safe renames.

    Verifies existence of non-null source columns, suggests close matches,
    and safely parks conflicting targets as ``<target>__origN`` before renaming.

    Args:
        df: Input Dask DataFrame.
        entry: YAML entry with ``internal_name`` and ``columns`` mapping.
        logger: Logger.

    Returns:
        dd.DataFrame: DataFrame with validated/renamed columns and base schema.

    Raises:
        ValueError: If required source columns are missing.
    """
    product_name = entry["internal_name"]
    columns_cfg = entry.get("columns") or {}
    non_null_map = {
        std: src for std, src in columns_cfg.items() if src not in (None, "", "null")
    }
    input_cols = list(map(str, df.columns))

    missing_sources = [src for src in non_null_map.values() if src not in input_cols]
    if missing_sources:
        suggestions = {
            src: difflib.get_close_matches(src, input_cols, n=3, cutoff=0.6)
            for src in missing_sources
        }
        raise ValueError(
            f"[{product_name}] Missing mapped source columns in input parquet: {missing_sources}\n"
            f"Configured (non-null) mapping: {non_null_map}\n"
            f"Closest matches: {suggestions}\n"
            f"Available columns (sample): {sorted(input_cols)[:30]} ..."
        )

    # Resolve collisions: if target exists and differs from its source, park it aside.
    for std, src in non_null_map.items():
        tgt = std
        if src != tgt and tgt in df.columns:
            base = f"{tgt}__orig"
            parked = base
            i = 1
            existing = set(map(str, df.columns))
            while parked in existing:
                parked = f"{base}{i}"
                i += 1
            logger.info(
                f"{product_name} Resolve rename collision: '{tgt}' already exists; "
                f"'{tgt}' -> '{parked}' before mapping '{src}' -> '{tgt}'"
            )
            df = df.rename(columns={tgt: parked})

    col_map = {src: std for std, src in non_null_map.items()}
    if col_map:
        logger.info(
            f"{product_name} Rename map (sample up to 6): {list(col_map.items())[:6]}"
        )
        df = df.rename(columns=col_map)
    else:
        logger.info(f"{product_name} No non-null column mappings; skipping rename.")

    # Tag source and ensure minimal base schema
    df = df.assign(source=product_name)
    base_schema = {
        "id": DTYPE_STR,
        "instrument_type": DTYPE_STR,
        "survey": DTYPE_STR,
        "ra": DTYPE_FLOAT,
        "dec": DTYPE_FLOAT,
        "z": DTYPE_FLOAT,
        "z_flag": DTYPE_FLOAT,
        "z_err": DTYPE_FLOAT,
        "object_type": DTYPE_STR,
    }
    for col, pd_dtype in base_schema.items():
        if col not in df.columns:
            df = _add_missing_with_dtype(df, col, pd_dtype)

    return df


def _rename_duplicate_columns_dd(
    df: dd.DataFrame, logger: logging.Logger
) -> dd.DataFrame:
    """Make column names unique across partitions by appending __dupN.

    Args:
        df: Input Dask DataFrame.
        logger: Logger.

    Returns:
        dd.DataFrame: DataFrame with unique column names.
    """

    def _renamer(pdf: pd.DataFrame) -> pd.DataFrame:
        cols = list(map(str, pdf.columns))
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        renamed_pairs: list[tuple[str, str]] = []

        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new_name = f"{c}__dup{seen[c]}"
                while new_name in seen:
                    seen[c] += 1
                    new_name = f"{c}__dup{seen[c]}"
                seen[new_name] = 0
                new_cols.append(new_name)
                renamed_pairs.append((c, new_name))

        if renamed_pairs:
            sample = ", ".join([f"{a}->{b}" for a, b in renamed_pairs[:6]])
            logger.warning(
                f"Renamed duplicate columns in partition: {sample}"
                f"{' ...' if len(renamed_pairs) > 6 else ''}"
            )

        out = pdf.copy()
        out.columns = new_cols
        return out

    meta_fixed = _renamer(df._meta)
    return df.map_partitions(_renamer, meta=meta_fixed)


def _rename_duplicate_columns_pd(
    pdf: pd.DataFrame, logger: logging.Logger
) -> pd.DataFrame:
    """Make pandas columns unique by appending __dupN.

    Args:
        pdf: Input pandas DataFrame.
        logger: Logger.

    Returns:
        pd.DataFrame: DataFrame with unique column names.
    """
    cols = pd.Index(map(str, pdf.columns))
    if cols.has_duplicates:
        seen: dict[str, int] = {}
        new_cols: list[str] = []
        renamed: list[tuple[str, str]] = []

        for c in cols:
            if c not in seen:
                seen[c] = 0
                new_cols.append(c)
            else:
                seen[c] += 1
                new = f"{c}__dup{seen[c]}"
                while new in seen:
                    seen[c] += 1
                    new = f"{c}__dup{seen[c]}"
                seen[new] = 0
                new_cols.append(new)
                renamed.append((c, new))

        if renamed:
            sample = ", ".join([f"{a}->{b}" for a, b in renamed[:6]])
            logger.warning(
                f"Renamed duplicate columns (pandas): {sample}"
                f"{' ...' if len(renamed) > 6 else ''}"
            )

        pdf = pdf.copy()
        pdf.columns = new_cols

    return pdf


def _drop_previous_results(df: dd.DataFrame, logger: logging.Logger) -> dd.DataFrame:
    """Discard result columns produced by earlier pipeline runs.

    This runs after the configured input mapping, so an input ``CRD_ID`` mapped
    to ``id`` is retained as the source identifier. All current and historical
    pipeline result columns are otherwise removed before new results are built.

    Args:
        df: Frame after rename.
        logger: Logger.

    Returns:
        dd.DataFrame: Frame without results from previous runs.
    """
    result_columns = {"CRD_ID", "compared_to", "group_id"}
    historical_prefixes = ("CRD_ID_prev", "compared_to_prev", "group_id_prev")
    to_drop = [
        col
        for col in map(str, df.columns)
        if col in result_columns or col.startswith(historical_prefixes)
    ]
    if to_drop:
        logger.info("Discard previous pipeline result columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


# -----------------------
# Type helpers
# -----------------------
def _add_missing_with_dtype(_df: dd.DataFrame, col: str, pd_dtype: Any) -> dd.DataFrame:
    """Add missing column with a specific dtype and valid Dask meta.

    Args:
        _df: Input DataFrame.
        col: Column to add.
        pd_dtype: Target pandas/Arrow dtype.

    Returns:
        dd.DataFrame: Frame with column added (if missing).
    """
    if col in _df.columns:
        return _df

    meta_added = _df._meta.assign(**{col: pd.Series(pd.array([], dtype=pd_dtype))})

    def _adder(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        p[col] = pd.Series(pd.NA, index=p.index, dtype=pd_dtype)
        return p

    return _df.map_partitions(_adder, meta=meta_added)


def _normalize_string_series_to_na(s: pd.Series) -> pd.Series:
    """Normalize to string dtype and coerce placeholders to <NA>.

    Args:
        s: Input series.

    Returns:
        pd.Series: Normalized string series.
    """
    s = s.astype(DTYPE_STR).str.strip()
    low = s.fillna("").str.lower()
    mask_empty = (low == "") | low.isin(["none", "null", "nan"])
    return s.mask(mask_empty, pd.NA).astype(DTYPE_STR)


def _to_nullable_boolean_strict(s: pd.Series) -> pd.Series:
    """Convert to nullable boolean strictly (non-bool -> <NA>).

    Args:
        s: Input series.

    Returns:
        Nullable boolean series.
    """
    if s.dtype == object or str(s.dtype).startswith("string"):
        s = s.astype(DTYPE_STR).str.strip()
        low = s.fillna("").str.lower()
        s = s.mask((low == "") | low.isin(["none", "null", "nan"]), pd.NA)

    vals = s.astype("object")
    mask_true = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is True)
    mask_false = vals.apply(lambda v: isinstance(v, (bool, np.bool_)) and v is False)

    out = pd.Series(pd.array([pd.NA] * len(vals), dtype=DTYPE_BOOL), index=vals.index)
    out[mask_true] = True
    out[mask_false] = False
    return out


def _normalize_schema_hints(hints: dict | None) -> dict:
    """Normalize YAML dtype hints to {'int','float','str','bool'}.

    Args:
        hints: Mapping column -> dtype.

    Returns:
        dict: Normalized hints.
    """
    if not hints:
        return {}
    norm = {}
    for col, dt in hints.items():
        if dt is None:
            continue
        k = str(col)
        v = str(dt).strip().lower()
        if v in {"int", "int64"}:
            norm[k] = "int"
        elif v in {"float", "float64", "double"}:
            norm[k] = "float"
        elif v in {"str", "string"}:
            norm[k] = "str"
        elif v in {"bool", "boolean"}:
            norm[k] = "bool"
    return norm


def _normalize_extra_columns_config(value: Any) -> dict[str, dict[str, str]]:
    """Validate and normalize ``param.extra_columns``.

    Args:
        value: Configured mapping of output column names to dtypes.

    Returns:
        Mapping of output names to normalized ``source`` and ``type`` values.

    Raises:
        TypeError: If the configured value is not a mapping.
        ValueError: If a column has an unsupported dtype.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            "param.extra_columns must be a mapping of column names to dtypes"
        )

    normalized: dict[str, dict[str, str]] = {}
    for output_name, raw_spec in value.items():
        output = str(output_name).strip()
        if not output:
            raise ValueError("param.extra_columns cannot contain an empty output name")

        if isinstance(raw_spec, dict):
            unknown = sorted(set(raw_spec) - {"source", "type"})
            if unknown:
                raise ValueError(
                    f"Unknown option(s) for param.extra_columns.{output}: {unknown}"
                )
            source = str(raw_spec.get("source", output)).strip()
            raw_type = raw_spec.get("type")
        else:
            source = output
            raw_type = raw_spec

        normalized_type = _normalize_schema_hints({output: raw_type}).get(output)
        if normalized_type is None:
            raise ValueError(
                "Unsupported dtype in param.extra_columns for column "
                f"'{output}'. Supported types: str, float, int, bool."
            )
        if not source:
            raise ValueError(f"param.extra_columns.{output}.source cannot be empty")
        normalized[output] = {"source": source, "type": normalized_type}

    reserved = {
        "CRD_ID",
        "id",
        "ra",
        "dec",
        "z",
        "z_flag",
        "z_err",
        "instrument_type",
        "survey",
        "source",
        "tie_result",
        "compared_to",
        "group_id",
        "z_flag_homogenized",
        "instrument_type_homogenized",
        "object_type_homogenized",
        "is_in_DP1_fields",
        "is_in_rubin_footprint",
    }
    conflicts = sorted(set(normalized) & reserved)
    if conflicts:
        raise ValueError(
            f"param.extra_columns cannot redefine pipeline columns: {conflicts}"
        )
    return normalized


def build_runtime_schema_hints(
    param_config: dict, translation_config: dict
) -> dict[str, str]:
    """Build schema hints that must survive every crossmatch round.

    Args:
        param_config: Pipeline ``param`` configuration.
        translation_config: Validated flags translation configuration.

    Returns:
        Mapping of output column names to ``str``, ``float``, ``int`` or ``bool``.
    """
    hints: dict[str, str] = {}
    if bool(translation_config.get("save_expr_columns", False)):
        hints.update(
            _normalize_schema_hints(translation_config.get("expr_column_schema"))
        )

    extra_columns = _normalize_extra_columns_config(param_config.get("extra_columns"))
    hints.update({output: spec["type"] for output, spec in extra_columns.items()})

    if _as_bool_config(param_config.get("insert_DP1_footprint_flag"), default=False):
        hints["is_in_DP1_fields"] = "int"
    if _as_bool_config(
        param_config.get("insert_rubin_footprint_flag"), default=False
    ):
        hints["is_in_rubin_footprint"] = "int"

    standard_priorities = {
        "z_flag_homogenized",
        "instrument_type_homogenized",
    }
    for column in translation_config.get("tiebreaking_priority", []) or []:
        name = str(column).strip()
        if name and name not in standard_priorities:
            hints[name] = "float"
    return hints


def _normalize_output_homogenized_columns_config(config: object) -> dict[str, str]:
    """Return output policy for homogenized columns."""
    valid = {"auto", "always", "never"}
    if config is None:
        supplied = {}
    elif isinstance(config, dict):
        supplied = dict(config)
    else:
        raise TypeError("param.output.homogenized_columns must be a mapping")

    unknown = sorted(set(supplied) - set(HOMOGENIZED_COLUMNS))
    if unknown:
        raise ValueError(
            f"Unknown param.output.homogenized_columns option(s): {unknown}"
        )

    result = {column: "always" for column in HOMOGENIZED_COLUMNS}
    for column, value in supplied.items():
        normalized = str(value).strip().lower()
        if normalized not in valid:
            raise ValueError(
                f"param.output.homogenized_columns.{column} must be one of "
                f"{sorted(valid)}"
            )
        result[column] = normalized
    return result


def _active_z_flag_filter(param_config: dict) -> bool:
    try:
        cut_value = float(param_config.get("z_flag_homogenized_value_to_cut"))
    except (TypeError, ValueError):
        return False
    return cut_value in {1.0, 2.0, 3.0, 4.0}


def _active_object_type_filter(param_config: dict) -> bool:
    inclusion = validate_object_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_unclassified_oth",
                "include_galaxy_oth",
                "include_star_oth",
                "include_agn_oth",
                "include_qso_oth",
                "include_galactic_oth",
            )
            if key in param_config
        }
    )
    return not all(inclusion.values())


def _homogenized_columns_used_by_runtime(
    *,
    param_config: dict,
    tiebreaking_priority: list,
) -> set[str]:
    used = set(tiebreaking_priority or []) & set(HOMOGENIZED_COLUMNS)
    if _active_z_flag_filter(param_config):
        used.add("z_flag_homogenized")
    if _active_instrument_type_filter(param_config):
        used.add("instrument_type_homogenized")
    if _active_object_type_filter(param_config):
        used.add("object_type_homogenized")
    return used


def _copy_extra_columns_from_sources(
    df: dd.DataFrame,
    columns: dict[str, dict[str, str]],
    logger: logging.Logger,
) -> dd.DataFrame:
    """Copy configured source columns before standard-column renaming."""
    for output, spec in columns.items():
        source = spec["source"]
        if source == output:
            continue
        if source not in df.columns:
            if output in df.columns:
                logger.info(
                    "Discard extra output column '%s': configured source '%s' "
                    "is absent",
                    output,
                    source,
                )
                df = df.drop(columns=[output])
            continue
        if output in df.columns:
            logger.info(
                "Replace extra output column '%s' with configured source '%s'",
                output,
                source,
            )
        else:
            logger.info("Copy extra column '%s' -> '%s'", source, output)
        df[output] = df[source]
    return df


def _apply_configured_columns(
    df: dd.DataFrame, columns: dict[str, dict[str, str]]
) -> dd.DataFrame:
    """Cast configured columns or create typed null columns when absent."""
    for col, spec in columns.items():
        kind = spec["type"]
        if col not in df.columns:
            dtype = {
                "str": DTYPE_STR,
                "float": DTYPE_FLOAT,
                "int": DTYPE_INT,
                "bool": DTYPE_BOOL,
            }[kind]
            df = _add_missing_with_dtype(df, col, dtype)
        elif kind == "str":
            df[col] = df[col].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            )
        elif kind == "float":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_FLOAT),
                meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
            )
        elif kind == "int":
            coerced = dd.to_numeric(df[col], errors="coerce")
            df[col] = coerced.map_partitions(
                lambda s: s.astype(DTYPE_INT),
                meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
            )
        elif kind == "bool":
            df[col] = df[col].map_partitions(
                _to_nullable_boolean_strict,
                meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
            )
    return df


def _normalize_types(
    df: dd.DataFrame, product_name: str, logger: logging.Logger
) -> tuple[dd.DataFrame, bool]:
    """Normalize core dtypes and clean values.

    Args:
        df: Frame after YAML rename.
        product_name: Catalog identifier.
        logger: Logger.

    Returns:
        Tuple[dd.DataFrame, bool]: (normalized frame, whether 'type' normalized).
    """
    # 1) Normalize string-like
    string_like = [
        "id",
        "instrument_type",
        "survey",
        "instrument_type_homogenized",
        "source",
    ]
    for col in string_like:
        if col in df.columns:
            df[col] = df[col].map_partitions(
                _normalize_string_series_to_na,
                meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
            )
            if col == "survey":
                df[col] = df[col].str.upper()

    # 2) Optional 'type'
    type_cast_ok = False
    if "type" in df.columns:
        try:
            df["type"] = (
                df["type"]
                .map_partitions(
                    _normalize_string_series_to_na,
                    meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                )
                .str.lower()
            )
            type_cast_ok = True
        except Exception as e:
            logger.warning(
                f"{product_name} Failed to normalize 'type' to lower-case: {e}"
            )
            type_cast_ok = False

    # 3) JADES/VIMOS z_flag mapping if needed
    def _map_special_partition(partition: pd.DataFrame) -> pd.DataFrame:
        p = partition.copy()
        if "survey" not in p or "z_flag" not in p:
            return p

        survey_uc = p["survey"].astype(str).str.upper()
        mask_jades = survey_uc == "JADES"
        mask_vimos = survey_uc == "VIMOS"
        if not (mask_jades.any() or mask_vimos.any()):
            return p

        zf_num = pd.to_numeric(p["z_flag"], errors="coerce")
        nonnum_mask = zf_num.isna()
        z_flag_str = p["z_flag"].astype(str)

        def map_jades(val):
            s = str(val).strip().upper()
            return JADES_LETTER_TO_SCORE.get(s, np.nan)

        def map_vimos(val):
            s = str(val).strip().upper()
            return VIMOS_FLAG_TO_SCORE.get(s, np.nan)

        idx_j = nonnum_mask & mask_jades
        if idx_j.any():
            z_flag_str.loc[idx_j] = z_flag_str.loc[idx_j].map(map_jades)

        idx_v = nonnum_mask & mask_vimos
        if idx_v.any():
            z_flag_str.loc[idx_v] = z_flag_str.loc[idx_v].map(map_vimos)

        mapped_numeric = pd.to_numeric(z_flag_str, errors="coerce")
        zf = zf_num.copy()
        zf[nonnum_mask] = mapped_numeric[nonnum_mask]
        p["z_flag"] = zf
        return p

    if "z_flag" in df.columns and "survey" in df.columns:
        df = df.map_partitions(_map_special_partition)

    # 4) Float-like coercion
    float_like = ["ra", "dec", "z", "z_err", "z_flag", "z_flag_homogenized"]
    numeric_columns = [col for col in float_like if col in df.columns]
    coerced_by_column = {
        col: dd.to_numeric(df[col], errors="coerce") for col in numeric_columns
    }
    invalid_masks = {
        col: dd.isna(coerced_by_column[col]) & ~dd.isna(df[col])
        for col in numeric_columns
    }
    invalid_counts = dask.compute(
        *[invalid_masks[col].sum() for col in numeric_columns]
    )
    for col, invalid_count in zip(numeric_columns, invalid_counts):
        if invalid_count > 0:
            sample_vals = df[col].loc[invalid_masks[col]].head(5, compute=True).tolist()
            raise ValueError(
                f"[{product_name}] Failed to convert '{col}' to numeric: "
                f"{invalid_count} non-numeric value(s). Examples: {sample_vals}."
            )
        df[col] = coerced_by_column[col].map_partitions(
            lambda s: s.astype(DTYPE_FLOAT),
            meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
        )

    return df, type_cast_ok


# -----------------------
# Config helpers
# -----------------------
def _as_bool_config(value: Any, default: bool) -> bool:
    """Parse bool-like config values robustly."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


# -----------------------
# CRD_ID generation
# -----------------------
CRD_ID_REQUIRED_HASH_COLUMNS = ("ra", "dec", "z")
CRD_ID_OPTIONAL_HASH_COLUMNS = (
    "id",
    "z_flag",
    "z_err",
    "survey",
    "source",
    "instrument_type",
)
CRD_ID_HASH_COLUMNS = CRD_ID_REQUIRED_HASH_COLUMNS + CRD_ID_OPTIONAL_HASH_COLUMNS


def _crd_id_diagnostic_columns(columns: Sequence[str]) -> list[str]:
    """Return stable columns to show when CRD_ID collisions are detected."""
    wanted = ("CRD_ID",) + CRD_ID_HASH_COLUMNS
    return [column for column in wanted if column in columns]


def _format_crd_signature_value(value: object) -> str:
    """Return a stable scalar representation for CRD_ID hashing."""
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, int, np.floating, np.integer)):
        number = float(value)
        if number == 0.0:
            number = 0.0
        return format(number, ".17g")
    return str(value).strip()


def _crd_signature(columns: Sequence[str], row: tuple[object, ...]) -> str:
    """Return the canonical row signature used for CRD_ID generation."""
    return "|".join(
        f"{column}={_format_crd_signature_value(value)}"
        for column, value in zip(columns, row)
    )


def _crd_hash_id(
    catalog_prefix: str, columns: Sequence[str], row: tuple[object, ...]
) -> str:
    """Return a short deterministic CRD_ID for one canonical row signature."""
    digest = hashlib.blake2b(
        _crd_signature(columns, row).encode("utf-8"), digest_size=8
    )
    return f"CRD{catalog_prefix}_{digest.hexdigest()}"


def _generate_crd_ids(
    df: dd.DataFrame,
    product_name: str,
    temp_dir: str,
    client: "Client | None" = None,
) -> dd.DataFrame:
    """Assign deterministic, catalog-scoped CRD_IDs from canonical input fields.

    Args:
        df: Input frame after schema normalization.
        product_name: Internal name (expects numeric prefix before underscore).
        temp_dir: Unused, kept for signature stability.
        client: Unused, kept for signature stability.

    Returns:
        dd.DataFrame: Frame with CRD_ID column (Arrow string dtype).

    Raises:
        ValueError: If numeric prefix cannot be extracted from product_name.
        KeyError: If any of the required identity columns are missing.
    """
    m = re.match(r"(\d+)_", product_name)
    if not m:
        raise ValueError(
            f"Could not extract numeric prefix from internal_name '{product_name}'"
        )
    catalog_prefix = m.group(1)

    missing = [
        column for column in CRD_ID_REQUIRED_HASH_COLUMNS if column not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required columns for CRD_ID generation: {missing}")
    hash_columns = [column for column in CRD_ID_HASH_COLUMNS if column in df.columns]

    def _add_crd(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        values = p[hash_columns].itertuples(index=False, name=None)
        p["CRD_ID"] = [
            _crd_hash_id(catalog_prefix, hash_columns, row) for row in values
        ]
        return p

    return df.map_partitions(
        _add_crd,
        meta=df._meta.assign(CRD_ID=pd.Series(pd.array([], dtype=DTYPE_STR))),
    )


def _log_crd_id_collision_diagnostics(
    df: dd.DataFrame,
    product_name: str,
    logger: logging.LoggerAdapter,
    *,
    duplicate_rows: int,
    max_groups: int = 5,
    max_rows: int = 50,
) -> None:
    """Log a bounded sample of duplicate CRD_ID groups before failing."""
    try:
        counts = df.groupby("CRD_ID").size().rename("_count").reset_index()
        duplicate_counts = counts[counts["_count"] > 1]
        duplicate_summary = duplicate_counts.sort_values(
            "_count", ascending=False
        ).head(max_groups, npartitions=-1)
        if duplicate_summary.empty:
            logger.error(
                "%s CRD_ID collision diagnostics found duplicate_rows=%d but no "
                "duplicate groups were sampled.",
                product_name,
                duplicate_rows,
            )
            return

        sample_ids = duplicate_summary["CRD_ID"].astype(str).tolist()
        diagnostic_columns = _crd_id_diagnostic_columns(df.columns)
        sample_rows = (
            df.loc[df["CRD_ID"].isin(sample_ids), diagnostic_columns]
            .head(max_rows, npartitions=-1)
            .to_dict("records")
        )
        logger.error(
            "%s CRD_ID collision diagnostics: duplicate_rows=%d "
            "sample_groups=%s sample_rows=%s",
            product_name,
            duplicate_rows,
            duplicate_summary.to_dict("records"),
            sample_rows,
        )
    except Exception as exc:
        logger.error(
            "%s CRD_ID collision diagnostics failed: %r",
            product_name,
            exc,
        )


def _validate_unique_crd_ids(
    df: dd.DataFrame, product_name: str, logger: logging.LoggerAdapter
) -> None:
    """Fail preparation when generated CRD_ID values are not unique."""
    total_rows, unique_ids = dask.compute(
        df.map_partitions(len).sum(),
        df["CRD_ID"].nunique(dropna=False),
    )
    total_rows = int(total_rows)
    unique_ids = int(unique_ids)
    duplicate_rows = total_rows - unique_ids
    if duplicate_rows:
        _log_crd_id_collision_diagnostics(
            df, product_name, logger, duplicate_rows=duplicate_rows
        )
        raise RuntimeError(
            f"{product_name}: generated non-unique CRD_ID values "
            f"(rows={total_rows}, unique_ids={unique_ids}, "
            f"duplicate_rows={duplicate_rows})"
        )
    logger.info(
        "%s CRD_ID uniqueness validated: rows=%d unique_ids=%d",
        product_name,
        total_rows,
        unique_ids,
    )


# -----------------------
# RA/DEC strict validation (fail fast)
# -----------------------
def _validate_ra_dec_or_fail(df: dd.DataFrame, product_name: str) -> None:
    """Validate RA/DEC finiteness and ranges; raise on invalid rows.

    Args:
        df: Frame with 'ra' and 'dec'.
        product_name: Catalog identifier.

    Raises:
        ValueError: If invalid RA/DEC entries are detected.
    """

    def _isfinite_series(s: pd.Series) -> pd.Series:
        arr = s.astype("float64")
        return pd.Series(np.isfinite(arr), index=s.index)

    isfinite_ra = df["ra"].map_partitions(_isfinite_series, meta=("ra", "bool"))
    isfinite_dec = df["dec"].map_partitions(_isfinite_series, meta=("dec", "bool"))

    ra64 = df["ra"].astype("float64")
    dec64 = df["dec"].astype("float64")

    in_range = (ra64 >= 0.0) & (ra64 < 360.0) & (dec64 >= -90.0) & (dec64 <= 90.0)
    invalid_mask = ~(isfinite_ra & isfinite_dec & in_range)

    na_ra, na_dec = df["ra"].isna().sum(), df["dec"].isna().sum()
    nonfinite_ra, nonfinite_dec = (~isfinite_ra).sum(), (~isfinite_dec).sum()
    oor_ra_low, oor_ra_high = (ra64 < 0.0).sum(), (ra64 >= 360.0).sum()
    oor_dec_low, oor_dec_high = (dec64 < -90.0).sum(), (dec64 > 90.0).sum()
    invalid_total = invalid_mask.sum()

    (
        na_ra,
        na_dec,
        nonfinite_ra,
        nonfinite_dec,
        oor_ra_low,
        oor_ra_high,
        oor_dec_low,
        oor_dec_high,
        invalid_total,
    ) = dask.compute(
        na_ra,
        na_dec,
        nonfinite_ra,
        nonfinite_dec,
        oor_ra_low,
        oor_ra_high,
        oor_dec_low,
        oor_dec_high,
        invalid_total,
    )

    if invalid_total > 0:
        cols = [
            c
            for c in ["CRD_ID", "id", "source", "survey", "ra", "dec"]
            if c in df.columns
        ]
        sample_records = (
            df[invalid_mask][cols].head(5, compute=True).to_dict(orient="records")
        )
        raise ValueError(
            f"[{product_name}] Invalid RA/DEC rows: {invalid_total}\n"
            f"  RA NaN={na_ra}, DEC NaN={na_dec}\n"
            f"  RA non-finite={nonfinite_ra}, DEC non-finite={nonfinite_dec}\n"
            f"  RA <0={oor_ra_low}, RA >=360={oor_ra_high}\n"
            f"  DEC <-90={oor_dec_low}, DEC >90={oor_dec_high}\n"
            f"  Sample (up to 5): {sample_records}"
        )


# -----------------------
# Footprint flagging
# -----------------------
def _load_rubin_border_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return hardcoded Rubin border RA/DEC arrays."""
    return RUBIN_BORDER_RA, RUBIN_BORDER_DEC


def _build_rubin_dec_limit_interpolator(
    border_ra: np.ndarray,
    border_dec: np.ndarray,
    *,
    bin_width: float = 0.5,
    margin_deg: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build RA/DEC lookup arrays to evaluate Rubin footprint membership."""
    border = pd.DataFrame(
        {
            "ra": np.asarray(border_ra, dtype=float) % 360.0,
            "dec": np.asarray(border_dec, dtype=float),
        }
    )
    border["ra_bin"] = np.floor(border["ra"] / bin_width) * bin_width

    upper = (
        border.groupby("ra_bin", as_index=False)
        .agg(dec_upper=("dec", "max"))
        .sort_values("ra_bin")
        .reset_index(drop=True)
    )
    upper["ra_center"] = upper["ra_bin"] + bin_width / 2.0
    upper["dec_limit"] = upper["dec_upper"] + margin_deg

    ra_grid = upper["ra_center"].to_numpy(dtype=float)
    dec_grid = upper["dec_limit"].to_numpy(dtype=float)

    # Handle 0/360 wrap-around in interpolation.
    ra_ext = np.concatenate([ra_grid - 360.0, ra_grid, ra_grid + 360.0])
    dec_ext = np.concatenate([dec_grid, dec_grid, dec_grid])
    return ra_ext, dec_ext


def _flag_dp1(df: dd.DataFrame) -> dd.DataFrame:
    """Flag rows within predefined DP1 circular fields.

    Args:
        df: Frame with 'ra' and 'dec' (degrees).

    Returns:
        dd.DataFrame: Adds 'is_in_DP1_fields' (nullable int).
    """
    ra_centers = np.deg2rad([r[0] for r in DP1_REGIONS])
    dec_centers = np.deg2rad([r[1] for r in DP1_REGIONS])
    radii = [r[2] for r in DP1_REGIONS]

    def _compute(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        ra_rad = np.deg2rad(p["ra"].to_numpy(dtype=float, copy=False))
        dec_rad = np.deg2rad(p["dec"].to_numpy(dtype=float, copy=False))
        in_any = np.zeros(len(p), dtype=bool)
        for ra_c, dec_c, rdeg in zip(ra_centers, dec_centers, radii):
            cos_ang = np.sin(dec_c) * np.sin(dec_rad) + np.cos(dec_c) * np.cos(
                dec_rad
            ) * np.cos(ra_rad - ra_c)
            ang_deg = np.rad2deg(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
            in_any |= ang_deg <= rdeg
        p["is_in_DP1_fields"] = pd.Series(in_any, index=p.index, dtype=DTYPE_INT)
        return p

    meta = df._meta.assign(is_in_DP1_fields=pd.Series(pd.array([], dtype=DTYPE_INT)))
    return df.map_partitions(_compute, meta=meta)


def _flag_rubin_footprint(df: dd.DataFrame) -> dd.DataFrame:
    """Flag rows that lie inside hardcoded Rubin footprint.

    The footprint rule follows the reference file logic: object is inside when
    ``dec <= dec_limit_at_ra`` where ``dec_limit_at_ra`` is interpolated from the
    upper boundary binned in 0.5 deg in RA and expanded by +1 deg in DEC.
    """
    border_ra, border_dec = _load_rubin_border_arrays()
    ra_ext, dec_ext = _build_rubin_dec_limit_interpolator(
        border_ra, border_dec, bin_width=0.5, margin_deg=1.0
    )

    def _compute(part: pd.DataFrame) -> pd.DataFrame:
        p = part.copy()
        ra = p["ra"].to_numpy(dtype=float) % 360.0
        dec = p["dec"].to_numpy(dtype=float)
        dec_limit = np.interp(ra, ra_ext, dec_ext)
        in_footprint = np.isfinite(ra) & np.isfinite(dec) & (dec <= dec_limit)
        p["is_in_rubin_footprint"] = pd.Series(
            np.where(in_footprint, 1, 0), index=p.index, dtype=DTYPE_INT
        )
        return p

    meta = df._meta.assign(
        is_in_rubin_footprint=pd.Series(pd.array([], dtype=DTYPE_INT))
    )
    return df.map_partitions(_compute, meta=meta)


# -----------------------
# Column selection
# -----------------------
def _extract_variables_from_expr(expr: str) -> set[str]:
    """Extract variable names referenced in an expression.

    Args:
        expr: Expression string.

    Returns:
        Set of variable names.
    """
    try:
        tree = _ast.parse(expr, mode="eval")
    except Exception:
        return set()

    class _Visitor(_ast.NodeVisitor):
        def __init__(self):
            self.vars = set()

        def visit_Name(self, node: _ast.Name) -> None:  # type: ignore[override]
            self.vars.add(node.id)
            self.generic_visit(node)

    v = _Visitor()
    v.visit(tree)
    return v.vars


def _select_output_columns(
    df: dd.DataFrame,
    translation_rules_uc: dict,
    tiebreaking_priority: list,
    used_type_fastpath: bool,
    param_config: dict | None = None,
    save_expr_columns: bool = False,
    schema_hints: dict | None = None,
    extra_columns: dict[str, dict[str, str]] | None = None,
) -> dd.DataFrame:
    """Assemble final output schema and coerce optional expression columns.

    Args:
      df: Frame after tie-breaking and footprint flagging.
      translation_rules_uc: Upper-cased translation rules.
      tiebreaking_priority: Priority columns to append if present.
      used_type_fastpath: Whether `type` was reused for instrument_type.
      param_config: Pipeline ``param`` configuration.
      save_expr_columns: Keep variables used in YAML expressions.
      schema_hints: Normalized hints {'int','float','str','bool'}.
      extra_columns: Configured columns to preserve or create as typed nulls.

    Returns:
      dd.DataFrame: Subset with deterministic column order.
    """
    # Base schema (only columns that exist will be kept at the end).
    final_cols = [
        "CRD_ID",
        "id",
        "ra",
        "dec",
        "z",
        "z_flag",
        "z_err",
        "instrument_type",
        "survey",
        "source",
        "tie_result",
        "is_in_DP1_fields",
        "is_in_rubin_footprint",
        "compared_to",
        # New optional current component label (will only be kept if present)
        "group_id",
    ]

    param_config = param_config or {}
    homogenized_output = _normalize_output_homogenized_columns_config(
        param_config.get("output_homogenized_columns")
    )
    runtime_used_homogenized = _homogenized_columns_used_by_runtime(
        param_config=param_config,
        tiebreaking_priority=tiebreaking_priority,
    )
    for column in HOMOGENIZED_COLUMNS:
        policy = homogenized_output[column]
        keep_auto = policy == "auto" and column in runtime_used_homogenized
        if column in df.columns and (policy == "always" or keep_auto):
            final_cols.append(column)

    # Normalize compared_to to nullable string if present.
    if "compared_to" in df.columns:
        df["compared_to"] = df["compared_to"].map_partitions(
            _normalize_string_series_to_na,
            meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
        )

    # Fast path: instrument_type <- type
    if used_type_fastpath and "type" in df.columns:
        df["instrument_type"] = df["type"].astype(DTYPE_STR)

    # Add tiebreaking columns if present and not already included.
    extra = [
        c
        for c in tiebreaking_priority
        if c not in final_cols and c in df.columns and c not in HOMOGENIZED_COLUMNS
    ]
    final_cols += extra

    # Collect variables referenced in YAML expressions (if requested).
    extra_expr_cols = set()
    if save_expr_columns:
        for ruleset in translation_rules_uc.values():
            for key in ["z_flag_translation", "instrument_type_translation"]:
                rule = ruleset.get(key, {})
                for cond in rule.get("conditions", []):
                    expr = cond.get("expr", "")
                    vars_in_expr = _extract_variables_from_expr(expr)
                    extra_expr_cols.update({v for v in vars_in_expr if v in df.columns})

    # Keep expression vars that are not already standard/final.
    standard = {"id", "ra", "dec", "z", "z_flag", "z_err", "instrument_type", "survey"}
    already = set(final_cols)
    needed = [c for c in extra_expr_cols if c not in standard and c not in already]
    if save_expr_columns:
        final_cols += needed

    # User-requested output columns are independent of translation and deduplication.
    extra_columns = extra_columns or {}
    df = _apply_configured_columns(df, extra_columns)
    final_cols += list(extra_columns)

    # Optional dtype coercions for expression vars (guided by schema_hints).
    schema_hints = schema_hints or {}
    if save_expr_columns and schema_hints:
        target_cols = [c for c in needed if c in schema_hints]
        for col in target_cols:
            kind = schema_hints[col]
            if col in df.columns:
                if kind == "str":
                    df[col] = df[col].map_partitions(
                        _normalize_string_series_to_na,
                        meta=pd.Series(pd.array([], dtype=DTYPE_STR)),
                    )
                elif kind == "float":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_FLOAT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_FLOAT)),
                    )
                elif kind == "int":
                    coerced = dd.to_numeric(df[col], errors="coerce")
                    df[col] = coerced.map_partitions(
                        lambda s: s.astype(DTYPE_INT),
                        meta=pd.Series(pd.array([], dtype=DTYPE_INT)),
                    )
                elif kind == "bool":
                    df[col] = df[col].map_partitions(
                        _to_nullable_boolean_strict,
                        meta=pd.Series(pd.array([], dtype=DTYPE_BOOL)),
                    )
            else:
                # Create missing columns with requested dtype.
                if kind == "str":
                    df = _add_missing_with_dtype(df, col, DTYPE_STR)
                elif kind == "float":
                    df = _add_missing_with_dtype(df, col, DTYPE_FLOAT)
                elif kind == "int":
                    df = _add_missing_with_dtype(df, col, DTYPE_INT)
                elif kind == "bool":
                    df = _add_missing_with_dtype(df, col, DTYPE_BOOL)

    # De-duplicate while preserving order, then filter to existing columns.
    final_cols = list(dict.fromkeys(final_cols))
    df = df[[c for c in final_cols if c in df.columns]]
    return df


# -----------------------
# Save parquet
# -----------------------
def _save_parquet(df: dd.DataFrame, temp_dir: str, product_name: str) -> str:
    """Write a partitioned Parquet artifact for the prepared catalog.

    Args:
        df: Prepared dataframe.
        temp_dir: Base temp directory.
        product_name: Internal name.

    Returns:
        str: Output directory path with Parquet files.
    """
    out_path = os.path.join(temp_dir, f"prepared_{product_name}")
    logger = logging.getLogger(LOGGER_NAME)

    df = _rename_duplicate_columns_dd(df, logger)
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        df.to_parquet(out_path, write_index=False, engine="pyarrow")
    return out_path


def _maybe_collection(
    out_path: str,
    logs_dir: str,
    logger: logging.Logger,
    client,
    combine_mode: str,
    *,
    schema_hints: dict | None = None,
    size_threshold_mb: int = 200,
) -> str:
    """Optionally build a Collection from prepared Parquet.

    Args:
        out_path: Prepared parquet path.
        logs_dir: Logs directory.
        logger: Logger.
        client: Dask client.
        combine_mode: Combine mode.
        schema_hints: Expr hints.
        size_threshold_mb: Fast-path threshold.

    Returns:
        str: Collection path or empty string for concatenate mode.
    """
    if combine_mode == "concatenate":
        return ""
    return _build_collection_with_retry(
        parquet_path=out_path,
        logs_dir=logs_dir,
        logger=logger,
        client=client,
        try_margin=True,
        schema_hints=schema_hints,
        size_threshold_mb=size_threshold_mb,
    )


# -----------------------
# Main orchestrator
# -----------------------
def _requires_z_flag_homogenization(combine_mode: str, cut_value: object) -> bool:
    """Whether preparation needs the semantic flag outside ranking priorities."""
    if combine_mode in {
        "concatenate_and_mark_duplicates",
        "concatenate_and_remove_duplicates",
    }:
        return True
    try:
        numeric_cut = float(cut_value)
    except (TypeError, ValueError):
        return False
    return numeric_cut in {1.0, 2.0, 3.0, 4.0}


def _active_instrument_type_filter(param_config: dict) -> bool:
    inclusion = validate_instrument_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_spectroscopic_ith",
                "include_grism_ith",
                "include_photometric_ith",
                "include_unclassified_ith",
            )
            if key in param_config
        }
    )
    return not all(inclusion.values())


def _filter_empty_result(
    df: dd.DataFrame,
    *,
    product_name: str,
    temp_dir: str,
    reason: str,
    detail_lines: list[str],
    logger: logging.LoggerAdapter | logging.Logger,
) -> tuple[dd.DataFrame, bool]:
    final_count = int(df.map_partitions(len).sum().compute())
    if final_count > 0:
        return df, False

    marker_path = os.path.join(temp_dir, f"prepared_{product_name}.empty")
    try:
        with open(marker_path, "w", encoding="utf-8") as fp:
            fp.write(f"{reason}\n")
            fp.write(f"product={product_name}\n")
            for line in detail_lines:
                fp.write(f"{line}\n")
            fp.write("rows_after=0\n")
    except Exception as e:
        logger.warning("Could not write empty-catalog marker %s: %s", marker_path, e)

    logger.warning(
        "[%s] Catalog is empty after preparation filters (%s); excluding it from "
        "subsequent HATS, crossmatch, and deduplication steps.",
        product_name,
        reason,
    )
    return df, True


def _apply_object_type_filter(
    df: dd.DataFrame,
    *,
    param_config: dict,
    product_name: str,
    temp_dir: str,
    logger: logging.LoggerAdapter | logging.Logger,
) -> tuple[dd.DataFrame, bool]:
    inclusion = validate_object_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_unclassified_oth",
                "include_galaxy_oth",
                "include_star_oth",
                "include_agn_oth",
                "include_qso_oth",
                "include_galactic_oth",
            )
            if key in param_config
        }
    )
    if all(inclusion.values()):
        return df, False

    normalized = df["object_type_homogenized"].astype("string").str.strip().str.lower()
    keep = normalized.map_partitions(
        lambda series: pd.Series(False, index=series.index),
        meta=pd.Series(dtype=bool),
    )
    if inclusion["include_unclassified_oth"]:
        keep = keep | normalized.isna()
    for object_type, key in _OBJECT_TYPE_TO_INCLUDE_KEY.items():
        if inclusion[key]:
            keep = keep | normalized.eq(object_type).fillna(False)

    out = df[keep]
    retained = int(out.map_partitions(len).sum().compute())
    logger.info(
        "[%s] Applied object_type_homogenized filter: %s rows retained; inclusion=%s",
        product_name,
        retained,
        inclusion,
    )
    return _filter_empty_result(
        out,
        product_name=product_name,
        temp_dir=temp_dir,
        reason="empty_after_object_type_homogenized_filter",
        detail_lines=[f"object_type_inclusion={inclusion}"],
        logger=logger,
    )


def _apply_instrument_type_filter(
    df: dd.DataFrame,
    *,
    param_config: dict,
    product_name: str,
    temp_dir: str,
    logger: logging.LoggerAdapter | logging.Logger,
) -> tuple[dd.DataFrame, bool]:
    inclusion = validate_instrument_type_inclusion(
        {
            key: param_config[key]
            for key in (
                "include_spectroscopic_ith",
                "include_grism_ith",
                "include_photometric_ith",
                "include_unclassified_ith",
            )
            if key in param_config
        }
    )
    if all(inclusion.values()):
        return df, False

    normalized = (
        df["instrument_type_homogenized"].astype("string").str.strip().str.lower()
    )
    keep = normalized.map_partitions(
        lambda series: pd.Series(False, index=series.index),
        meta=pd.Series(dtype=bool),
    )
    if inclusion["include_unclassified_ith"]:
        keep = keep | normalized.isna()
    for instrument_type, key in _INSTRUMENT_TYPE_TO_INCLUDE_KEY.items():
        if inclusion[key]:
            keep = keep | normalized.eq(instrument_type).fillna(False)

    out = df[keep]
    retained = int(out.map_partitions(len).sum().compute())
    logger.info(
        "[%s] Applied instrument_type_homogenized filter: %s rows retained; inclusion=%s",
        product_name,
        retained,
        inclusion,
    )
    return _filter_empty_result(
        out,
        product_name=product_name,
        temp_dir=temp_dir,
        reason="empty_after_instrument_type_homogenized_filter",
        detail_lines=[f"instrument_type_inclusion={inclusion}"],
        logger=logger,
    )


def _required_homogenized_columns(
    *,
    param_config: dict,
    tiebreaking_priority: list,
) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {column: [] for column in HOMOGENIZED_COLUMNS}
    priority_set = set(tiebreaking_priority or [])
    for column in HOMOGENIZED_COLUMNS:
        if column in priority_set:
            required[column].append("tiebreaking_priority")
    if _active_z_flag_filter(param_config):
        required["z_flag_homogenized"].append("z_flag_homogenized_value_to_cut")
    if _active_instrument_type_filter(param_config):
        required["instrument_type_homogenized"].append("instrument_type_filter")
    if _active_object_type_filter(param_config):
        required["object_type_homogenized"].append("object_type_filter")
    return {column: reasons for column, reasons in required.items() if reasons}


def _write_homogenized_metadata(
    df: dd.DataFrame,
    *,
    product_name: str,
    temp_dir: str,
    required_columns: dict[str, list[str]],
    logger: logging.LoggerAdapter | logging.Logger,
) -> dict:
    counts = {}
    for column in HOMOGENIZED_COLUMNS:
        counts[column] = int(df[column].count().compute()) if column in df.columns else 0

    metadata = {
        "product": product_name,
        "homogenized_non_null_counts": counts,
        "required_homogenized_columns": required_columns,
    }
    marker_path = os.path.join(temp_dir, f"prepared_{product_name}.homogenized.json")
    try:
        with open(marker_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)
    except Exception as e:
        logger.warning("Could not write homogenized metadata %s: %s", marker_path, e)

    for column, reasons in required_columns.items():
        if counts.get(column, 0) == 0:
            logger.warning(
                "[%s] Required homogenized column '%s' has no non-null values "
                "in this catalog; reasons=%s. The driver will fail if this is "
                "true for all input catalogs.",
                product_name,
                column,
                reasons,
            )
    return metadata


def validate_combine_configuration(
    combine_mode: object,
    tiebreaking_priority: object,
    cut_value: object,
    logger: logging.LoggerAdapter | logging.Logger | None = None,
) -> tuple[str, list[str]]:
    """Validate combine semantics before any catalog preparation starts."""
    normalized_mode = str(combine_mode or "").strip().lower()
    valid_modes = {
        "concatenate",
        "concatenate_and_mark_duplicates",
        "concatenate_and_remove_duplicates",
    }
    if normalized_mode not in valid_modes:
        raise ValueError(
            f"Invalid combine_type={combine_mode!r}; expected one of "
            f"{sorted(valid_modes)}"
        )

    if tiebreaking_priority is None:
        priorities: list[str] = []
    elif isinstance(tiebreaking_priority, (list, tuple)):
        priorities = [str(value).strip() for value in tiebreaking_priority]
    else:
        raise TypeError("tiebreaking_priority must be a list of column names")
    if any(not value for value in priorities):
        raise ValueError("tiebreaking_priority cannot contain empty column names")
    if len(set(priorities)) != len(priorities):
        raise ValueError("tiebreaking_priority cannot contain duplicate columns")
    if normalized_mode != "concatenate" and not priorities:
        raise ValueError(
            f"tiebreaking_priority must be non-empty for {normalized_mode}"
        )

    return normalized_mode, priorities


def _normalize_custom_tiebreaking_priorities(
    df: dd.DataFrame,
    priorities: list[str],
    product_name: str,
    logger: logging.LoggerAdapter | logging.Logger,
) -> dd.DataFrame:
    """Require and normalize generic ranking columns for one input catalog."""
    custom_priorities = [
        column
        for column in priorities
        if column
        not in {"z_flag_homogenized", "instrument_type_homogenized"}
    ]
    for column in custom_priorities:
        if column not in df.columns:
            raise ValueError(
                f"[{product_name}] Custom tiebreaking priority '{column}' is "
                "missing; custom priorities must exist in every input catalog"
            )

        original = df[column]
        numeric = dd.to_numeric(original, errors="coerce")
        total_rows, original_non_null, valid_numeric, coerced_invalid = dask.compute(
            df.map_partitions(len).sum(),
            (~original.isna()).sum(),
            (~numeric.isna()).sum(),
            ((~original.isna()) & numeric.isna()).sum(),
        )
        total_rows = int(total_rows)
        original_non_null = int(original_non_null)
        valid_numeric = int(valid_numeric)
        coerced_invalid = int(coerced_invalid)
        if valid_numeric == 0:
            raise ValueError(
                f"[{product_name}] Custom tiebreaking priority '{column}' has "
                "no valid numeric values"
            )
        invalid_fraction = (
            coerced_invalid / original_non_null if original_non_null else 0.0
        )
        logger.info(
            "[%s] Custom priority '%s': rows=%d valid_numeric=%d "
            "missing=%d coerced_invalid=%d invalid_fraction=%.6f; "
            "normalized_dtype=float64",
            product_name,
            column,
            total_rows,
            valid_numeric,
            total_rows - original_non_null,
            coerced_invalid,
            invalid_fraction,
        )
        df = df.assign(**{column: numeric.astype("float64")})
    return df


def prepare_catalog(
    entry: dict,
    translation_config: dict,
    param_config: dict,
    logs_dir: str,
    temp_dir: str,
    combine_mode: str = "concatenate_and_mark_duplicates",
) -> tuple[str, str, str, str, str]:
    """Prepare a spectroscopic catalog for the CRC pipeline.

    Args:
        entry: Product descriptor with keys like {"path", "internal_name", ...}.
        translation_config: YAML-derived rules.
        logs_dir: Logs directory.
        temp_dir: Temp workspace for artifacts.
        combine_mode: Combine strategy (kept for compatibility).

    Returns:
        Tuple[str, str, str, str, str]: (collection_path, "ra", "dec", internal_name, "").
    """
    # Ensure central logger in this process (worker-safe)
    try:
        ensure_crc_logger(logs_dir)
    except Exception:
        pass

    try:
        client = _get_client()
    except Exception:
        client = None

    product_name = entry["internal_name"]
    base_logger = _get_logger()
    lg = _phase_logger(base_logger, phase="preparation", product=product_name)

    # ===== START PHASE (per catalog) =====
    lg.info(f"START prepare_catalog product={product_name}")

    extra_columns = _normalize_extra_columns_config(param_config.get("extra_columns"))

    z_flag_homogenized_value_to_cut = param_config.get(
        "z_flag_homogenized_value_to_cut", 0
    )

    # 1) Load product
    ph = ProductHandle(entry["path"])
    df = ph.to_ddf()
    entry = _prefer_pipeline_output_id_mapping(entry, list(df.columns), lg)
    df = _copy_extra_columns_from_sources(df, extra_columns, lg)

    # 2) Validate & rename, base schema
    df = _validate_and_rename(df, entry, lg)
    df = _drop_previous_results(df, lg)

    # 3) Honor user-provided homogenized columns
    df = _honor_user_homogenized_mapping(df, entry, product_name, lg)

    # 4) Normalize types and values
    df, type_cast_ok = _normalize_types(df, product_name, lg)

    # 5) Assign CRD_IDs
    df = _generate_crd_ids(df, product_name, temp_dir, client=client)
    if bool(translation_config.get("validate_crd_id_uniqueness", False)):
        _validate_unique_crd_ids(df, product_name, lg)

    # 6) Homogenized fields
    (
        df,
        used_type_fastpath,
        tiebreaking_priority,
        instrument_type_priority,  # noqa: F841
        translation_rules_uc,
    ) = _homogenize(
        df,
        translation_config,
        product_name,
        lg,
        type_cast_ok=type_cast_ok,
        require_z_flag_homogenized=_requires_z_flag_homogenization(
            combine_mode,
            z_flag_homogenized_value_to_cut,
        ),
        require_instrument_type_homogenized=_active_instrument_type_filter(
            param_config
        ),
        require_object_type_homogenized=_active_object_type_filter(param_config),
    )
    df = _normalize_custom_tiebreaking_priorities(
        df,
        list(tiebreaking_priority),
        product_name,
        lg,
    )
    required_homogenized = _required_homogenized_columns(
        param_config=param_config,
        tiebreaking_priority=list(tiebreaking_priority),
    )
    _write_homogenized_metadata(
        df,
        product_name=product_name,
        temp_dir=temp_dir,
        required_columns=required_homogenized,
        logger=lg,
    )

    # 7) Apply cut based on z_flag_homogenized if requested
    if (
        z_flag_homogenized_value_to_cut is not None
        and "z_flag_homogenized" in df.columns
    ):
        try:
            cut_val = float(z_flag_homogenized_value_to_cut)
        except (TypeError, ValueError):
            lg.warning(
                "Invalid z_flag_homogenized_value_to_cut=%s; skipping cut.",
                z_flag_homogenized_value_to_cut,
            )
        else:
            if cut_val == 0.0:
                # Zero is the explicit no-cut sentinel. It is summarized once
                # by the driver instead of repeated for every input catalog.
                pass
            elif cut_val not in {1.0, 2.0, 3.0, 4.0}:
                lg.warning(
                    "Invalid z_flag_homogenized_value_to_cut=%s; use 0 to disable "
                    "the cut or one of 1, 2, 3, 4. Skipping cut.",
                    z_flag_homogenized_value_to_cut,
                )
            else:
                df = df[df["z_flag_homogenized"] >= cut_val]
                final_count = int(df.map_partitions(len).sum().compute())
                lg.info(
                    "Applied z_flag_homogenized cut >= %s: %s rows retained",
                    cut_val,
                    final_count,
                )
                _, empty_after_filter = _filter_empty_result(
                    df,
                    product_name=product_name,
                    temp_dir=temp_dir,
                    reason="empty_after_z_flag_homogenized_cut",
                    detail_lines=[f"z_flag_homogenized_value_to_cut={cut_val}"],
                    logger=lg,
                )
                if empty_after_filter:
                    lg.info(
                        f"END prepare_catalog product={product_name} empty_after_cut"
                    )
                    return "", "ra", "dec", product_name, "empty_after_cut"

    df, empty_after_filter = _apply_instrument_type_filter(
        df,
        param_config=param_config,
        product_name=product_name,
        temp_dir=temp_dir,
        logger=lg,
    )
    if empty_after_filter:
        lg.info(f"END prepare_catalog product={product_name} empty_after_filter")
        return "", "ra", "dec", product_name, "empty_after_filter"

    df, empty_after_filter = _apply_object_type_filter(
        df,
        param_config=param_config,
        product_name=product_name,
        temp_dir=temp_dir,
        logger=lg,
    )
    if empty_after_filter:
        lg.info(f"END prepare_catalog product={product_name} empty_after_filter")
        return "", "ra", "dec", product_name, "empty_after_filter"

    # 8) Optional persist + repartition. Sizing by bytes requires another full
    # pass, so production keeps the existing partitions unless explicitly asked.
    if _as_bool_config(
        translation_config.get("repartition_prepared_catalogs", False),
        default=False,
    ):
        part_size = str(
            translation_config.get("prepared_partition_size", "256MB") or "256MB"
        )
        try:
            if client is not None:
                df = df.persist()
                wait(df)
            with dask.config.set({"dataframe.shuffle.method": "tasks"}):
                df = df.repartition(partition_size=part_size)
            lg.info(
                "Persisted and repartitioned: partition_size=%s npartitions=%s",
                part_size,
                df.npartitions,
            )
        except Exception as e:
            lg.warning("Persist/repartition skipped or failed: %s", e)
    else:
        lg.info("Prepared-catalog repartition disabled; preserving current partitions.")

    # 9) Init compared_to/tie_result
    df = _add_missing_with_dtype(df, "compared_to", DTYPE_STR)
    df = _add_missing_with_dtype(df, "tie_result", DTYPE_INT8)
    df["tie_result"] = df["tie_result"].map_partitions(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(1).astype(DTYPE_INT8),
        meta=pd.Series(pd.array([], dtype=DTYPE_INT8)),
    )

    # 10) Geometry validation
    _validate_ra_dec_or_fail(df, product_name)

    # 11) Footprint flags
    insert_dp1 = _as_bool_config(
        param_config.get("insert_DP1_footprint_flag", False), default=False
    )
    insert_rubin = _as_bool_config(
        param_config.get("insert_rubin_footprint_flag", False), default=False
    )

    if insert_dp1:
        df = _flag_dp1(df)
    if insert_rubin:
        df = _flag_rubin_footprint(df)

    # 12) Assemble final columns
    df = _select_output_columns(
        df,
        translation_rules_uc,
        tiebreaking_priority,
        used_type_fastpath,
        param_config,
        save_expr_columns=translation_config.get("save_expr_columns", False),
        schema_hints=_normalize_schema_hints(
            translation_config.get("expr_column_schema")
        ),
        extra_columns=extra_columns,
    )

    # Coalesce partitions for write
    try:
        if client is not None:
            nworkers = max(len(client.scheduler_info().get("workers", {})), 1)
        else:
            nworkers = 1
    except Exception:
        nworkers = 1
    target_out_parts = max(2 * nworkers, 8)
    try:
        with dask.config.set({"dataframe.shuffle.method": "tasks"}):
            df_to_write = df.repartition(npartitions=target_out_parts)
        lg.info(
            f"Output repartitioned for write: npartitions={df_to_write.npartitions}"
        )
    except Exception as e:
        lg.warning(
            f"Output repartition failed; writing current npartitions. Reason: {e}"
        )
        df_to_write = df

    # 13) Write prepared parquet
    out_path = _save_parquet(df_to_write, temp_dir, product_name)

    # 14) Build collection (always)
    schema_hints_raw = dict(translation_config.get("expr_column_schema") or {})
    schema_hints_raw.update(
        {output: spec["type"] for output, spec in extra_columns.items()}
    )
    collection_path = _build_collection_with_retry(
        parquet_path=out_path,
        logs_dir=logs_dir,
        logger=lg,
        client=client,
        try_margin=True,
        schema_hints=schema_hints_raw,
        size_threshold_mb=200,
        margin_threshold=float(translation_config.get("margin_threshold_arcsec", 5.0)),
    )

    # ===== END PHASE (per catalog) =====
    lg.info(f"END prepare_catalog product={product_name} path={collection_path}")

    return collection_path, "ra", "dec", product_name, ""
