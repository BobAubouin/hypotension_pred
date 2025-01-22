import numpy as np
from typing import TypedDict

STATIC_DATA_NAMES = ["age", "bmi", "asa", "preop_cr", "preop_htn", "opname"]
STATIC_NAME_TO_DTYPES = {
    "age": np.uint16,
    "bmi": np.float16,
    "preop_cr": np.float32,
    "asa": np.uint16,
    "preop_htn": np.uint16,
    "opname": "category",
}

SAMPLING_TIME = 2


class TrackConfig(TypedDict):
    name: str
    tracks: list[str]


TRACKS_CONFIG = [
    TrackConfig(
        name="Solar8000",
        tracks=[
            "ART_MBP",
            "ART_SBP",
            "ART_DBP",
            "HR",
            "RR_CO2",
            "PLETH_SPO2",
            "ETCO2",
            "BT",
        ],
    ),
    TrackConfig(
        name="Orchestra",
        tracks=["PPF20_CT",
                "RFTN20_CT",
                "VASO_RATE",
                "PHEN_RATE",
                "NEPI_RATE",
                "EPI_RATE",
                "DOPA_RATE",
                "DOBU_RATE",
                "DTZ_RATE",
                "NTG_RATE",
                "NPS_RATE",
                ],
    ),
    TrackConfig(name="Primus", tracks=["MAC"]),
    # TrackConfig(name="SNUADC", tracks=["ART"]),
    TrackConfig(name="BIS", tracks=["BIS"]),
]

DEVICE_NAME_TO_SAMPLING_RATE = {
    "Solar8000": 2,
    "Primus": 7,
    "BIS": 1,
}
