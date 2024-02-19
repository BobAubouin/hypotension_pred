from typing import TypedDict

STATIC_DATA_NAMES = ["age", "bmi", "asa", "preop_cr", "preop_htn", "opname"]

SAMPLING_TIME = 1


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
            "RR",
            "PLETH_SPO2",
            "ETCO2",
        ],
    ),
    TrackConfig(name="Orchestra", tracks=["PPF20_CT"]),
    TrackConfig(name="Primus", tracks=["MAC"]),
]

DEVICE_NAME_TO_SAMPLING_RATE = {
    "Solar8000": 2,
    "Primus": 7,
}
