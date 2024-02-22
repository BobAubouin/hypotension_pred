import asyncio
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from aiohttp import ClientSession, ClientTimeout, ClientError
from tqdm.asyncio import tqdm

from hp_pred.constants import VITAL_API_BASE_URL

TIMEOUT = ClientTimeout(total=None, sock_connect=10, sock_read=10)
MAX_TRIES = 3


async def _retrieve_csv_text(track_url: str, session: ClientSession) -> str:
    attempt = 0
    client_error: ClientError

    while attempt < MAX_TRIES:
        try:
            async with session.get(track_url) as response:
                return await response.text()
        except ClientError as e:
            client_error = e

    raise ClientError(client_error)


def _read_csv_sync(csv_text: str, case_id: int) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_text), na_values="-nan(ind)").assign(
        caseid=case_id
    )


async def _read_csv_async(
    track_url: str, case_id: int, executor: ThreadPoolExecutor, session: ClientSession
) -> pd.DataFrame:
    csv_text = await _retrieve_csv_text(track_url, session)

    loop = asyncio.get_running_loop()
    track_raw_data = await loop.run_in_executor(
        executor, _read_csv_sync, csv_text, case_id
    )

    return track_raw_data


async def retrieve_tracks_raw_data_async(
    tracks_url_and_case_id: list[tuple[str, int]]
) -> list[pd.DataFrame]:

    async with ClientSession(base_url=VITAL_API_BASE_URL, timeout=TIMEOUT) as session:
        with ThreadPoolExecutor() as executor:
            read_tasks = [
                _read_csv_async(track_url, case_id, executor, session)
                for track_url, case_id in tracks_url_and_case_id
            ]
            tracks_raw_data: list[pd.DataFrame] = []
            for read_csv_task in tqdm.as_completed(read_tasks, mininterval=1):
                track_url_and_raw_data = await read_csv_task
                tracks_raw_data.append(track_url_and_raw_data)

    return tracks_raw_data
