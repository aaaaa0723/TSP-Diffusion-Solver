"""Bounded retries for read-only OSRM/Overpass requests."""
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def configure_session(session):
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=.5,
                  status_forcelist=(429, 502, 503, 504), allowed_methods=('GET', 'POST'))
    for scheme in ('http://', 'https://'):
        session.mount(scheme, HTTPAdapter(max_retries=retry))
    return session
