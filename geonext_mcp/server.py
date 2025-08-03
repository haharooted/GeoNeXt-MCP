"""
GeoNeXt-MCP
"""

from __future__ import annotations

import inspect
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, TypedDict, Union

from dotenv import load_dotenv
from fastmcp import FastMCP
from geopy.distance import distance as geodistance
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import (
    Nominatim,
    ArcGIS,
    Bing,
    Photon,
    GoogleV3,
    Pelias,
    MapBox,
)

###############################################################################
# Environment & logging
###############################################################################
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "geonext-mcp.log")
_MAX_ROLLS = int(os.getenv("GEOCODER_MAX_ROLLS", "2"))

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=LOG_LEVEL,
    handlers=[
        logging.StreamHandler(),
        *( [logging.FileHandler(LOG_FILE, encoding="utf-8")] if LOG_FILE else [] ),
    ],
    force=True,  # override any existing configuration – important in notebooks / reloads
)

logger = logging.getLogger("geonext-mcp")
logger.debug("Logging initialised – level=%s, file=%s", LOG_LEVEL, LOG_FILE)

###############################################################################
# FastMCP server
###############################################################################
mcp = FastMCP("GeoNeXt‑MCP", dependencies=["geopy"])

###############################################################################
# Geocoder factory helpers
###############################################################################
Provider = Photon | Nominatim | ArcGIS | Bing | GoogleV3 | Pelias | MapBox


def _safe_geocode(geo: Provider, query: str, **extra):
    """Call ``geo.geocode`` but drop kwargs the provider doesn’t accept."""
    sig = inspect.signature(geo.geocode)
    accepted = {k: v for k, v in extra.items() if k in sig.parameters}
    logger.debug("_safe_geocode(%s) accepted kwargs: %s", geo.__class__.__name__, accepted)
    return geo.geocode(query, **accepted)


def _build_geocoder(provider: str | None = None) -> Provider:
    provider = (provider or os.getenv("GEOCODER_PROVIDER", "photon")).lower()
    logger.debug("Building geocoder for provider=%s", provider)

    if provider == "nominatim":
        geo = Nominatim(
            user_agent=os.getenv("NOMINATIM_USER_AGENT", "geonext-mcp/0.4.0"),
            domain=os.getenv("NOMINATIM_URL", "nominatim.openstreetmap.org"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=60,
        )

    elif provider == "photon":
        geo = Photon(
            user_agent=os.getenv("PHOTON_USER_AGENT", "geonext-mcp/0.4.0"),
            domain=os.getenv("PHOTON_URL", "photon.komoot.io"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=10,
        )

    elif provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required when provider=google")
        geo = GoogleV3(api_key=key, timeout=10)

    elif provider == "bing":
        key = os.getenv("BING_API_KEY")
        if not key:
            raise RuntimeError("BING_API_KEY is required when provider=bing")
        geo = Bing(api_key=key, timeout=10)

    elif provider == "arcgis":
        geo = ArcGIS(
            username=os.getenv("ARC_USERNAME"),
            password=os.getenv("ARC_PASSWORD"),
            referer=os.getenv("ARC_REFERER"),
            timeout=10,
        )

    elif provider in {"pelias", "geocodeearth", "geocode_earth"}:
        key = os.getenv("PELIAS_API_KEY")
        if not key:
            raise RuntimeError("PELIAS_API_KEY is required when provider=pelias")
        geo = Pelias(
            api_key=key,
            domain=os.getenv("PELIAS_URL", "api.geocode.earth"),
            scheme=os.getenv("SCHEME", "https"),
            user_agent=os.getenv("PELIAS_USER_AGENT", "geonext-mcp/0.4.0"),
            timeout=10,
        )

    elif provider == "mapbox":
        key = os.getenv("MAPBOX_API_KEY")
        if not key:
            raise RuntimeError("MAPBOX_API_KEY is required when provider=mapbox")
        geo = MapBox(
            api_key=key,
            user_agent=os.getenv("MAPBOX_USER_AGENT", "geonext-mcp/0.4.0"),
            timeout=10,
        )

    else:
        raise ValueError(f"Unsupported geocoder provider: {provider!r}")

    logger.debug("Created %s geocoder: %s", provider, geo)
    return geo

###############################################################################
# Per‑provider throttle policy
###############################################################################
_PROVIDER_POLICY: dict[str, dict[str, float | int]] = {
    # "photon": dict(delay=1.0, retries=2, err_wait=2.0),
    # "nominatim": dict(delay=1.0, retries=2, err_wait=2.0),
    # "pelias": dict(delay=0.3, retries=2, err_wait=1.5),
    # "mapbox": dict(delay=0.15, retries=2, err_wait=1.0),
    # "google": dict(delay=0.1, retries=2, err_wait=1.0),
    # "bing": dict(delay=0.1, retries=2, err_wait=1.0),
    # "arcgis": dict(delay=0.2, retries=2, err_wait=1.0),
    "photon":    dict(delay=1.0),
    "nominatim": dict(delay=1.0),
    "pelias":    dict(delay=0.3),
    "mapbox":    dict(delay=0.15),
    "google":    dict(delay=0.1),
    "bing":      dict(delay=0.1),
    "arcgis":    dict(delay=0.2),
}
logger.debug("Provider policy: %s", _PROVIDER_POLICY)

###############################################################################
# Rate‑Limiter cache
###############################################################################
_geocode_cache: dict[str, RateLimiter] = {}


def _rate_limited_geocode_for(provider: str) -> RateLimiter:
    provider = provider.lower()
    if provider not in _geocode_cache:
        policy = _PROVIDER_POLICY.get(provider, {"delay": 1.0})
        delay    = policy["delay"]
        err_wait = policy.get("err_wait", delay)

        _geocode_cache[provider] = RateLimiter(
            partial(_safe_geocode, _build_geocoder(provider)),
            min_delay_seconds=delay,
            max_retries=0,
            error_wait_seconds=err_wait,
            swallow_exceptions=False,
        )
    return _geocode_cache[provider]


###############################################################################
# GeoResult type (with provider field)
###############################################################################
class GeoResult(TypedDict, total=False):
    provider: str            # which backend produced the hit
    latitude: float
    longitude: float
    address: str
    details: Dict[str, Any]
    bounding_box: List[str]
    raw: Dict[str, Any]      # provider JSON

###############################################################################
# Provider cascade helper
###############################################################################
_DEFAULT_CHAIN = ["photon", "nominatim", "pelias", "mapbox", "google"]


def _geocode_with_chain(
    query: str,
    chain: list[str],
    max_results: int,
) -> List[GeoResult]:
    """
    Try up to `_MAX_ROLLS` providers **only when the current one raises
    an exception**.  
    If a provider returns an empty result set, that is considered a
    *successful* answer and we stop there.
    """
    errors: list[str] = []

    for rolls, prov in enumerate(chain):
        if rolls >= _MAX_ROLLS:
            break

        try:
            # Will raise on timeout / HTTP error, otherwise always returns
            hits = _geocode_single_provider(query, max_results, prov)
            return hits               # even if `hits` is [], we are done
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            errors.append(f"{prov}: {exc}")
            continue                  # roll to the next provider

    logger.warning("All attempted providers errored for %r – %s", query, " | ".join(errors))
    return []                         # every try raised an exception



def _geocode_single_provider(
    location: str,
    max_results: int,
    provider: str,
) -> List[GeoResult]:
    logger.debug("_geocode_single_provider(query=%s, provider=%s, max_results=%d)", location, provider, max_results)
    geo_fn = _rate_limited_geocode_for(provider)

    raw_locs = geo_fn(
        location,
        exactly_one=False,
        limit=max_results,
        addressdetails=True,
    )

    results: List[GeoResult] = []
    for loc in (raw_locs or [])[:max_results]:
        r = loc.raw or {}
        results.append(
            GeoResult(
                provider=provider,
                latitude=loc.latitude,
                longitude=loc.longitude,
                address=loc.address,
                details=r.get("address", {}),
                bounding_box=r.get("boundingbox", []),
                raw=r,
            )
        )
    logger.debug("%s returned %d result(s) for %r", provider, len(results), location)
    return results

###############################################################################
# FastMCP tools
###############################################################################
@mcp.tool()
def geocode_location(
    location: str,
    max_results: int = 5,
    provider: str = "auto",
) -> List[GeoResult]:
    """
    Geocode an address / place string.

    * If ``provider="auto"`` (default) If it gets an error from the server it will try the chain
      Photon → Nominatim → Pelias → Mapbox → Google until something returns a) an empty list - no results or b) results
    * Otherwise it queries the specified backend directly.
    """
    provider = provider.lower()
    logger.info(
        "geocode_location(location=%r, max_results=%d, provider=%s)",
        location,
        max_results,
        provider,
    )
    try:
        if provider == "auto":
            return _geocode_with_chain(location, _DEFAULT_CHAIN, max_results)
        return _geocode_single_provider(location, max_results, provider)
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.error("geocode_location error (%s): %s", provider, exc)
        return []


@mcp.tool()
def reverse_geocode(lat: float, lon: float, provider: str = "photon") -> Optional[GeoResult]:
    """
    Reverse‑geocode a lat/lon pair.  
    ``provider`` is exposed mainly for testing; defaults to Photon.
    """
    logger.info("reverse_geocode(lat=%s, lon=%s, provider=%s)", lat, lon, provider)
    rev_fn = _build_geocoder(provider).reverse
    limiter = RateLimiter(
        rev_fn,
        min_delay_seconds=_PROVIDER_POLICY.get(provider, {}).get("delay", 1.0),
        swallow_exceptions=False,
    )

    try:
        loc = limiter((lat, lon), addressdetails=True)
        if not loc:
            logger.debug("No reverse‑geocode hit for %s,%s via %s", lat, lon, provider)
            return None
        raw = loc.raw or {}
        result = GeoResult(
            provider=provider,
            latitude=lat,
            longitude=lon,
            address=loc.address,
            details=raw.get("address", {}),
            bounding_box=raw.get("boundingbox", []),
            raw=raw,
        )
        logger.debug("Reverse‑geocode result: %s", result)
        return result
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.error("reverse_geocode error (%s): %s", provider, exc)
        return None


@mcp.tool()
def geocode_with_details(location: str, provider: str = "photon") -> Optional[GeoResult]:
    """Single best match with extra address details & bounding box."""
    logger.info("geocode_with_details(location=%r, provider=%s)", location, provider)
    geo = _build_geocoder(provider)
    try:
        loc = geo.geocode(location, addressdetails=True)
        if not loc:
            logger.debug("No detailed geocode hit for %r via %s", location, provider)
            return None
        raw = loc.raw or {}
        result = GeoResult(
            provider=provider,
            latitude=loc.latitude,
            longitude=loc.longitude,
            address=loc.address,
            details=raw.get("address", {}),
            bounding_box=raw.get("boundingbox", []),
            raw=raw,
        )
        logger.debug("Detailed geocode result: %s", result)
        return result
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.error("geocode_with_details error (%s): %s", provider, exc)
        return None

###############################################################################
# Bulk helpers still use provider cascade
###############################################################################
@mcp.tool()
def geocode_locations(
    locations: Union[str, List[str]],
    max_results: int = 5,
    provider: str = "auto",
) -> Union[List[GeoResult], List[List[GeoResult]]]:
    """
    Geocode one **or many** location strings. Uses the same provider logic
    as ``geocode_location``.
    """
    logger.info("geocode_locations(%r, provider=%s)", locations, provider)

    queries = [locations] if isinstance(locations, str) else locations
    all_results: List[List[GeoResult]] = []

    for query in queries:
        logger.debug("Bulk geocode query=%r", query)
        all_results.append(geocode_location(query, max_results, provider))

    return all_results[0] if isinstance(locations, str) else all_results

###############################################################################
# Legacy helpers
###############################################################################
@mcp.tool()
def geocode_multiple_locations(locations: List[str]) -> List[Optional[GeoResult]]:
    """Bulk geocode; **first candidate only** per query (legacy)."""
    logger.info("geocode_multiple_locations(len=%d)", len(locations))
    results: List[Optional[GeoResult]] = []
    for loc in locations:
        logger.debug("Legacy bulk geocode query=%r", loc)
        matches = geocode_location(loc, max_results=1)
        results.append(matches[0] if matches else None)
    return results


@mcp.tool()
def reverse_geocode_multiple_locations(coords: List[List[float]]) -> List[Optional[GeoResult]]:
    """Bulk reverse‑geocode a list of [lat, lon] pairs."""
    logger.info("reverse_geocode_multiple_locations(len=%d)", len(coords))
    return [
        reverse_geocode(lat, lon) if len(pair) == 2 else None
        for pair in coords
        for lat, lon in [pair]
    ]


###############################################################################
# Distance helpers (unchanged, but log inputs/outputs)
###############################################################################
@mcp.tool(name="distance_between_coords")
def _distance_between_coords(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    unit: str = "kilometers",
) -> float:
    """Great‑circle distance between two lat/lon points."""
    logger.debug(
        "distance_between_coords(%s,%s -> %s,%s, unit=%s)", lat1, lon1, lat2, lon2, unit
    )
    dist = geodistance((lat1, lon1), (lat2, lon2))
    res = dist.miles if unit.lower().startswith("mile") else dist.kilometers
    logger.debug("Computed distance: %s %s", res, unit)
    return res


@mcp.tool(name="distance_between_addresses")
def distance_between_addresses(
    address1: str,
    address2: str,
    unit: str = "kilometers",
) -> Optional[float]:
    """Distance between two address strings (uses first match for each)."""
    logger.info("distance_between_addresses(%r, %r, unit=%s)", address1, address2, unit)
    loc1 = geocode_location(address1, max_results=1)
    loc2 = geocode_location(address2, max_results=1)
    if not (loc1 and loc2):
        logger.warning("Could not geocode one or both addresses: %s | %s", loc1, loc2)
        return None
    p1, p2 = loc1[0], loc2[0]
    return _distance_between_coords(
        p1["latitude"],
        p1["longitude"],
        p2["latitude"],
        p2["longitude"],
        unit,
    )


###############################################################################
# Entrypoint
###############################################################################

def main() -> None:
    logger.info("Starting GeoNeXt‑MCP server…")
    mcp.run(
        transport="sse",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
