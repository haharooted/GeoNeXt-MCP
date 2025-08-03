from __future__ import annotations

import logging
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
    overload,            # still imported for future‑proofing
)

from dotenv import load_dotenv
from fastmcp import FastMCP
import inspect
from functools import partial
from geopy.distance import distance as geodistance
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import (
    Nominatim,
    ArcGIS,
    Bing,
    Photon,
    GoogleV3,
    Pelias,          # NEW
    MapBox,          # NEW
)


_PROVIDER_POLICY = {
    "photon":     dict(delay=1.0, retries=2,  err_wait=2.0),
    "nominatim":  dict(delay=1.0, retries=2,  err_wait=2.0),
    "pelias":     dict(delay=0.3, retries=2,  err_wait=1.0),
    "mapbox":     dict(delay=0.3, retries=2, err_wait=1.0),
    "google":     dict(delay=0.3, retries=2,  err_wait=1.0),
}

def _safe_geocode(geo, query: str, **extra):
    """
    Call ``geo.geocode`` but drop any kwargs the provider does not accept.
    """
    sig = inspect.signature(geo.geocode)
    accepted = {k: v for k, v in extra.items() if k in sig.parameters}
    return geo.geocode(query, **accepted)

load_dotenv()

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("geonext-mcp")

# ── optional file output ────────────────────────────────────────────────────
log_path = os.getenv("LOG_FILE", "geonext-mcp.log")
if log_path:
    _fh = logging.FileHandler(log_path, encoding="utf-8")
    _fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    )
    logger.addHandler(_fh)

###############################################################################
# FastMCP server instance
###############################################################################
mcp = FastMCP(
    "GeoNeXt‑MCP",
    dependencies=["geopy"],
)

###############################################################################
# Geocoder factory
###############################################################################
Provider = Photon | Nominatim | ArcGIS | Bing | GoogleV3 | Pelias | MapBox

def _build_geocoder(provider: str | None = None) -> Provider:
    """
    Return a configured *synchronous* geocoder instance.

    Parameters
    ----------
    provider : str | None
        Name of the backend. If *None*, falls back to the
        ``GEOCODER_PROVIDER`` environment variable or **photon**.
    """
    provider = (provider or os.getenv("GEOCODER_PROVIDER", "photon")).lower()

    if provider == "nominatim":
        return Nominatim(
            user_agent=os.getenv("NOMINATIM_USER_AGENT", "geonext-mcp/0.3.0"),
            domain=os.getenv("NOMINATIM_URL", "nominatim.openstreetmap.org"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=60,
        )

    if provider == "photon":
        return Photon(
            user_agent=os.getenv("PHOTON_USER_AGENT", "geonext-mcp/0.3.0"),
            domain=os.getenv("PHOTON_URL", "photon.komoot.io"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=10,
        )

    if provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required when provider=google")
        return GoogleV3(api_key=key, timeout=10)

    if provider == "bing":
        key = os.getenv("BING_API_KEY")
        if not key:
            raise RuntimeError("BING_API_KEY is required when provider=bing")
        return Bing(api_key=key, timeout=10)

    if provider == "arcgis":
        return ArcGIS(
            username=os.getenv("ARC_USERNAME"),
            password=os.getenv("ARC_PASSWORD"),
            referer=os.getenv("ARC_REFERER"),
            timeout=10,
        )

    if provider in {"pelias", "geocodeearth", "geocode_earth"}:
        key = os.getenv("PELIAS_API_KEY")
        if not key:
            raise RuntimeError("PELIAS_API_KEY is required when provider=pelias")
        return Pelias(                               # see geopy docs §Pelias :contentReference[oaicite:0]{index=0}
            api_key=key,
            domain=os.getenv("PELIAS_URL", "api.geocode.earth"),
            scheme=os.getenv("SCHEME", "https"),
            user_agent=os.getenv("PELIAS_USER_AGENT", "geonext-mcp/0.3.0"),
            timeout=10,
        )

    if provider == "mapbox":
        key = os.getenv("MAPBOX_API_KEY")
        if not key:
            raise RuntimeError("MAPBOX_API_KEY is required when provider=mapbox")
        return MapBox(                              # see geopy docs §MapBox  :contentReference[oaicite:1]{index=1}
            api_key=key,
            user_agent=os.getenv("MAPBOX_USER_AGENT", "geonext-mcp/0.3.0"),
            timeout=10,
        )

    raise ValueError(f"Unsupported geocoder provider: {provider!r}")

# ──────────────────────────────────────────────────────────────────────────────
# Global *default* geocoder (still Photon unless env overrides)
# ──────────────────────────────────────────────────────────────────────────────
geocoder: Provider = _build_geocoder()
min_delay = float(os.getenv("GEOCODER_MIN_DELAY", "1.0"))

geocode = RateLimiter(partial(_safe_geocode, geocoder), min_delay_seconds=min_delay)
reverse = RateLimiter(geocoder.reverse,             min_delay_seconds=min_delay)

_DEFAULT_CHAIN = ["photon", "nominatim", "pelias", "mapbox", "google"]

def _geocode_with_chain(query: str,
                        chain: list[str],
                        max_results: int) -> list[GeoResult]:
    errors: list[str] = []
    for prov in chain:
        try:
            hits = geocode_location(query, max_results=max_results, provider=prov)
            if hits:                              # success – use them!
                return hits
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            errors.append(f"{prov}: {exc}")
    logger.warning("All providers failed for %r – %s", query, " | ".join(errors))
    return []

# ──────────────────────────────────────────────────────────────────────────────
# Per‑provider Rate‑Limiter cache (used only by geocode_location)
# ──────────────────────────────────────────────────────────────────────────────
_geocode_cache: dict[str, RateLimiter] = {}

def _rate_limited_geocode_for(provider: str) -> RateLimiter:
    provider = provider.lower()
    if provider not in _geocode_cache:
        policy = _PROVIDER_POLICY.get(provider, {"delay": 1.0, "retries": 2, "err_wait": 2.0})
        _geocode_cache[provider] = RateLimiter(
            partial(_safe_geocode, _build_geocoder(provider)),
            min_delay_seconds   = policy["delay"],
            max_retries         = policy["retries"],
            error_wait_seconds  = policy["err_wait"],
            swallow_exceptions  = False,           # <-- propagate!
        )
    return _geocode_cache[provider]


###############################################################################
# Typed return payloads
###############################################################################
class GeoResult(TypedDict, total=False):
    provider: str
    latitude: float
    longitude: float
    address: str
    details: Dict[str, Any]
    bounding_box: List[str]
    raw: Dict[str, Any]

# ──────────────────────────────────────────────────────────────────────────────
# geocode_location – provider now selectable at call‑time
# ──────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def geocode_location(
    location: str,
    max_results: int = 5,
    provider: str = "photon",
) -> List[GeoResult]:
    """
    Geocode *one* address / place string **using the specified `provider`.**

    Parameters
    ----------
    location : str
        Free‑text place or address.
    max_results : int, default = 5
        Upper bound on returned candidates.
    provider : str, default ``"photon"``
        Which geocoder backend to query (``photon``, ``nominatim``, ``pelias``,
        ``mapbox``, ``google``). If one fails then try again witb another geocoder in the prioritised above order.
    """
    geo_fn = _rate_limited_geocode_for(provider)

    try:
        raw_locs = geo_fn(
            location,
            exactly_one=False,
            limit=max_results,
            addressdetails=True,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("geocode_location error (%s): %s", provider, exc)
        return []

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
    return results


@mcp.tool()
def reverse_geocode(lat: float, lon: float) -> Optional[GeoResult]:
    """Reverse‑geocode a lat/lon pair to the nearest address."""
    try:
        loc = reverse((lat, lon))
        if not loc:
            return None
        raw = loc.raw or {}
        return GeoResult(
            latitude=lat,
            longitude=lon,
            address=loc.address,
            details=raw.get("address", {}),
            bounding_box=raw.get("boundingbox", []),
            raw=raw,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("reverse_geocode error: %s", exc)
        return None


@mcp.tool()
def geocode_with_details(location: str) -> Optional[GeoResult]:
    """Geocode a string with extra address details & bounding box."""
    try:
        loc = geocoder.geocode(location, addressdetails=True)
        if not loc:
            return None
        raw = loc.raw or {}
        return GeoResult(
            latitude=loc.latitude,
            longitude=loc.longitude,
            address=loc.address,
            details=raw.get("address", {}),
            bounding_box=raw.get("boundingbox", []),
            raw=raw,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("geocode_with_details error: %s", exc)
        return None


###############################################################################
# Bulk / multi helpers
###############################################################################
@mcp.tool()
def geocode_locations(
    locations: Union[str, List[str]],
    max_results: int = 5,
) -> Union[List[GeoResult], List[List[GeoResult]]]:
    """
    Geocode one **or many** location strings, returning *up to* ``max_results``
    candidates for each query.
    """
    queries = [locations] if isinstance(locations, str) else locations
    all_results: List[List[GeoResult]] = []

    for query in queries:
        try:
            raw_locs = geocode(
                query,
                exactly_one=False,
                limit=max_results,
                addressdetails=True,
            )
        except (GeocoderTimedOut, GeocoderServiceError) as exc:
            logger.warning("geocode_locations error: %s", exc)
            all_results.append([])
            continue

        candidates: List[GeoResult] = []
        if raw_locs:
            for loc in raw_locs[:max_results]:
                r = loc.raw or {}
                candidates.append(
                    GeoResult(
                        latitude=loc.latitude,
                        longitude=loc.longitude,
                        address=loc.address,
                        details=r.get("address", {}),
                        bounding_box=r.get("boundingbox", []),
                        raw=r,
                    )
                )
        all_results.append(candidates)

    return all_results[0] if isinstance(locations, str) else all_results


###############################################################################
# Legacy helpers (still useful)
###############################################################################
@mcp.tool()
def geocode_multiple_locations(
    locations: List[str],
) -> List[Optional[GeoResult]]:
    """
    Bulk geocode; **first candidate only** per query (legacy compatibility).
    """
    results: List[Optional[GeoResult]] = []
    for loc in locations:
        matches = geocode_location(loc, max_results=1)
        results.append(matches[0] if matches else None)
    return results


@mcp.tool()
def reverse_geocode_multiple_locations(
    coords: List[List[float]],
) -> List[Optional[GeoResult]]:
    """Bulk reverse‑geocode a list of [lat, lon] pairs."""
    return [
        reverse_geocode(lat, lon) if len(pair) == 2 else None
        for pair in coords
        for lat, lon in [pair]  # destructuring for clarity
    ]


###############################################################################
# Distance helpers
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
    dist = geodistance((lat1, lon1), (lat2, lon2))
    return dist.miles if unit.lower().startswith("mile") else dist.kilometers


@mcp.tool(name="distance_between_addresses")
def distance_between_addresses(
    address1: str,
    address2: str,
    unit: str = "kilometers",
) -> Optional[float]:
    """Distance between two address strings (uses first match for each)."""
    loc1 = geocode_location(address1, max_results=1)
    loc2 = geocode_location(address2, max_results=1)
    if not (loc1 and loc2):
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
    mcp.run(
        transport="sse",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
