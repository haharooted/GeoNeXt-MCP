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
from geopy.distance import distance as geodistance
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import (
    Nominatim,
    ArcGIS,
    Bing,
    Photon,
    GoogleV3,
)

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
Provider = Nominatim | ArcGIS | Bing | Photon | GoogleV3


def _build_geocoder() -> Provider:
    provider = os.getenv("GEOCODER_PROVIDER", "nominatim").lower()

    if provider == "nominatim":
        return Nominatim(
            user_agent=os.getenv("NOMINATIM_USER_AGENT", "geonext-mcp/0.2.0"),
            domain=os.getenv("NOMINATIM_URL", "nominatim.openstreetmap.org"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=60,
        )

    if provider == "photon":
        return Photon(
            user_agent=os.getenv("PHOTON_USER_AGENT", "geonext-mcp/0.2.0"),
            domain=os.getenv("PHOTON_URL", "photon.komoot.io"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=10,
        )

    if provider == "google":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY is required when GEOCODER_PROVIDER=google"
            )
        return GoogleV3(api_key=key, timeout=10)

    if provider == "bing":
        key = os.getenv("BING_API_KEY")
        if not key:
            raise RuntimeError("BING_API_KEY is required when GEOCODER_PROVIDER=bing")
        return Bing(api_key=key, timeout=10)

    if provider == "arcgis":
        return ArcGIS(
            username=os.getenv("ARC_USERNAME"),
            password=os.getenv("ARC_PASSWORD"),
            referer=os.getenv("ARC_REFERER"),
            timeout=10,
        )

    raise ValueError(f"Unsupported geocoder provider: {provider}")


geocoder: Provider = _build_geocoder()
min_delay = float(os.getenv("GEOCODER_MIN_DELAY", "1.0"))
geocode = RateLimiter(geocoder.geocode, min_delay_seconds=min_delay)
reverse = RateLimiter(geocoder.reverse, min_delay_seconds=min_delay)

###############################################################################
# Typed return payloads
###############################################################################
class GeoResult(TypedDict, total=False):
    """Standardised response shape for all geocoding tools."""
    latitude: float
    longitude: float
    address: str
    details: Dict[str, Any]
    bounding_box: List[str]
    raw: Dict[str, Any]  # full provider JSON for maximum context


###############################################################################
# Core helpers
###############################################################################
@mcp.tool()
def geocode_location(location: str, max_results: int = 5) -> List[GeoResult]:
    """
    Geocode *one* address / place string.

    Returns an **ordered list** (possibly empty) of up to ``max_results``
    candidates, each as a ``GeoResult``.
    """
    try:
        raw_locs = geocode(
            location,
            exactly_one=False,
            limit=max_results,
            addressdetails=True,
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("geocode_location error: %s", exc)
        return []

    results: List[GeoResult] = []
    if raw_locs:
        for loc in raw_locs[:max_results]:
            r = loc.raw or {}
            results.append(
                GeoResult(
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
