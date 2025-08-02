from __future__ import annotations
import os
import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
    overload,
)

from dotenv import load_dotenv
from fastmcp import FastMCP
from geopy.geocoders import Nominatim, ArcGIS, Bing
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import distance as geodistance

load_dotenv()

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("geonext-mcp")

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
Provider = Nominatim | ArcGIS | Bing


def _build_geocoder() -> Provider:
    provider = os.getenv("GEOCODER_PROVIDER", "nominatim").lower()
    if provider == "nominatim":
        return Nominatim(
            user_agent=os.getenv("NOMINATIM_USER_AGENT", "geonext-mcp/0.2.0"),
            domain=os.getenv("NOMINATIM_URL", "nominatim.openstreetmap.org"),
            scheme=os.getenv("SCHEME", "https"),
            timeout=10,
        )
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
# Low‑level helpers (unchanged)
###############################################################################
@mcp.tool()
def geocode_location(location: str) -> Optional[GeoResult]:
    """Convert an address / place name to lat, lon and formatted address."""
    try:
        loc = geocode(location)
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
        logger.warning("geocode_location error: %s", exc)
        return None


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
    """Geocode with bounding box & raw address details (when available)."""
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
# >>> NEW MAIN TOOL FOR THE LLM <<<
###############################################################################
@mcp.tool(name="geocode_locations")
def geocode_locations(
    locations: Union[str, List[str]],
    max_results: int = 5,
) -> Union[List[GeoResult], List[List[GeoResult]]]:
    """
    Geocode one **or** many location strings and return **up to `max_results`**
    candidates *per* query.

    ──────────────────────────────────────────────────────────────────────────
    Args:
        locations: A single location string **or** a list of them.
        max_results: Maximum number of candidates returned for each query
                     (default = 5).

    Returns:
        • If the input is a single string  -> List[GeoResult]\n
        • If the input is a list[str]      -> List[List[GeoResult]] (same order)

    Notes for the LLM:
        • Each GeoResult contains latitude, longitude, address, address
          components (`details`), bounding box and the provider's raw JSON.
        • Use the extra metadata to choose the best match given the context
          of the original text. If none appear to match, you may discard them.
        • Respect `confidence` and `precision` rules in your final answer.
    """
    queries = [locations] if isinstance(locations, str) else locations
    all_results: List[List[GeoResult]] = []

    for query in queries:
        try:
            raw_locs = geocode(  # Rate‑limited wrapper
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
# Legacy bulk helpers remain (optional now but harmless)
###############################################################################
@mcp.tool()
def geocode_multiple_locations(
    locations: List[str],
) -> List[Optional[GeoResult]]:
    """Bulk geocode; single top result per query (legacy)."""
    return [geocode_location(loc) for loc in locations]


@mcp.tool()
def reverse_geocode_multiple_locations(
    coords: List[List[float]],
) -> List[Optional[GeoResult]]:
    """Bulk reverse‑geocode a list of [lat, lon] pairs."""
    return [
        reverse_geocode(lat, lon) if len(pair) == 2 else None
        for pair in coords
        for lat, lon in [pair]  # quick destructuring
    ]


@mcp.tool()
def distance_between_addresses(
    address1: str, address2: str, unit: str = "kilometers"
) -> Optional[float]:
    """Distance between two addresses."""
    loc1 = geocode_location(address1)
    loc2 = geocode_location(address2)
    if not (loc1 and loc2):
        return None
    return _distance_between_coords(
        loc1["latitude"], loc1["longitude"], loc2["latitude"], loc2["longitude"], unit
    )


@mcp.tool(name="distance_between_coords")
def _distance_between_coords(
    lat1: float, lon1: float, lat2: float, lon2: float, unit: str = "kilometers"
) -> float:
    """Distance between two lat/lon points."""
    dist = geodistance((lat1, lon1), (lat2, lon2))
    return dist.miles if unit.lower().startswith("mile") else dist.kilometers

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
