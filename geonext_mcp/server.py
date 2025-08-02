from __future__ import annotations
import os
import logging
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from fastmcp import FastMCP
from geopy.geocoders import Nominatim, ArcGIS, Bing
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import distance as geodistance

load_dotenv()  # honours a local .env in dev

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
    description="Geocoding & distance tools exposed over the Model Context Protocol",
    dependencies=["geopy"],  # shows up in the MCP manifest
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
    latitude: float
    longitude: float
    address: str
    details: dict[str, Any]
    bounding_box: list[str]

###############################################################################
# MCP tools
###############################################################################
@mcp.tool()
def geocode_location(location: str) -> Optional[GeoResult]:
    """Convert an address / place name to lat, lon and formatted address."""
    try:
        loc = geocode(location)
        return (
            {"latitude": loc.latitude, "longitude": loc.longitude, "address": loc.address}
            if loc
            else None
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("geocode_location error: %s", exc)
        return None


@mcp.tool()
def reverse_geocode(lat: float, lon: float) -> Optional[GeoResult]:
    """Reverse‑geocode a lat/lon pair to the nearest address."""
    try:
        loc = reverse((lat, lon))
        return (
            {"latitude": lat, "longitude": lon, "address": loc.address}
            if loc
            else None
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
        )
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        logger.warning("geocode_with_details error: %s", exc)
        return None


@mcp.tool()
def geocode_multiple_locations(
    locations: List[str],
) -> List[Optional[GeoResult]]:
    """Bulk geocode; honour min‑delay to stay within service quotas."""
    results: List[Optional[GeoResult]] = []
    for loc in locations:
        results.append(geocode_location(loc))
    return results


@mcp.tool()
def reverse_geocode_multiple_locations(
    coords: List[List[float]],
) -> List[Optional[GeoResult]]:
    """Bulk reverse‑geocode; coords is a list of [lat, lon] pairs."""
    results: List[Optional[GeoResult]] = []
    for lat, lon in (pair for pair in coords if len(pair) == 2):
        results.append(reverse_geocode(lat, lon))
    return results


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
def main() -> None:  # makes `python -m geonext_mcp` possible
    mcp.run(
        transport="http",          # Fast streaming HTTP transport
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )

if __name__ == "__main__":
    main()
