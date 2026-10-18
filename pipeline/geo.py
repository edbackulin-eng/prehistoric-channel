"""Places named in the narration, and the land they sit on, for the map animation.

The land outline is Natural Earth's 1:110m land polygons (public domain, no
attribution required), fetched once and cached. Places come only from the
narration: a city or country is put on the map when it is spoken, never because
a model thought it relevant, so a dot on the map is always something said.
"""
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "work" / "cache"
LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "v5.1.2/geojson/ne_110m_land.geojson")

# name -> (lat, lon). Cities first; a country falls back to a central point.
PLACES = {
    "paris": (48.86, 2.35), "warsaw": (52.23, 21.01), "miami": (25.76, -80.19),
    "sydney": (-33.87, 151.21), "london": (51.51, -0.13), "berlin": (52.52, 13.40),
    "moscow": (55.76, 37.62), "seoul": (37.57, 126.98), "tokyo": (35.68, 139.69),
    "seattle": (47.61, -122.33), "new york": (40.71, -74.01), "los angeles": (34.05, -118.24),
    "san francisco": (37.77, -122.42), "chicago": (41.88, -87.63), "boston": (42.36, -71.06),
    "washington": (38.91, -77.04), "new orleans": (29.95, -90.07), "toronto": (43.65, -79.38),
    "vancouver": (49.28, -123.12), "mexico city": (19.43, -99.13), "madrid": (40.42, -3.70),
    "rome": (41.90, 12.50), "stockholm": (59.33, 18.07), "oslo": (59.91, 10.75),
    "helsinki": (60.17, 24.94), "amsterdam": (52.37, 4.90), "kyiv": (50.45, 30.52),
    "prague": (50.08, 14.44), "vienna": (48.21, 16.37), "dublin": (53.35, -6.26),
    "melbourne": (-37.81, 144.96), "auckland": (-36.85, 174.76), "beijing": (39.90, 116.40),
    "hong kong": (22.32, 114.17), "singapore": (1.35, 103.82), "mumbai": (19.08, 72.88),
    "cairo": (30.04, 31.24), "sao paulo": (-23.55, -46.63), "buenos aires": (-34.60, -58.38),
    "honolulu": (21.31, -157.86), "hawaii": (20.8, -156.3), "alaska": (64.2, -149.5),
    "texas": (31.0, -99.0), "california": (36.8, -119.4), "florida": (27.8, -81.7),
    "arizona": (34.0, -111.1), "united states": (39.8, -98.6), "america": (39.8, -98.6),
    "canada": (56.1, -106.3), "united kingdom": (54.0, -2.0), "britain": (54.0, -2.0),
    "england": (52.5, -1.5), "france": (46.6, 2.2), "germany": (51.2, 10.4),
    "poland": (52.0, 19.1), "spain": (40.4, -3.7), "italy": (42.8, 12.6),
    "russia": (61.5, 105.3), "china": (35.9, 104.2), "japan": (36.2, 138.3),
    "south korea": (35.9, 127.8), "korea": (35.9, 127.8), "australia": (-25.3, 133.8),
    "india": (20.6, 79.0), "brazil": (-14.2, -51.9), "mexico": (23.6, -102.6),
    "sweden": (60.1, 18.6), "norway": (60.5, 8.5), "finland": (61.9, 25.7),
    "ukraine": (48.4, 31.2), "netherlands": (52.1, 5.3), "ireland": (53.4, -8.2),
    "new zealand": (-40.9, 174.9), "antarctica": (-82.0, 0.0),
}
_NAME = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in PLACES), key=len, reverse=True))
                   + r")\b", re.I)


def places_in(text):
    """Places spoken in `text`, in order, without repeats: [(Name, lat, lon)].

    A country is dropped when a city in it is also named, so one place is not
    marked twice; "United States Navy" is not a place."""
    found = []
    for m in _NAME.finditer(text):
        k = m.group(1).lower()
        after = text[m.end():m.end() + 8].lower()
        if k in ("united states", "america") and re.match(r"\s+(navy|army|air|government|congress)", after):
            continue
        name = m.group(1).title()
        if all(f[0].lower() != k for f in found):
            found.append((name, *PLACES[k]))
    return found


def land():
    """Land polygons as lists of (lon, lat) rings."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "ne_110m_land.geojson"
    if not p.exists():
        req = urllib.request.Request(LAND_URL, headers={"User-Agent": "closed-tabs/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            p.write_bytes(r.read())
    rings = []
    for f in json.loads(p.read_text(encoding="utf-8"))["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for poly in polys:
            rings.append(poly[0])
    return rings
