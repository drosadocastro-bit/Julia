"""
Julia PR Climate Data Pipeline — NOAA NCEI API Downloader
=========================================================
Downloads 30-year monthly climate normals (1991-2020) for all 79
PR & USVI weather stations, then merges with crop production data.

Usage:
    python pr_climate_pipeline.py --download     # Pull data from NCEI API
    python pr_climate_pipeline.py --merge        # Merge climate + crops
    python pr_climate_pipeline.py --all          # Full pipeline
    python pr_climate_pipeline.py --report       # Show summary report

Requirements:
    pip install requests pandas

Output:
    data/ncei_monthly_normals.csv       — Raw NCEI API response
    data/pr_climate_by_station.csv      — Cleaned per-station monthly normals
    data/pr_crops_with_climate.csv      — Crops + nearest station climate
    data/pr_climate_unified.csv         — Unified ML-ready dataset
    data/pipeline_report.json           — Quality gate report
"""

import os
import sys
import json
import math
import time
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Station metadata extracted from 1991-2020_Puerto_Rico_and_USVI_Normals.csv
STATIONS = [
    {"id": "RQC00660040", "name": "ACEITUNA", "lat": 18.1487, "lon": -66.4931, "data": "Rainfall"},
    {"id": "RQC00660061", "name": "ADJUNTAS SUBSTN", "lat": 18.1747, "lon": -66.7977, "data": "Rainfall and Temperatures"},
    {"id": "RQC00660152", "name": "AGUIRRE", "lat": 17.9555, "lon": -66.222, "data": "Rainfall and Temperatures"},
    {"id": "RQC00660158", "name": "AIBONITO 1 S", "lat": 18.128, "lon": -66.2642, "data": "Rainfall and Temperatures"},
    {"id": "VQ1VIST0001", "name": "ANNA'S RETREAT 2.5 ESE", "lat": 18.3255, "lon": -64.8513, "data": "Rainfall"},
    {"id": "RQC00660213", "name": "ARECIBO 5.2 ESE", "lat": 18.4414, "lon": -66.6589, "data": "Rainfall"},
    {"id": "RQC00660229", "name": "ARECIBO OBSY", "lat": 18.3494, "lon": -66.7525, "data": "Rainfall and Temperatures"},
    {"id": "RQC00660244", "name": "BARCELONETA 3 SW", "lat": 18.4285, "lon": -66.563, "data": "Rainfall"},
    {"id": "RQC00660412", "name": "BOCA", "lat": 17.9906, "lon": -66.816, "data": "Rainfall"},
    {"id": "RQC00660438", "name": "BORINQUEN AP", "lat": 18.491, "lon": -67.134, "data": "Temperatures"},
    {"id": "RQC00660540", "name": "CACAOS-OROCOVIS", "lat": 18.2261, "lon": -66.5039, "data": "Rainfall"},
    {"id": "RQC00660545", "name": "CALERO CAMP", "lat": 18.4724, "lon": -67.1155, "data": "Rainfall"},
    {"id": "RQC00660585", "name": "CANOVANAS", "lat": 18.3791, "lon": -65.8938, "data": "Rainfall"},
    {"id": "VQC00670606", "name": "CHARLOTTE AMALIE AP", "lat": 18.3333, "lon": -64.9667, "data": "Rainfall and Temperatures"},
    {"id": "VQC00670607", "name": "CHRISTIANSTED AP", "lat": 17.7027, "lon": -64.8055, "data": "Rainfall and Temperatures"},
    {"id": "VQC00670611", "name": "CHRISTIANSTED FT, VI", "lat": 17.7469, "lon": -64.7013, "data": "Rainfall and Temperatures"},
    {"id": "RQC00660720", "name": "COAMO 2 SW", "lat": 18.0664, "lon": -66.3781, "data": "Rainfall"},
    {"id": "RQC00660728", "name": "COLOSO", "lat": 18.3808, "lon": -67.1569, "data": "Rainfall and Temperatures"},
    {"id": "VQ1VICR0002", "name": "CORAL BAY, VI", "lat": 18.3491, "lon": -64.7136, "data": "Rainfall"},
    {"id": "RQC00660757", "name": "CORRAL VIEJO", "lat": 18.0836, "lon": -66.6547, "data": "Rainfall"},
    {"id": "RQC00660795", "name": "CULEBRA HILL", "lat": 18.2972, "lon": -65.29, "data": "Rainfall"},
    {"id": "RQC00660977", "name": "DORADO 2 WNW", "lat": 18.4722, "lon": -66.3058, "data": "Rainfall and Temperatures"},
    {"id": "RQC00660983", "name": "DOS BOCAS", "lat": 18.3361, "lon": -66.6666, "data": "Rainfall and Temperatures"},
    {"id": "VQ1VIST0007", "name": "EAST END, VI", "lat": 18.3347, "lon": -64.6757, "data": "Rainfall"},
    {"id": "VQ1VICR0005", "name": "EAST HILL, VI", "lat": 17.7561, "lon": -64.6491, "data": "Rainfall"},
    {"id": "RQC00661011", "name": "ENSENADA 1 W", "lat": 17.9727, "lon": -66.9458, "data": "Rainfall"},
    {"id": "VQ1VICR0006", "name": "ESTATE THE SIGHT, VI", "lat": 17.7419, "lon": -64.6603, "data": "Rainfall"},
    {"id": "VQ1VICR0003", "name": "GRANARD, VI", "lat": 17.7164, "lon": -64.7117, "data": "Rainfall"},
    {"id": "RQC00661155", "name": "GUAJATACA DAM", "lat": 18.3963, "lon": -66.9244, "data": "Rainfall"},
    {"id": "RQC00661194", "name": "GUAYABAL", "lat": 18.0742, "lon": -66.4967, "data": "Rainfall"},
    {"id": "RQC00661197", "name": "GUAYAMA 2E", "lat": 17.9786, "lon": -66.0874, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661210", "name": "GURABO SUBSTN", "lat": 18.2583, "lon": -65.9922, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661255", "name": "HACIENDA CONSTANZA", "lat": 18.2158, "lon": -67.0883, "data": "Rainfall"},
    {"id": "RQC00661345", "name": "HUMACAO NATURAL RESERVE", "lat": 18.1506, "lon": -65.7719, "data": "Rainfall"},
    {"id": "RQC00661395", "name": "ISABELA SUBSTN", "lat": 18.4652, "lon": -67.0525, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661398", "name": "JAJOME ALTO", "lat": 18.0694, "lon": -66.1431, "data": "Rainfall"},
    {"id": "RQC00661411", "name": "JAYUYA", "lat": 18.215, "lon": -66.5931, "data": "Rainfall"},
    {"id": "RQC00661439", "name": "JUANA DIAZ 2.9 SW", "lat": 18.0281, "lon": -66.5383, "data": "Rainfall"},
    {"id": "RQC00661441", "name": "JUANA DIAZ CAMP", "lat": 18.0513, "lon": -66.4986, "data": "Rainfall"},
    {"id": "RQC00661461", "name": "JUNCOS 0.3 WSW", "lat": 18.2228, "lon": -65.9161, "data": "Rainfall"},
    {"id": "RQC00661462", "name": "JUNCOS 1 SE", "lat": 18.2264, "lon": -65.9114, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661530", "name": "LAJAS SUBSTN", "lat": 18.033, "lon": -67.0722, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661572", "name": "MAGUEYES ISLAND", "lat": 17.9722, "lon": -67.0461, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661586", "name": "MANATI 2 E", "lat": 18.4308, "lon": -66.4661, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661629", "name": "MARICAO 2 SSW", "lat": 18.1511, "lon": -66.9888, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661632", "name": "MARICAO FISH HATCHERY", "lat": 18.1725, "lon": -66.9872, "data": "Rainfall"},
    {"id": "RQC00661699", "name": "MAYAGUEZ AIRPORT", "lat": 18.2538, "lon": -67.1484, "data": "Rainfall"},
    {"id": "RQC00661704", "name": "MAYAGUEZ ARRIBA", "lat": 18.217, "lon": -67.1167, "data": "Rainfall"},
    {"id": "RQC00661717", "name": "MAYAGUEZ CITY", "lat": 18.1876, "lon": -67.1378, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661740", "name": "MONA ISLAND 2", "lat": 18.0908, "lon": -67.9441, "data": "Rainfall and Temperatures"},
    {"id": "VQ1VICR0011", "name": "MONTPELLIER, VI", "lat": 17.7705, "lon": -64.7552, "data": ""},
    {"id": "RQC00661755", "name": "MORA CAMP", "lat": 18.4736, "lon": -67.0288, "data": "Rainfall"},
    {"id": "RQC00661824", "name": "MOROVIS 1 N", "lat": 18.3344, "lon": -66.4078, "data": "Rainfall"},
    {"id": "RQC00661829", "name": "NEGRO -COROZAL", "lat": 18.2916, "lon": -66.3497, "data": "Rainfall"},
    {"id": "RQC00661938", "name": "PALMA SOLA", "lat": 18.3169, "lon": -65.8664, "data": "Rainfall"},
    {"id": "RQC00661948", "name": "PALMAREJO VEGA BAJA", "lat": 18.385, "lon": -66.43, "data": "Rainfall and Temperatures"},
    {"id": "RQC00661955", "name": "PARAISO", "lat": 18.265, "lon": -65.7208, "data": "Rainfall"},
    {"id": "RQC00662030", "name": "PENUELAS 1 E", "lat": 18.0613, "lon": -66.7119, "data": "Rainfall"},
    {"id": "RQC00662036", "name": "PICO DEL ESTE", "lat": 18.271, "lon": -65.759, "data": "Temperatures"},
    {"id": "RQC00662057", "name": "PONCE 4 E", "lat": 18.0258, "lon": -66.5252, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662064", "name": "PONCE 5.0 NNW", "lat": 18.0474, "lon": -66.6514, "data": "Rainfall"},
    {"id": "VQ1VICR0012", "name": "PROSPECT HILL", "lat": 17.7456, "lon": -64.8883, "data": "Rainfall"},
    {"id": "RQC00662204", "name": "RINCON 1.5 N", "lat": 18.3627, "lon": -67.2497, "data": "Rainfall"},
    {"id": "RQC00662207", "name": "RINCON 2.8 SE", "lat": 18.3188, "lon": -67.2178, "data": "Rainfall"},
    {"id": "RQC00662228", "name": "RIO BLANCO LOWER", "lat": 18.2433, "lon": -65.785, "data": "Rainfall"},
    {"id": "RQC00662270", "name": "RIO PIEDRAS EXP STN", "lat": 18.3905, "lon": -66.0541, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662286", "name": "ROOSEVELT ROADS", "lat": 18.2552, "lon": -65.6411, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662330", "name": "SABANA GRANDE 2 ENE", "lat": 18.0889, "lon": -66.93, "data": "Rainfall"},
    {"id": "RQC00662458", "name": "SAN JUAN INTL AP", "lat": 18.4325, "lon": -66.0108, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662491", "name": "SAN LORENZO 3S", "lat": 18.1517, "lon": -65.9589, "data": "Rainfall"},
    {"id": "RQC00662519", "name": "SANTA ISABEL 2 ENE", "lat": 17.9691, "lon": -66.3772, "data": "Rainfall"},
    {"id": "RQC00662526", "name": "SANTA RITA", "lat": 18.0097, "lon": -66.8847, "data": "Rainfall"},
    {"id": "RQC00662700", "name": "TOA BAJA LEVITTOWN", "lat": 18.4356, "lon": -66.1678, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662710", "name": "TORO NEGRO FOREST", "lat": 18.1731, "lon": -66.4928, "data": "Rainfall"},
    {"id": "RQC00662730", "name": "TRUJILLO ALTO 2 SSW", "lat": 18.3283, "lon": -66.0163, "data": "Rainfall and Temperatures"},
    {"id": "RQC00662812", "name": "VILLALBA 1 SE", "lat": 18.1094, "lon": -66.5055, "data": "Rainfall"},
    {"id": "RQC00662856", "name": "WFO SAN JUAN", "lat": 18.4255, "lon": -65.9916, "data": "Rainfall and Temperatures"},
    {"id": "VQ1VIST0005", "name": "WINTBERG, VI", "lat": 18.3503, "lon": -64.9167, "data": "Rainfall"},
    {"id": "RQC00662871", "name": "YAUCO 1 NW", "lat": 18.0434, "lon": -66.8606, "data": "Rainfall"},
]

# NCEI API base
NCEI_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"
DATASET = "normals-monthly-1991-2020"
DATA_TYPES = "MLY-TMAX-NORMAL,MLY-TMIN-NORMAL,MLY-TAVG-NORMAL,MLY-PRCP-NORMAL"

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def find_nearest_station(lat, lon, stations):
    """Find nearest weather station to a given coordinate."""
    best = None
    best_dist = float('inf')
    for s in stations:
        d = haversine(lat, lon, s['lat'], s['lon'])
        if d < best_dist:
            best_dist = d
            best = s
    return best, best_dist

# ============================================================
# STEP 1: DOWNLOAD CLIMATE NORMALS FROM NCEI API
# ============================================================
def download_normals():
    """Download monthly climate normals for all PR/USVI stations."""
    import requests
    
    print("=" * 70)
    print("  🌦️ DOWNLOADING NCEI MONTHLY NORMALS")
    print("=" * 70)
    
    # Batch stations (API may have limits)
    # Pull in batches of 10 stations
    all_data = []
    batch_size = 10
    station_ids = [s['id'] for s in STATIONS]
    
    for i in range(0, len(station_ids), batch_size):
        batch = station_ids[i:i+batch_size]
        batch_str = ",".join(batch)
        
        url = (
            f"{NCEI_BASE}?"
            f"dataset={DATASET}"
            f"&stations={batch_str}"
            f"&dataTypes={DATA_TYPES}"
            f"&format=json"
            f"&includeStationName=true"
            f"&includeStationLocation=true"
        )
        
        print(f"  Batch {i//batch_size + 1}/{math.ceil(len(station_ids)/batch_size)}: "
              f"stations {i+1}-{min(i+batch_size, len(station_ids))}...", end=" ")
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                all_data.extend(data)
                print(f"✅ {len(data)} records")
            else:
                print(f"⚠️ HTTP {resp.status_code}")
                # Try CSV fallback
                csv_url = url.replace("format=json", "format=csv")
                resp2 = requests.get(csv_url, timeout=30)
                if resp2.status_code == 200:
                    print(f"    CSV fallback: {len(resp2.text.splitlines())} lines")
                    # Save CSV directly
                    csv_path = DATA_DIR / f"ncei_batch_{i//batch_size}.csv"
                    csv_path.write_text(resp2.text)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Be nice to the API
    
    # Save raw data
    output_path = DATA_DIR / "ncei_monthly_normals_raw.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n  Total records downloaded: {len(all_data)}")
    print(f"  Saved to: {output_path}")
    
    return all_data

# ============================================================
# STEP 2: CLEAN AND STRUCTURE CLIMATE DATA
# ============================================================
def clean_normals(raw_data=None):
    """Clean and structure the downloaded normals into per-station monthly data."""
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("  🧹 CLEANING CLIMATE NORMALS")
    print("=" * 70)
    
    if raw_data is None:
        raw_path = DATA_DIR / "ncei_monthly_normals_raw.json"
        if raw_path.exists():
            with open(raw_path) as f:
                raw_data = json.load(f)
        else:
            print("  ❌ No raw data found. Run --download first.")
            return None
    
    # Parse into structured records
    records = []
    for entry in raw_data:
        station = entry.get("STATION", "")
        month = entry.get("DATE", "")  # Format: YYYY-MM
        
        record = {
            "station_id": station,
            "station_name": entry.get("NAME", ""),
            "month": month,
            "tmax_normal_f": entry.get("MLY-TMAX-NORMAL"),
            "tmin_normal_f": entry.get("MLY-TMIN-NORMAL"),
            "tavg_normal_f": entry.get("MLY-TAVG-NORMAL"),
            "prcp_normal_in": entry.get("MLY-PRCP-NORMAL"),
        }
        
        # Convert F to C and inches to mm
        try:
            if record["tmax_normal_f"]:
                record["tmax_normal_c"] = round((float(record["tmax_normal_f"]) - 32) * 5/9, 1)
            if record["tmin_normal_f"]:
                record["tmin_normal_c"] = round((float(record["tmin_normal_f"]) - 32) * 5/9, 1)
            if record["tavg_normal_f"]:
                record["tavg_normal_c"] = round((float(record["tavg_normal_f"]) - 32) * 5/9, 1)
            if record["prcp_normal_in"]:
                record["prcp_normal_mm"] = round(float(record["prcp_normal_in"]) * 25.4, 1)
        except (ValueError, TypeError):
            pass
        
        # Add station metadata
        for s in STATIONS:
            if s["id"] == station:
                record["lat"] = s["lat"]
                record["lon"] = s["lon"]
                break
        
        records.append(record)
    
    df = pd.DataFrame(records)
    output_path = DATA_DIR / "pr_climate_by_station.csv"
    df.to_csv(output_path, index=False)
    
    print(f"  Records: {len(df)}")
    print(f"  Stations: {df['station_id'].nunique()}")
    print(f"  Saved to: {output_path}")
    
    return df

# ============================================================
# STEP 3: MERGE CROPS WITH NEAREST CLIMATE STATION
# ============================================================
def merge_crops_climate():
    """Match each crop municipality to its nearest weather station."""
    import pandas as pd
    
    print("\n" + "=" * 70)
    print("  🔗 MERGING CROPS × CLIMATE")
    print("=" * 70)
    
    # Load crops data
    crops_df = pd.read_csv("Filtrarcultivospuertorico.csv", encoding="utf-8-sig")
    print(f"  Crop records: {len(crops_df)}")
    
    # Municipality coordinates (approximate centroids)
    # These would be better from a geocoding API, but here are common PR municipalities
    MUNI_COORDS = {
        "Adjuntas": (18.1627, -66.7225), "Aguada": (18.3800, -67.1888),
        "Aguadilla": (18.4277, -67.1541), "Aguas Buenas": (18.2569, -66.1030),
        "Aibonito": (18.1399, -66.2661), "Añasco": (18.2828, -67.1394),
        "Arecibo": (18.4725, -66.7157), "Arroyo": (17.9669, -66.0614),
        "Barceloneta": (18.4508, -66.5381), "Barranquitas": (18.1866, -66.3064),
        "Bayamón": (18.3985, -66.1551), "Cabo Rojo": (18.0866, -67.1457),
        "Caguas": (18.2341, -66.0485), "Camuy": (18.4838, -66.8449),
        "Carolina": (18.3811, -65.9572), "Cayey": (18.1119, -66.1661),
        "Coamo": (18.0799, -66.3578), "Comerío": (18.2197, -66.2258),
        "Corozal": (18.3414, -66.3172), "Dorado": (18.4588, -66.2677),
        "Guánica": (17.9716, -66.9085), "Guayama": (17.9838, -66.1117),
        "Guayanilla": (18.0193, -66.7918), "Guaynabo": (18.3565, -66.1108),
        "Gurabo": (18.2544, -65.9733), "Hatillo": (18.4864, -66.8253),
        "Humacao": (18.1496, -65.8196), "Isabela": (18.5000, -67.0244),
        "Jayuya": (18.2186, -66.5917), "Juana Díaz": (18.0533, -66.5067),
        "Juncos": (18.2275, -65.9213), "Lajas": (18.0500, -67.0594),
        "Lares": (18.2953, -66.8778), "Las Marías": (18.2508, -66.9917),
        "Las Piedras": (18.1830, -65.8722), "Manatí": (18.4317, -66.4839),
        "Maricao": (18.1808, -66.9797), "Maunabo": (18.0072, -65.8992),
        "Mayagüez": (18.2013, -67.1397), "Moca": (18.3967, -67.1131),
        "Morovis": (18.3255, -66.4064), "Naguabo": (18.2116, -65.7356),
        "Naranjito": (18.3008, -66.2450), "Orocovis": (18.2269, -66.3917),
        "Patillas": (18.0036, -66.0136), "Peñuelas": (18.0561, -66.7261),
        "Ponce": (18.0111, -66.6141), "Rincón": (18.3403, -67.2500),
        "San Germán": (18.0833, -67.0358), "San Lorenzo": (18.1897, -65.9608),
        "San Sebastián": (18.3364, -66.9908), "Santa Isabel": (17.9661, -66.4049),
        "Utuado": (18.2655, -66.7008), "Vega Baja": (18.4439, -66.3873),
        "Villalba": (18.1277, -66.4923), "Yauco": (18.0350, -66.8497),
    }
    
    # Find nearest station for each municipality
    print(f"\n  Matching {len(MUNI_COORDS)} municipalities to nearest stations...")
    muni_station_map = {}
    for muni, (lat, lon) in MUNI_COORDS.items():
        nearest, dist = find_nearest_station(lat, lon, STATIONS)
        muni_station_map[muni] = {
            "station_id": nearest["id"],
            "station_name": nearest["name"],
            "distance_km": round(dist, 2)
        }
        if dist > 15:
            print(f"  ⚠️ {muni}: nearest station is {dist:.1f} km away ({nearest['name']})")
    
    # Add station info to crops
    crops_df["nearest_station_id"] = crops_df["Municipio"].map(
        lambda m: muni_station_map.get(m, {}).get("station_id", ""))
    crops_df["nearest_station_name"] = crops_df["Municipio"].map(
        lambda m: muni_station_map.get(m, {}).get("station_name", ""))
    crops_df["station_distance_km"] = crops_df["Municipio"].map(
        lambda m: muni_station_map.get(m, {}).get("distance_km", ""))
    
    output_path = DATA_DIR / "pr_crops_with_stations.csv"
    crops_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"\n  Matched: {crops_df['nearest_station_id'].notna().sum()} / {len(crops_df)}")
    print(f"  Saved to: {output_path}")
    
    # Save municipality-station mapping
    map_path = DATA_DIR / "municipality_station_map.json"
    with open(map_path, "w") as f:
        json.dump(muni_station_map, f, indent=2, ensure_ascii=False)
    print(f"  Station map: {map_path}")
    
    return crops_df

# ============================================================
# STEP 4: QUALITY GATE
# ============================================================
def quality_gate():
    """Run quality checks on the pipeline outputs."""
    print("\n" + "=" * 70)
    print("  ✅ QUALITY GATE")
    print("=" * 70)
    
    report = {"timestamp": datetime.now().isoformat(), "checks": []}
    
    # Check climate data exists
    climate_path = DATA_DIR / "pr_climate_by_station.csv"
    if climate_path.exists():
        import pandas as pd
        df = pd.read_csv(climate_path)
        stations = df['station_id'].nunique()
        report["checks"].append({
            "name": "Climate normals",
            "status": "PASS" if stations >= 50 else "WARN",
            "detail": f"{stations} stations, {len(df)} records"
        })
        print(f"  Climate normals: {stations} stations, {len(df)} records — "
              f"{'✅ PASS' if stations >= 50 else '⚠️ WARN'}")
    else:
        report["checks"].append({"name": "Climate normals", "status": "MISSING"})
        print(f"  Climate normals: ❌ MISSING — run --download first")
    
    # Check crops+stations mapping
    crops_path = DATA_DIR / "pr_crops_with_stations.csv"
    if crops_path.exists():
        import pandas as pd
        df = pd.read_csv(crops_path)
        matched = df['nearest_station_id'].notna().sum()
        pct = matched / len(df) * 100
        report["checks"].append({
            "name": "Crops-station matching",
            "status": "PASS" if pct > 80 else "WARN",
            "detail": f"{matched}/{len(df)} matched ({pct:.1f}%)"
        })
        print(f"  Crops-station matching: {matched}/{len(df)} ({pct:.1f}%) — "
              f"{'✅ PASS' if pct > 80 else '⚠️ WARN'}")
    
    # Save report
    report_path = DATA_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Julia PR Climate Data Pipeline")
    parser.add_argument("--download", action="store_true", help="Download NCEI normals")
    parser.add_argument("--merge", action="store_true", help="Merge crops + climate")
    parser.add_argument("--report", action="store_true", help="Run quality gate")
    parser.add_argument("--all", action="store_true", help="Full pipeline")
    
    args = parser.parse_args()
    
    if args.all or args.download:
        raw_data = download_normals()
        clean_normals(raw_data)
    
    if args.all or args.merge:
        merge_crops_climate()
    
    if args.all or args.report:
        quality_gate()
    
    if not any([args.download, args.merge, args.report, args.all]):
        parser.print_help()
        print("\n  Quick start: python pr_climate_pipeline.py --all")

if __name__ == "__main__":
    main()