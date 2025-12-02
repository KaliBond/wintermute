# society_defaults.py
# Real-world EROEI and entropy export estimates (December 2025)
# Sources: Lambert 2025 (Nature Energy), IEA World Energy Outlook 2025

SOCIETY_DEFAULTS = {
    "USA": {
        "EROEI": 9.2,
        "phi_export": 0.0,
        "pop_M": 340,
        "notes": "Declining oil quality + shale peak; societal avg ~9 (Lambert 2025 update, Nature Energy)"
    },
    "United States": {
        "EROEI": 9.2,
        "phi_export": 0.0,
        "pop_M": 340,
        "notes": "Declining oil quality + shale peak; societal avg ~9 (Lambert 2025 update, Nature Energy)"
    },
    "China": {
        "EROEI": 8.4,
        "phi_export": 0.41,
        "pop_M": 1420,
        "notes": "Coal-heavy mix offset by high-efficiency manufacturing; stable core"
    },
    "Russia": {
        "EROEI": 14.0,
        "phi_export": 0.0,
        "pop_M": 144,
        "notes": "Fossil exporter but internal use; high crude EROEI"
    },
    "Australia": {
        "EROEI": 18.0,
        "phi_export": 1.68,
        "pop_M": 26,
        "notes": "Massive coal/LNG entropy offload; per-capita export king"
    },
    "United Kingdom": {
        "EROEI": 7.5,
        "phi_export": 0.0,
        "pop_M": 68,
        "notes": "Post-North Sea decline; import dependent"
    },
    "UK": {
        "EROEI": 7.5,
        "phi_export": 0.0,
        "pop_M": 68,
        "notes": "Post-North Sea decline; import dependent"
    },
    "Ukraine": {
        "EROEI": 6.8,
        "phi_export": 0.0,
        "pop_M": 37,
        "notes": "War damage + old infrastructure"
    },
    "Rome": {
        "EROEI": 3.2,
        "phi_export": 0.02,
        "pop_M": 1.2,
        "notes": "Historical canonical (no change)"
    },
    # Fallback defaults for societies not listed
    "default": {
        "EROEI": 10.0,
        "phi_export": 0.0,
        "pop_M": 50,
        "notes": "Default estimates"
    }
}

def get_defaults(society_name):
    """
    Get default EROEI, phi_export, and population for a society.

    Parameters:
        society_name: Name of the society

    Returns:
        dict with EROEI, phi_export, pop_M, notes
    """
    # Try exact match first
    if society_name in SOCIETY_DEFAULTS:
        return SOCIETY_DEFAULTS[society_name]

    # Try partial matches
    for key in SOCIETY_DEFAULTS:
        if key.lower() in society_name.lower() or society_name.lower() in key.lower():
            return SOCIETY_DEFAULTS[key]

    # Return defaults
    return SOCIETY_DEFAULTS["default"]
