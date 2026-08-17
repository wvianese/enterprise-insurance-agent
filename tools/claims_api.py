claims = {
    "104": [
        {"type": "flood", "year": 2024, "status": "approved"},
        {"type": "water_leak", "year": 2025, "status": "denied"},
    ],
    "205": [
        {"type": "fire", "year": 2025, "status": "approved"},
    ],
}

def get_claims(customer_id):
    return claims.get(customer_id, [])