customers = {
    "104": {
        "policy": "HomePlus",
        "flood_zone": "medium",
        "property_type": "house"
    },
    "205": {
        "policy": "BasicHome",
        "flood_zone": "low",
        "property_type": "flat"
    }
}

def get_customer(customer_id):
    return customers.get(customer_id)