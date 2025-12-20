# Import necessary libraries
import logging
import requests
from requests.exceptions import RequestException

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
JUPITER_API = "https://api.jupiter.aggregate/v3/price"
BIRDEYE_API = "https://api.birdeye.so/public/v1/market"

# DRY_RUN setting for debugging (safeguards to prevent real execution)
DRY_RUN = True

# Function to fetch data from an API with robust error handling
def fetch_api_data(url, params=None):
    try:
        logging.info("Fetching data from API: %s with params: %s", url, params)
        if DRY_RUN:
            logging.debug("DRY_RUN enabled, skipping actual request.")
            return {'dry_run': True}

        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors

        logging.info("Data fetched successfully from %s", url)
        return response.json()
    except RequestException as e:
        logging.error("Network error occurred while accessing %s: %s", url, e)
        return None
    except ValueError as e:
        logging.error("Failed to decode JSON response from %s: %s", url, e)
        return None

# Verify endpoints are returning expected results
def verify_endpoint_results(endpoint_data):
    if endpoint_data is None:
        logging.warning("Endpoint returned None. Check logs for prior errors.")
        return False
    if 'dry_run' in endpoint_data:
        logging.debug("Dry run data passed verification.")
        return True  # Always accept during DRY_RUN
    if not isinstance(endpoint_data, dict):
        logging.warning("Endpoint data not in expected dictionary format: %s", endpoint_data)
        return False
    logging.info("Endpoint data verified successfully.")
    return True

# Example usage
if __name__ == "__main__":
    jupiter_data = fetch_api_data(JUPITER_API, params={"token": "SOL"})
    if verify_endpoint_results(jupiter_data):
        logging.info("Jupiter data:", jupiter_data)

    birdeye_data = fetch_api_data(BIRDEYE_API, params={"limit": 10})
    if verify_endpoint_results(birdeye_data):
        logging.info("Birdeye data:", birdeye_data)