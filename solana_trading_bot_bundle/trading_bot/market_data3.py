# Import necessary modules
from datetime import datetime
import logging
import requests
import os

# Set up consistent logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Endpoints
JUPITER_API = "https://quote-api.jup.ag/v4/quote"
BIRDEYE_API = "https://public-api.birdeye.so"  # Updated for consistency

# DRY_RUN environment variable
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

def fetch_market_data_jupiter(identifier):
    """
    Fetch market data from Jupiter API.
    """
    try:
        logger.info("Fetching market data from Jupiter API for identifier: %s", identifier)
        response = requests.get(f"{JUPITER_API}/{identifier}")
        response.raise_for_status()
        data = response.json()
        logger.debug("Jupiter response: %s", data)
        return data
    except requests.RequestException as e:
        logger.error("Error while fetching data from Jupiter API: %s", e)
        return None

def fetch_market_data_birdeye(identifier):
    """
    Fetch market data from Birdeye API.
    """
    try:
        logger.info("Fetching market data from Birdeye API for identifier: %s", identifier)
        response = requests.get(f"{BIRDEYE_API}/v1/price?identifier={identifier}")
        response.raise_for_status()
        data = response.json()
        logger.debug("Birdeye response: %s", data)
        return data
    except requests.RequestException as e:
        logger.error("Error while fetching data from Birdeye API: %s", e)
        return None

def main():
    """
    Main execution function.
    """
    logger.info("Starting market data retrieval")
    if DRY_RUN:
        logger.info("Running in DRY_RUN mode. No API calls will be made.")
        return

    identifiers = ["SOL", "ETH", "BTC"]
    for identifier in identifiers:
        logger.info("Processing identifier: %s", identifier)
        jupiter_data = fetch_market_data_jupiter(identifier)
        birdeye_data = fetch_market_data_birdeye(identifier)

        if jupiter_data:
            logger.info("Jupiter Data for %s: %s", identifier, jupiter_data)
        else:
            logger.warning("No data received from Jupiter for %s", identifier)

        if birdeye_data:
            logger.info("Birdeye Data for %s: %s", identifier, birdeye_data)
        else:
            logger.warning("No data received from Birdeye for %s", identifier)

    logger.info("Market data retrieval completed")

if __name__ == "__main__":
    main()