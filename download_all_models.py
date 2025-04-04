import whisper
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of models to download (adjust as needed)
# Using the same choices as the main script for consistency
MODELS_TO_DOWNLOAD = [
    'tiny.en', 'tiny',
    'base.en', 'base',
    'small.en', 'small',
    'medium.en', 'medium',
    'large-v1', 'large-v2', 'large-v3', 'large'
]

def download_models():
    """Iterates through the model list and downloads each one."""
    logging.info(f"Starting download process for {len(MODELS_TO_DOWNLOAD)} models...")
    successful_downloads = 0
    failed_downloads = []

    for model_name in MODELS_TO_DOWNLOAD:
        try:
            logging.info(f"--- Attempting to download model: '{model_name}' ---")
            # whisper.load_model will download if not present in cache
            whisper.load_model(model_name)
            logging.info(f"Successfully loaded/downloaded model: '{model_name}'")
            successful_downloads += 1
        except Exception as e:
            logging.error(f"Failed to download model '{model_name}'. Error: {e}")
            failed_downloads.append(model_name)
        logging.info("---") # Separator

    logging.info("=== Download Process Complete ===")
    logging.info(f"Successfully loaded/downloaded: {successful_downloads} models.")
    if failed_downloads:
        logging.warning(f"Failed to download: {len(failed_downloads)} models: {failed_downloads}")
    else:
        logging.info("All specified models downloaded successfully!")

if __name__ == "__main__":
    download_models() 