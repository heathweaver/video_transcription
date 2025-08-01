import os
import asyncio
import aiohttp
import logging
from pathlib import Path
import sys
import argparse
import traceback
import time
from typing import Optional, Set, Dict
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants - will be overridden by environment variables if set
DOWNLOAD_DIR = os.getenv('DOWNLOAD_DIR', '/data/videos')
TRACKING_DIR = os.getenv('TRACKING_DIR', '/data/tracking')
DOWNLOADED_FILE = os.path.join(TRACKING_DIR, 'downloaded.txt')
DOWNLOAD_LIST_FILE = os.path.join(TRACKING_DIR, 'download_list.txt')
FILE_SIZES_FILE = os.path.join(TRACKING_DIR, 'file_sizes.json')

# Download configuration
DOWNLOAD_TIMEOUT = 14400  # 4 hours
CHUNK_TIMEOUT = 1800  # 30 minutes per chunk
MAX_RETRIES = 5
RETRY_DELAY = 300  # 5 minutes between retries
MAX_CONCURRENT_DOWNLOADS = 2  # Reduced to 2 concurrent downloads
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
RATE_LIMIT_DELAY = 60  # 1 minute delay between initial requests
MAX_CONCURRENT_REQUESTS = 1  # Rate limit the initial HEAD requests

# Speed monitoring
SPEED_CHECK_INTERVAL = 60  # Check speed every minute
MIN_SPEED_BYTES_PER_SECOND = 1024  # 1KB/s minimum speed

async def check_network_connectivity():
    """Check if network connectivity is working"""
    try:
        # Check DNS resolution
        logger.info("Checking DNS resolution...")
        result = await asyncio.create_subprocess_exec(
            'nslookup', 'wccdownload.on24.com',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        if result.returncode == 0:
            logger.info("DNS resolution working")
        else:
            logger.error(f"DNS resolution failed: {stderr.decode()}")
            return False

        # Check network connectivity with ping to on24
        logger.info("Checking network connectivity to on24...")
        result = await asyncio.create_subprocess_exec(
            'ping', '-c', '4', 'wccdownload.on24.com',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        if result.returncode == 0:
            logger.info("Network connectivity working")
            logger.info(f"Ping results: {stdout.decode()}")
        else:
            logger.error(f"Network connectivity failed: {stderr.decode()}")
            return False

        return True
    except Exception as e:
        logger.error(f"Network check failed: {str(e)}")
        return False

def get_downloaded_files() -> Set[str]:
    """Get set of already downloaded files."""
    if not os.path.exists(DOWNLOADED_FILE):
        return set()
    with open(DOWNLOADED_FILE, 'r') as f:
        return {line.strip() for line in f if line.strip()}

def get_expected_sizes() -> Dict[str, int]:
    """Get expected file sizes from the JSON file."""
    if not os.path.exists(FILE_SIZES_FILE):
        return {}
    with open(FILE_SIZES_FILE, 'r') as f:
        return json.load(f)

def is_url(path):
    """Check if the path is a URL."""
    return path.startswith(('http://', 'https://'))

def get_filename(path):
    """Extract filename from URL or path."""
    if is_url(path):
        return path.split('/')[-1]
    return os.path.basename(path)

def mark_as_downloaded(filename):
    """Mark a file as downloaded."""
    with open(DOWNLOADED_FILE, 'a') as f:
        f.write(f"\n{filename}")  # Add newline before the filename

def is_file_complete(filename: str, expected_size: int) -> bool:
    """Check if a file is completely downloaded."""
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return False
    return os.path.getsize(file_path) == expected_size

async def get_expected_file_size(url: str, session: aiohttp.ClientSession) -> Optional[int]:
    """Get the expected file size from a HEAD request."""
    try:
        async with session.head(url) as response:
            if response.status == 200:
                size = int(response.headers.get('Content-Length', 0))
                if size > 0:
                    return size
                logger.warning(f"Could not determine file size for {url}")
            else:
                logger.warning(f"HEAD request failed for {url}: HTTP {response.status}")
    except Exception as e:
        logger.error(f"Error getting file size for {url}: {str(e)}")
    return None

async def download_video(session, url, filename, retry_count: int = 0) -> bool:
    """Download a video file with retry logic."""
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    start_time = time.time()
    last_speed_check = start_time
    last_bytes_downloaded = 0
    last_progress_time = start_time
    
    try:
        logger.info(f"Starting download of {filename}")
        
        # Add delay between downloads to avoid rate limiting
        if retry_count == 0:  # Only on first attempt
            logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before starting download...")
            await asyncio.sleep(RATE_LIMIT_DELAY)
        
        timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
        async with session.get(url, timeout=timeout) as response:
            if response.status == 200:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                last_progress = 0
                stall_count = 0  # Track number of consecutive stalls
                
                with open(output_path, 'wb') as f:
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                response.content.read(CHUNK_SIZE),
                                timeout=CHUNK_TIMEOUT
                            )
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Speed monitoring - less aggressive
                            current_time = time.time()
                            if current_time - last_speed_check >= SPEED_CHECK_INTERVAL:
                                bytes_since_last_check = downloaded - last_bytes_downloaded
                                speed = bytes_since_last_check / (current_time - last_speed_check)
                                logger.info(f"Download speed for {filename}: {speed/1024:.2f} KB/s")
                                
                                if speed < MIN_SPEED_BYTES_PER_SECOND:
                                    stall_count += 1
                                    logger.warning(f"Download speed too slow for {filename}: {speed/1024:.2f} KB/s (stall count: {stall_count})")
                                    
                                    # Only fail if we've stalled multiple times
                                    if stall_count >= 3:
                                        file_sizes = get_expected_sizes()
                                        expected_size = file_sizes.get(filename)
                                        if expected_size and downloaded < expected_size:
                                            logger.error(f"Download stalled multiple times for {filename}: got {downloaded} bytes, expected {expected_size}")
                                            raise asyncio.TimeoutError("Download stalled multiple times")
                                else:
                                    stall_count = 0  # Reset stall count if speed is good
                                
                                last_speed_check = current_time
                                last_bytes_downloaded = downloaded
                            
                            if total_size > 0:
                                progress = (downloaded / total_size) * 100
                                current_time = time.time()
                                
                                # Only log if integer percent increased
                                if int(progress) > int(last_progress):
                                    last_progress = progress
                                    last_progress_time = current_time
                                    logger.info(f"Download progress for {filename}: {int(progress)}% (elapsed: {int(current_time - start_time)}s)")
                                elif current_time - last_progress_time > CHUNK_TIMEOUT:
                                    # No progress for too long, check if we're stalled
                                    file_sizes = get_expected_sizes()
                                    expected_size = file_sizes.get(filename)
                                    if expected_size and downloaded < expected_size:
                                        stall_count += 1
                                        logger.error(f"Download stalled for {filename}: got {downloaded} bytes, expected {expected_size} (stall count: {stall_count})")
                                        if stall_count >= 3:
                                            raise asyncio.TimeoutError("Download stalled multiple times")
                        except asyncio.TimeoutError:
                            logger.error(f"Chunk download timeout for {filename}")
                            raise

                logger.info(f"Successfully downloaded {filename}")
                mark_as_downloaded(filename)
                return True
            else:
                error_msg = f"Failed to download {filename}: HTTP {response.status}"
                if response.status == 404:
                    error_msg += " - File not found"
                elif response.status == 403:
                    error_msg += " - Access forbidden"
                elif response.status == 401:
                    error_msg += " - Authentication required"
                logger.error(error_msg)
                return False
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Network error downloading {filename}: {str(e)}")
        if retry_count < MAX_RETRIES - 1:
            # Exponential backoff for retries
            retry_delay = RETRY_DELAY * (2 ** retry_count)
            logger.info(f"Retrying download of {filename} in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
            return await download_video(session, url, filename, retry_count + 1)
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {filename}: {str(e)}")
        return False

async def download_videos(limit: Optional[int] = None) -> None:
    """Download videos from the list that haven't been downloaded yet."""
    # Create directories if they don't exist
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Get already downloaded files
    downloaded_files = get_downloaded_files()
    logger.info(f"Found {len(downloaded_files)} already downloaded files")
    
    # Read the download list
    with open(DOWNLOAD_LIST_FILE, 'r') as f:
        video_urls = [line.strip() for line in f if line.strip()]
    
    # Filter for URLs only and exclude already downloaded files
    video_urls = [url for url in video_urls if is_url(url)]
    video_urls = [url for url in video_urls if get_filename(url) not in downloaded_files]
    
    if not video_urls:
        logger.info("No new videos to download")
        return
    
    # Apply limit if specified
    if limit is not None:
        video_urls = video_urls[:limit]
        logger.info(f"Downloading {len(video_urls)} videos (limited)")
    else:
        logger.info(f"Downloading {len(video_urls)} videos")
    
    # Create a semaphore to limit concurrent downloads
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    
    async def download_with_semaphore(url):
        async with semaphore:
            filename = get_filename(url)
            return await download_video(session, url, filename)
    
    # Start downloads with rate limiting
    timeout = aiohttp.ClientTimeout(total=None)  # No timeout for the session
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Process videos in batches
        for i in range(0, len(video_urls), MAX_CONCURRENT_DOWNLOADS):
            batch = video_urls[i:i + MAX_CONCURRENT_DOWNLOADS]
            tasks = [download_with_semaphore(url) for url in batch]
            await asyncio.gather(*tasks)
            
            # Add a delay between batches to avoid overwhelming the server
            if i + MAX_CONCURRENT_DOWNLOADS < len(video_urls):
                logger.info(f"Waiting {RATE_LIMIT_DELAY} seconds before starting next batch...")
                await asyncio.sleep(RATE_LIMIT_DELAY)

async def main():
    """Main entry point."""
    args = parse_args()
    await download_videos(args.limit)

def parse_args():
    parser = argparse.ArgumentParser(description='Download videos.')
    parser.add_argument('--limit', type=int, help='Limit the number of videos to process')
    return parser.parse_args()

if __name__ == '__main__':
    asyncio.run(main()) 