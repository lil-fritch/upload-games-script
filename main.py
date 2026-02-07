import sqlite3
import asyncio
import aiohttp
import json
import logging
import datetime
import time
import os
import argparse
from typing import List, Dict, Any
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMServerFailure(Exception):
    """Custom exception raised when LLM server has too many consecutive errors."""
    pass

class AsyncLLMClient:
    def __init__(self, api_url=None, api_key=None, model=None):
        self.api_url = api_url or os.getenv("LLM_API_URL", "http://localhost:11434")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL_FAST", "llama3.2:latest")
        self.delay = float(os.getenv("LLM_DELAY", "1.0"))
        
        self.consecutive_errors = 0
        self.max_consecutive_errors = int(os.getenv("LLM_MAX_ERRORS", "10"))

    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        # Circuit breaker check
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise LLMServerFailure(f"Stopped after {self.consecutive_errors} consecutive LLM errors.")

        # Add delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        is_chat_completions = "/v1/chat/completions" in self.api_url or "/completions" in self.api_url
        
        headers = {
            "Content-Type": "application/json",
            "Connection": "close"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if is_chat_completions:
            endpoint = self.api_url
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature
            }
        else:
            # Legacy Ollama
            if self.api_url.endswith("/api/generate"):
                 endpoint = self.api_url
            else:
                 endpoint = f"{self.api_url}/api/generate"
                 
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": temperature
            }

        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload, headers=headers) as response:
                    if response.status != 200:
                        self.consecutive_errors += 1
                        logger.error(f"LLM Error {response.status}: {await response.text()}")
                        
                        if self.consecutive_errors >= self.max_consecutive_errors:
                            raise LLMServerFailure(f"Too many consecutive errors ({self.consecutive_errors})")
                        return None
                    
                    # Reset error count on success
                    self.consecutive_errors = 0
                    
                    data = await response.json()
                    
                    if is_chat_completions:
                        result = data.get("choices", [])[0].get("message", {}).get("content", "")
                    else:
                        result = data.get("response", "")
                    
                    return result

        except LLMServerFailure:
            # Re-raise explicit server failure
            raise
        except Exception as e:
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise LLMServerFailure(f"Too many consecutive errors ({self.consecutive_errors}) - Last error: {e}")
            
            logger.error(f"Async LLM Generation failed: {e}")
            return None


DB_PATH = os.getenv("DB_PATH", "data/db/slotslaunch.db")
STRAPI_API_URL = os.getenv("STRAPI_API_URL", "https://strapi.safercase.app/api/games")
STRAPI_API_TOKEN = os.getenv("STRAPI_API_TOKEN", "")
SLOTS_LAUNCH_TOKEN = os.getenv("SLOTS_LAUNCH_TOKEN", "")
STATE_FILE = "migration_progress.txt"
CONCURRENCY_LIMIT = 12


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed_slugs() -> set:
    if not os.path.exists(STATE_FILE):
        return set()
    try:
        with open(STATE_FILE, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logger.error(f"Error loading state file: {e}")
        return set()

def mark_as_processed(slug: str):
    try:
        with open(STATE_FILE, 'a') as f:
            f.write(f"{slug}\n")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def fetch_games_from_db() -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM games")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching from DB: {e}")
        return []


async def generate_description_async(client: AsyncLLMClient, game: Dict[str, Any]) -> str:
    """Async wrapper using AsyncLLMClient."""
    
    name = game.get('name', 'Unknown Game')
    provider = game.get('provider', 'Unknown Provider')
    original_desc = game.get('description', '')
    game_type = game.get('type', '')
    rtp = game.get('rtp', '')
    min_bet = game.get('min_bet', '')
    max_bet = game.get('max_bet', '')
    max_win = game.get('max_win_per_spin', '')
    
    context_lines = [
        f"Name: {name}",
        f"Provider: {provider}",
    ]
    if original_desc: context_lines.append(f"Base Description: {original_desc}")
    if game_type: context_lines.append(f"Type: {game_type}")
    if rtp: context_lines.append(f"RTP: {rtp}%")
    if min_bet and max_bet: context_lines.append(f"Bet Range: {min_bet} - {max_bet}")
    elif min_bet: context_lines.append(f"Min Bet: {min_bet}")
    if max_win: context_lines.append(f"Max Win: {max_win}")
    
    context_block = "\n".join(context_lines)
    
    prompt = f"""
    Write a short, factual summary for the online casino game "{name}" based on the following features:
    
    {context_block}
    
    Include key statstics like RTP or Max Win if present.
    Mention it is available to play.
    Do not include markdown or quotes.
    """
    
    try:
        description = await client.generate(prompt, 0.7)
        if not description:
             fallback = f"{name} is a {game_type or 'casino'} game by {provider}"
             if rtp: fallback += f" with {rtp}% RTP"
             fallback += "."
             return fallback
        return description.strip().replace('"', '')
    except LLMServerFailure:
        raise
    except Exception as e:
        logger.error(f"LLM Error for {name}: {e}")
        return f"Play {name} by {provider}."

async def check_if_exists(session: aiohttp.ClientSession, slug: str) -> bool:
    """Fast check if game exists in Strapi before processing."""
    try:
        headers = {"Authorization": f"Bearer {STRAPI_API_TOKEN}"}
        url = f"{STRAPI_API_URL}?filters[slug][$eq]={slug}&fields[0]=slug"
        
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                # If we get any data, the game exists
                return len(data.get('data', [])) > 0
            return False
    except Exception as e:
        logger.error(f"Check Exists Error {slug}: {e}")
        return False

async def upload_game_async(session: aiohttp.ClientSession, payload: Dict[str, Any]) -> str:
    """Async upload to Strapi."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STRAPI_API_TOKEN}",
        "User-Agent": "Python/3.9 aiohttp/3.8"
    }
    
    try:
        body = {"data": payload}
        async with session.post(STRAPI_API_URL, json=body, headers=headers) as response:
            if response.status in [200, 201]:
                return "success"
            
            text = await response.text()
            if response.status == 400 and "must be unique" in text:
                return "skipped"
            
            logger.error(f"FAIL: '{payload['title']}' - Status: {response.status} - RESPONSE: {text}")
            return "failed"
    except Exception as e:
        logger.error(f"Upload Error: {e}")
        return "failed"

def prepare_payload(game, desc):
    categories = []
    g_type = game.get('type')
    if g_type: categories.append(str(g_type).lower().replace(' ', '-'))
    prov = game.get('provider')
    if prov: categories.append(str(prov).lower().replace(' ', '-'))
    if not categories: categories = ["others"]
    categories = list(set(categories))

    tags = ["online-casino"] 
    slug = game.get('slug')
    if slug: tags.append(slug)
    rtp_raw = game.get('rtp')
    if rtp_raw:
        try:
            rtp_val = float(str(rtp_raw).replace('%', '').strip())
            if rtp_val > 96.0: tags.append("high-rtp")
        except ValueError: pass
    tags = list(set(tags))
        
    return {
        "title": game.get('name'),
        "meta_title": game.get('name'),
        "description": desc,
        "date": datetime.datetime.now().isoformat(),
        "image": game.get('thumb'),
        "categories": categories,
        "tags": tags,        
        "slug": game.get('slug'),
        "iframe_url": f"{game.get('url')}?token={SLOTS_LAUNCH_TOKEN}" if game.get('url') else None,
        "provider": game.get('provider'),
        "content": desc,
        "draft": False
    }

async def process_one_game(semaphore, session, client, game):
    name = game.get('name')
    slug = game.get('slug')
    
    async with semaphore:
        desc = await generate_description_async(client, game)
        
        payload = prepare_payload(game, desc)
        status = await upload_game_async(session, payload)
        
        if status == "success" or status == "skipped":
            if status == "success": logger.info(f"SUCCESS: {name}")
            else: logger.info(f"SKIPPED (API): {name} (already exists)")
            mark_as_processed(slug)
        
        return status

async def main_async_full(limit=0):
    logger.info("Starting ASYNC Full Migration...")
    
    try:
        llm_client = AsyncLLMClient()
    except Exception as e:
        logger.critical(f"LLM Init Failed: {e}")
        return

    games = fetch_games_from_db()
    processed_slugs = load_processed_slugs()
    games_to_process = [g for g in games if g.get('slug') not in processed_slugs]
    
    if limit > 0:
        games_to_process = games_to_process[:limit]
        
    logger.info(f"Processing {len(games_to_process)} games with concurrency {CONCURRENCY_LIMIT}")

    # Async Loop
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    start_time = time.time()
    total_games = len(games_to_process)
    completed_count = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        # ProgressBar
        pbar = tqdm(total=total_games, desc="Uploading Games", unit="game")

        # Create a wrapper to track progress
        async def tracked_process(game):
            status = await process_one_game(semaphore, session, llm_client, game)
            pbar.update(1)
            pbar.set_postfix_str(f"Last: {status}")
            return status

        for game in games_to_process:
            task = asyncio.create_task(tracked_process(game))
            tasks.append(task)
        
        # Gather results with circuit breaker support
        try:
            results = await asyncio.gather(*tasks)
        except LLMServerFailure as e:
            pbar.close()
            logger.critical(f"\n\n!!! ABORTED MIGRATION !!!")
            logger.critical(f"Reason: {e}")
            logger.critical("Cancelling pending tasks...")
            for t in tasks:
                if not t.done():
                    t.cancel()
            return

        pbar.close()
            
    success = results.count("success")
    skipped = results.count("skipped")
    failed = results.count("failed")
    
    print(f"\nMigration Complete in {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
    print(f"Uploaded: {success}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="full")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "full":
        asyncio.run(main_async_full(args.limit))
    else:
        print("Legacy modes removed in async version. Use --mode full")
