import logging
import os
from datetime import datetime

import pandas as pd

logger = logging.getLogger("result")

CACHE_PATH_BL = "pdll/caching/baseline_cache.parquet"
CACHE_PATH_PW = "pdll/caching/pairwise_cache.parquet"

cache_BL = pd.DataFrame()
cache_PW = pd.DataFrame()

cache_stats_BL = {"hits": 0, "misses": 0}
cache_stats_PW = {"hits": 0, "misses": 0}


def load_cache(is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PW
    else:
        cache_path = CACHE_PATH_BL

    if os.path.exists(cache_path):
        read_cache_file(cache_path, is_pairwise)
    else:
        initialize_new_cache(is_pairwise)
    logger.debug("loaded cache")


def read_cache_file(cache_path, is_pairwise: bool):
    global cache_PW, cache_BL
    if is_pairwise:
        cache_PW = pd.read_parquet(cache_path)
    else:
        cache_BL = pd.read_parquet(cache_path)
    logger.debug("read cache file")


def initialize_new_cache(is_pairwise: bool):
    cache_df = pd.DataFrame(columns=["model", "prompt", "score", "timestamp"])
    global cache_PW, cache_BL
    if is_pairwise:
        cache_PW = cache_df
    else:
        cache_BL = cache_df
    logger.debug("initialized new cache")


def save_cache(cache: pd.DataFrame, is_pairwise: bool):
    if is_pairwise:
        cache_path = CACHE_PATH_PW
    else:
        cache_path = CACHE_PATH_BL
    cache.to_parquet(cache_path, index=False)
    logger.debug("saved cache file")


def lookup_in_cache(model: str, prompt: str, is_pairwise: bool):
    logger.debug("looked-up prompt in cache")
    if is_pairwise:
        global cache_PW, cache_stats_PW
        cached_row = cache_PW[
            (cache_PW["prompt"] == prompt) & (cache_PW["model"] == model)
        ]
        if not cached_row.empty:
            cache_stats_PW["hits"] += 1
            return cached_row.iloc[0]["score"]
        else:
            cache_stats_PW["misses"] += 1

    else:
        global cache_BL, cache_stats_BL
        cached_row = cache_BL[
            (cache_BL["prompt"] == prompt) & (cache_BL["model"] == model)
        ]
        if not cached_row.empty:
            cache_stats_BL["hits"] += 1
            return cached_row.iloc[0]["score"]
        else:
            cache_stats_BL["misses"] += 1


def new_cache_entry(model: str, prompt: str, score, is_pairwise: bool):
    timestamp = datetime.now().isoformat()
    if is_pairwise:
        global cache_PW
        new_entry = pd.DataFrame(
            [[model, prompt, score, timestamp]],
            columns=["model", "prompt", "score", "timestamp"],
        )
        cache_PW = pd.concat([cache_PW, new_entry], ignore_index=True)
        save_cache(cache_PW, is_pairwise)
    else:
        global cache_BL
        new_entry = pd.DataFrame(
            [[model, prompt, score, timestamp]],
            columns=["model", "prompt", "score", "timestamp"],
        )
        cache_BL = pd.concat([cache_BL, new_entry], ignore_index=True)
        save_cache(cache_BL, is_pairwise)
    logger.debug("created new cache entry")


def print_cache_stats(is_pairwise: bool):
    if is_pairwise:
        cache_stats = cache_stats_PW
        logger.info("Pairwise Cache-Stats:")
    else:
        cache_stats = cache_stats_BL
        logger.info("Baseline Cache-Stats:")
    logger.info(f"Cache hits: {cache_stats['hits']}")
    logger.info(f"Cache misses: {cache_stats['misses']}\n")
    logger.debug("printed cache stats\n")


load_cache(True)
print(cache_PW.tail(50))

essay = "I personally believe that computersdo benefit society in a lot of ways. It can give you all the information you need with just a click of a button, entertainment is easy to find by searching the internet, and most importantly, it improves communication between people. One way computers benefit society is, by making it easier for everyone to get information about something. This is needed, especially in school. It can help kids get their information about the person they are writing about in their report.Getting information easily on the internet can also help people in other ways. What if someone needed to get the hospital's phone number in case of an emergency? He can use the computer to help him find that phone number. Another way computers benefit society is by allowing us to get any form of entertainment easily. This is what majority of the people in our society rely on. Alot of people use the computer to play games, listen to music, and watch online videos. If someone was bored right now, he or she could use the computer to play fun games. What if that person thought it was a little quiet in his or her room? He or she could then use the computer to listen to music. Without computers we would probably be really bored for the rest of our lives. Finally, and most importantly, computers help improve communication between people. This is probably what everyone uses the computer for. Now you can send electronic mail to your friends in just seconds, post and view pictures of your family or relatives and you can even use an instant messanger to write to your friend and get messages from your friend in the fastest speed possible! Without computers, communication between others would be very hard. As you can see, computers can give you information easily, entertainment is not hard to find, and it can even improve communication between people around our society and the world.! Computers are very helpful to us and improves everything about the way we live."

cached_row = cache_PW[
    (essay in cache_PW["prompt"]) & (cache_PW["model"] == "gpt-4o-mini")
]

print(cached_row)
