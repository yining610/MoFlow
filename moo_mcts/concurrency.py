from concurrent.futures import ThreadPoolExecutor


def parallel_map(fn, items: list, max_workers: int = 8) -> list:
    """Map `fn` over `items` concurrently, returning results in input order."""
    if len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(len(items), max_workers)) as pool:
        return list(pool.map(fn, items))
