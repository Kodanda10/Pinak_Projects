import time
import numpy as np
import threading
import concurrent.futures
import os
import sqlite3
import random
import logging
from app.services.vector_store import VectorStore
from app.services.faiss_vector_store import FaissVectorStore
from app.core.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_vector_store(cls, name, num_vectors=10000, dim=384, query_count=100):
    logger.info(f"--- Benchmarking {name} with {num_vectors} vectors ---")

    # Use a temp directory
    ext = "index" if "Faiss" in name else "npy"
    index_path = f"bench_vectors_{num_vectors}.{ext}"
    if os.path.exists(index_path):
        os.remove(index_path)

    vs = cls(index_path, dim)

    # Generate vectors
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = list(range(num_vectors))

    # Add vectors
    start_time = time.time()
    if name == "FaissVectorStore":
        # Faiss might need batching or is fast enough
        vs.add_vectors(vectors, ids)
    else:
        vs.add_vectors(vectors, ids)

    vs.save()
    add_time = time.time() - start_time
    logger.info(f"Added {num_vectors} vectors in {add_time:.4f}s ({num_vectors/add_time:.2f} vec/s)")

    # Query vectors
    query_vectors = np.random.rand(query_count, dim).astype(np.float32)
    start_time = time.time()
    for q in query_vectors:
        vs.search(q, k=10)
    query_time = time.time() - start_time
    avg_query_time = query_time / query_count
    logger.info(f"Performed {query_count} queries in {query_time:.4f}s (Avg: {avg_query_time*1000:.2f}ms/query)")

    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)

def benchmark_database_concurrency(num_threads=10, ops_per_thread=50):
    logger.info(f"--- Benchmarking Database Concurrency with {num_threads} threads ---")

    db_path = "bench_db.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_path)

    def worker(tid):
        for i in range(ops_per_thread):
            try:
                # Use semantic add which writes to DB and FTS
                db.add_semantic(f"content-{tid}-{i}", ["tag"], "bench_tenant", "bench_proj", 123)
            except Exception as e:
                logger.error(f"Thread {tid} failed: {e}")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    total_time = time.time() - start_time
    total_ops = num_threads * ops_per_thread
    logger.info(f"Performed {total_ops} DB writes in {total_time:.4f}s ({total_ops/total_time:.2f} ops/s)")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    benchmark_vector_store(VectorStore, "NumPyVectorStore")
    benchmark_vector_store(FaissVectorStore, "FaissVectorStore")
    benchmark_database_concurrency()
