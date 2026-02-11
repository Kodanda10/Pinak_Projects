import os
import sys
import logging
import shutil

# Ensure we can import app
sys.path.append(os.getcwd())

from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_agent_flow():
    # Setup clean environment
    data_dir = "data_test_repro"
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    os.environ["PINAK_CONFIG_PATH"] = "app/core/config.json"
    os.environ["PINAK_EMBEDDING_BACKEND"] = "dummy" # Use deterministic encoder

    # Initialize Service
    logger.info("Initializing Memory Service...")
    # Mock config to point to our test data dir
    service = MemoryService()
    service.data_root = data_dir
    service.db_path = os.path.join(data_dir, "memory.db")
    service.vector_path = os.path.join(data_dir, "vectors.index.npy")
    # Re-init components with new paths
    from app.core.database import DatabaseManager
    from app.services.vector_store import VectorStore
    service.db = DatabaseManager(service.db_path)
    if service.vector_enabled:
         service.vector_store = VectorStore(service.vector_path, service.embedding_dim)

    tenant = "test_tenant"
    project = "test_project"
    client_id = "test_agent"

    # 1. Add Memory
    logger.info("Adding Memory...")
    content = "The secret code is 12345."
    mem = MemoryCreate(content=content, tags=["secret"])
    res = service.add_memory(mem, tenant, project, client_id=client_id)
    logger.info(f"Added Memory ID: {res.id}")

    # 2. Search Memory (Vector)
    logger.info("Searching Memory (Vector)...")
    # Since we use dummy encoder, "secret" and "code" will hash to vectors.
    # The dummy encoder is random based on string hash, so "The secret code is 12345."
    # should be found by a query that generates a similar vector?
    # Actually, _DeterministicEncoder makes completely different vectors for different strings.
    # So "The secret code" will have a totally different vector than "What is the secret code?".
    # This means Vector Search with Dummy Encoder is useless for semantic similarity testing.
    # It only tests EXACT string matches if we pass the exact same string,
    # OR we rely on Hybrid Search's keyword part.

    # Let's test Hybrid Search with a keyword match, which should work.
    query = "secret code"
    results = service.search_memory(query, tenant, project, k=5)

    found = False
    for r in results:
        logger.info(f"Result: {r.content} (Score/Dist: {r.distance})")
        if "12345" in r.content:
            found = True

    if found:
        logger.info("SUCCESS: Memory found via Hybrid Search.")
    else:
        logger.error("FAILURE: Memory not found.")
        sys.exit(1)

    # 3. Test Working Memory
    logger.info("Testing Working Memory...")
    service.working_add("Thinking about deployment...", tenant, project)
    working_list = service.working_list(tenant, project)

    found_working = False
    for w in working_list:
        if "deployment" in w.get("value", ""):
             found_working = True
             break

    if found_working:
        logger.info("SUCCESS: Working memory updated.")
    else:
        logger.error("FAILURE: Working memory check failed.")
        sys.exit(1)

    # Cleanup
    shutil.rmtree(data_dir)

if __name__ == "__main__":
    test_agent_flow()
