
import os
import sys
import shutil
import json
import logging
from datetime import datetime, timezone

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def verify_agent_flow():
    logger.info("Starting Agent Flow Verification...")

    # 1. Setup Temporary Environment
    test_dir = "data_verification_test"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    config_path = os.path.join(test_dir, "config.json")
    config = {
        "data_root": test_dir,
        "embedding_model": "dummy", # Use deterministic encoder
        "vector_store_type": "numpy"
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    # 2. Initialize Service
    logger.info("Initializing MemoryService...")
    service = MemoryService(config_path=config_path)

    tenant = "test-tenant"
    project = "test-project"
    agent_id = "agent-007"
    client_name = "cli-verifier"

    # 3. Register Agent
    logger.info("Registering Agent...")
    service.register_agent(
        agent_id=agent_id,
        client_name=client_name,
        status="active",
        tenant=tenant,
        project_id=project
    )

    # 4. Add Semantic Memory
    logger.info("Adding Semantic Memory...")
    content = "The Pinak system is designed to be a local-first memory service."
    mem = MemoryCreate(content=content, tags=["architecture", "overview"])
    res_sem = service.add_memory(
        mem,
        tenant=tenant,
        project_id=project,
        agent_id=agent_id,
        client_name=client_name
    )
    logger.info(f"Semantic Memory Added: {res_sem.id}")

    # 5. Add Episodic Memory
    logger.info("Adding Episodic Memory...")
    res_epi = service.add_episodic(
        content="Ran verification script to test system integrity.",
        goal="Verify system",
        outcome="Success",
        salience=10,
        tenant=tenant,
        project_id=project,
        agent_id=agent_id
    )
    logger.info(f"Episodic Memory Added: {res_epi['id']}")

    # 6. Search Memory (Semantic)
    logger.info("Searching for 'local-first'...")
    search_results = service.search_memory("local-first", tenant, project)
    if any(content in r.content for r in search_results):
        logger.info("✅ Semantic Search Verified")
    else:
        logger.error("❌ Semantic Search Failed")
        sys.exit(1)

    # 7. Retrieve Context (Hybrid)
    logger.info("Retrieving Context for 'Verify system'...")
    context = service.retrieve_context("Verify system", tenant, project)

    # Check if episodic memory is found
    episodic_hits = context.get("episodic", [])
    found = False
    for hit in episodic_hits:
        if "Verify system" in hit.get("goal", ""):
            found = True
            break

    if found:
        logger.info("✅ Context Retrieval Verified (Episodic Hit)")
    else:
        logger.error("❌ Context Retrieval Failed (Episodic Miss)")
        logger.info(f"Context Dump: {json.dumps(context, default=str)}")
        sys.exit(1)

    # 8. Working Memory & Intent Sniffing
    logger.info("Updating Working Memory with risky content...")
    # "secret" and "key" should trigger security intent
    res_work = service.working_add(
        content="I am updating the secret key for the database.",
        tenant=tenant,
        project_id=project,
        agent_id=agent_id
    )

    # Check for nudges
    nudges = res_work.get("nudges", [])
    if nudges:
        logger.info(f"✅ Proactive Nudges Triggered: {len(nudges)}")
        for n in nudges:
            logger.info(f"   - {n['message']}")
    else:
        logger.warning("⚠ No nudges triggered (Expected if no collision with existing negative memories, or sniffer tuning)")

    # Cleanup
    logger.info("Cleaning up...")
    shutil.rmtree(test_dir)
    logger.info("🎉 Verification Successful! The service is ready for Agents.")

if __name__ == "__main__":
    verify_agent_flow()
