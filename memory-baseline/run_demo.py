
from pinak.memory.manager import MemoryManager
import time

def run_demo():
    """Runs a full demonstration of the MemoryManager."""
    print("\n--- Starting MemoryManager Demo ---")
    try:
        mem = MemoryManager()
    except Exception as e:
        print(f"\n[ERROR] Could not initialize MemoryManager: {e}")
        print("Please ensure Redis is installed and running ('brew services start redis').")
        return

    # 1. Auto-seeding initial memories
    print("\n--- Step 1: Auto-seeding initial memories ---")
    initial_memories = [
        "The user prefers Python for scripting and data analysis.",
        "The agent's name is Vayu.",
        "The primary goal is to assist the user with software engineering tasks efficiently.",
        "The user values robust, well-documented code."
    ]
    for memory in initial_memories:
        mem.add_memory(memory)
    print("\nAuto-seeding complete.")

    # 2. Retrieving a memory
    print("\n--- Step 2: Retrieving a relevant memory ---")
    query = "What is my preferred language for scripts?"
    results = mem.retrieve_memory(query, k=1)
    print(f"QUERY: '{query}'")
    if results:
        print(f"FOUND: '{results[0]['content']}' (Distance: {results[0]['distance']:.2f})")
    else:
        print("FOUND: No relevant memory.")

    # 3. Using Working Memory (Redis)
    print("\n--- Step 3: Testing Working Memory (Redis) ---")
    mem.set_working_memory("current_task_id", "task-12345")
    retrieved_task = mem.get_working_memory("current_task_id")
    print(f"Set 'current_task_id' to 'task-12345'.")
    print(f"Retrieved from Redis: '{retrieved_task}'")
    assert retrieved_task == "task-12345"
    print("Working memory test passed.")

    # 4. Testing Delete Guardrail
    print("\n--- Step 4: Testing Delete Guardrail (Soft Delete) ---")
    memory_to_delete_content = "This is a temporary memory."
    memory_id_to_delete = mem.add_memory(memory_to_delete_content)
    print(f"Added a temporary memory with ID: {memory_id_to_delete}")
    
    # Retrieve it to show it exists
    results_before_delete = mem.retrieve_memory(memory_to_delete_content, k=1)
    assert results_before_delete and results_before_delete[0]['id'] == memory_id_to_delete
    print("Retrieved the memory successfully before deletion.")

    # Soft-delete it
    mem.delete_memory(memory_id_to_delete)

    # Try to retrieve it again
    results_after_delete = mem.retrieve_memory(memory_to_delete_content, k=1)
    if not results_after_delete or results_after_delete[0]['id'] != memory_id_to_delete:
        print("Successfully verified that soft-deleted memory is not retrieved.")
    else:
        print("[ERROR] Soft-deleted memory was still retrieved.")
    print("Delete guardrail test passed.")

    # 5. Testing Purge
    print("\n--- Step 5: Testing Permanent Purge ---")
    initial_count = mem.index.ntotal
    mem.purge_deleted_memories()
    final_count = mem.index.ntotal
    print(f"Initial memory count: {initial_count}, Final memory count: {final_count}")
    assert final_count < initial_count
    print("Purge test passed.")

    print("\n--- Demo completed successfully! ---")

if __name__ == "__main__":
    run_demo()
