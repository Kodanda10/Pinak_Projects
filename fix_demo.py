with open('demo_world_beating.py', 'r') as f:
    content = f.read()

# Fix the duplicate print issue
content = content.replace(
    '        print("✅ World-beating system initialized successfully!")\n        print(\n        print(\n            f"📊 Graph expansion nodes: {self.engine.graph_expander.knowledge_graph.get_statistics()}"\n        )\n    async def demonstrate_stage_1_intent_analysis(self):',
    '        print("✅ World-beating system initialized successfully!")\n        print(\n            f"📊 Graph expansion nodes: {self.engine.graph_expander.knowledge_graph.get_statistics()}"\n        )\n\n    async def demonstrate_stage_1_intent_analysis(self):'
)

with open('demo_world_beating.py', 'w') as f:
    f.write(content)

print('Fixed demo_world_beating.py')
