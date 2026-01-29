---
name: ironclad-testing
description: Enforces 0.01% Apple-level testing standards. LOAD FIRST before any code generation. Mandates 95%+ coverage, mutation testing, chaos engineering, property-based tests, and formal invariant verification. Use for ALL projects requiring production-grade reliability.
---

# Ironclad Testing Protocol

**MANDATORY**: Load this skill FIRST before writing ANY code. This protocol ensures code that never breaks.

## Prime Directives

```
1. NO CODE WITHOUT TEST PLAN — Define test strategy before implementation
2. NO MERGE WITHOUT 95%+ COVERAGE — Line AND branch coverage required
3. NO DEPLOY WITHOUT CHAOS TESTING — Fault injection is mandatory
4. NO RELEASE WITHOUT MUTATION TESTING — Tests must catch injected bugs
5. PROVE INVARIANTS — Critical properties must be formally verified
```

## Coverage Requirements

| Metric | Minimum | Target |
|--------|---------|--------|
| Line Coverage | 95% | 100% |
| Branch Coverage | 90% | 95% |
| Mutation Score | 90% | 95% |
| Flaky Test Rate | 0% | 0% |

### Coverage Configuration (Vitest)

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      thresholds: {
        lines: 95,
        branches: 90,
        functions: 95,
        statements: 95
      },
      exclude: ['**/*.test.ts', '**/*.spec.ts', '**/index.ts']
    }
  }
});
```

### Coverage Configuration (Jest)

```javascript
// jest.config.js
module.exports = {
  coverageThreshold: {
    global: {
      branches: 90,
      functions: 95,
      lines: 95,
      statements: 95
    }
  }
};
```

## Test Pyramid

```
                    ┌─────────────┐
                    │   E2E (10%) │  ← Full system scenarios
                   ─┴─────────────┴─
                 ┌───────────────────┐
                 │ Integration (20%) │  ← Component interactions
                ─┴───────────────────┴─
              ┌───────────────────────────┐
              │      Unit Tests (70%)     │  ← Isolated logic
              └───────────────────────────┘
```

## Mandatory Test Categories

### 1. Unit Tests (Every Function)

```typescript
describe('function', () => {
  it('handles happy path', () => {});
  it('handles edge case: empty input', () => {});
  it('handles edge case: null/undefined', () => {});
  it('handles edge case: boundary values', () => {});
  it('handles error case: invalid input', () => {});
  it('handles error case: dependency failure', () => {});
});
```

### 2. Property-Based Tests (Critical Logic)

```typescript
import fc from 'fast-check';

it('property: function is idempotent', () => {
  fc.assert(fc.property(fc.anything(), (input) => {
    const result1 = fn(input);
    const result2 = fn(input);
    expect(result1).toEqual(result2);
  }));
});

it('property: never violates invariant', () => {
  fc.assert(fc.property(fc.anything(), (input) => {
    const result = fn(input);
    expect(invariantHolds(result)).toBe(true);
  }), { numRuns: 10000 });
});
```

### 3. Chaos Tests (Fault Injection)

```typescript
describe('chaos', () => {
  it('survives random network latency', async () => {
    injectLatency({ min: 100, max: 5000, random: true });
    await expect(operation()).resolves.not.toThrow();
  });

  it('survives process kill mid-operation', async () => {
    const op = operation();
    setTimeout(() => process.kill(process.pid, 'SIGTERM'), randomMs());
    // System must recover cleanly
  });

  it('survives clock jump backward', async () => {
    mockTime.jump(-30000); // 30 seconds backward
    await expect(timeBasedOperation()).resolves.not.toThrow();
  });

  it('survives corrupted input', async () => {
    const corrupted = corruptPayload(validPayload);
    await expect(parsePayload(corrupted)).rejects.toThrow();
    // Must not crash, must throw cleanly
  });
});
```

### 4. Concurrency Tests (Race Conditions)

```typescript
describe('concurrency', () => {
  it('handles simultaneous access correctly', async () => {
    const results = await Promise.all(
      Array(100).fill(null).map(() => concurrentOperation())
    );
    expect(invariant(results)).toBe(true);
  });

  it('handles interleaved operations', async () => {
    for (let seed = 0; seed < 1000; seed++) {
      const interleaving = generateInterleaving(seed);
      const result = await executeWithInterleaving(interleaving);
      expect(result).toSatisfyInvariant();
    }
  });
});
```

### 5. Integration Tests (Real Components)

```typescript
describe('integration', () => {
  // NO MOCKS for integration tests
  let realComponentA: ComponentA;
  let realComponentB: ComponentB;

  beforeEach(() => {
    realComponentA = new ComponentA();
    realComponentB = new ComponentB(realComponentA);
  });

  it('components interact correctly under load', async () => {
    const ops = Array(1000).fill(null).map(() => 
      realComponentB.operation()
    );
    await Promise.all(ops);
    expect(realComponentA.state).toBeConsistent();
  });
});
```

### 6. E2E Scenario Tests

Every system must test these failure scenarios:

```typescript
const MANDATORY_SCENARIOS = [
  'normal-operation',
  'primary-crash-recovery',
  'network-partition',
  'data-corruption-recovery',
  'clock-skew-handling',
  'resource-exhaustion',
  'concurrent-access-race',
  'graceful-degradation',
  'failover-and-failback',
  'long-haul-stability'
];
```

## Mutation Testing

### Setup (Stryker)

```bash
npm install --save-dev @stryker-mutator/core @stryker-mutator/typescript-checker
```

```javascript
// stryker.conf.js
module.exports = {
  mutate: ['src/**/*.ts', '!src/**/*.test.ts'],
  testRunner: 'vitest',
  thresholds: { high: 95, low: 90, break: 85 }
};
```

### Interpretation

| Score | Meaning |
|-------|---------|
| 95%+ | Production ready |
| 90-95% | Acceptable with review |
| <90% | NOT ACCEPTABLE — add more tests |

## Stability Testing Protocol

### Before Every Commit

```bash
# Run 100 times, fail on ANY failure
for i in {1..100}; do
  npm test || exit 1
done
```

### Before Every Deploy

```bash
# Run 1000 times with random seeds
for i in {1..1000}; do
  SEED=$RANDOM npm test || exit 1
done
```

### Weekly Long-Haul Test

```bash
# Run system for 1 hour under continuous load
npm run test:longhail -- --duration=3600
```

## Invariant Verification

For critical systems, define and verify invariants:

```typescript
// Define invariants as executable assertions
const INVARIANTS = {
  'single-leader': () => {
    const leaders = getAllNodes().filter(n => n.isLeader);
    return leaders.length <= 1;
  },
  'no-data-loss': () => {
    const written = getWrittenRecords();
    const stored = getStoredRecords();
    return written.every(w => stored.includes(w));
  },
  'monotonic-epoch': () => {
    const epochs = getEpochHistory();
    return epochs.every((e, i) => i === 0 || e > epochs[i-1]);
  }
};

// Run after EVERY operation in tests
afterEach(() => {
  Object.entries(INVARIANTS).forEach(([name, check]) => {
    expect(check(), `Invariant violated: ${name}`).toBe(true);
  });
});
```

## Code Review Checklist

Before approving ANY code:

```
□ Line coverage ≥ 95%
□ Branch coverage ≥ 90%
□ Mutation score ≥ 90%
□ Zero flaky tests (100 runs)
□ All edge cases covered
□ All error paths tested
□ Chaos tests pass
□ Concurrency tests pass
□ Integration tests pass
□ Invariants verified
□ No TODO/FIXME in tests
□ No skipped tests
□ No test timeouts > 5s
```

## Anti-Patterns to Reject

| Anti-Pattern | Why It's Bad |
|--------------|--------------|
| `expect(true).toBe(true)` | Vacuous assertion |
| `// @ts-ignore` in tests | Hiding type errors |
| `skip` or `only` committed | Incomplete test suite |
| Mocking the system under test | Not testing real behavior |
| `setTimeout` in tests | Flaky timing |
| No assertions in test | Test does nothing |
| Catching errors without asserting | Swallowing failures |
| `any` types in test code | Losing type safety |

## Quick Reference Commands

```bash
# Run with coverage
npm test -- --coverage

# Run mutation testing
npx stryker run

# Run stability test (100x)
for i in {1..100}; do npm test || exit 1; done

# Run specific chaos tests
npm test -- --grep="chaos"

# Generate coverage report
npm test -- --coverage --reporter=html

# Check for flaky tests
npm run test:flaky-check
```

## Test File Naming Convention

```
src/
  component.ts
  component.test.ts          # Unit tests
  component.integration.ts   # Integration tests
  component.chaos.ts         # Chaos tests
  component.property.ts      # Property-based tests
tests/
  e2e/
    scenario-name.e2e.ts     # E2E scenarios
  stress/
    component.stress.ts      # Load/stress tests
```

## Remember

> "If you can't break your code with tests, an attacker or edge case will break it in production."

> "Every line of code not covered by a test is a bug waiting to happen."

> "The goal isn't to pass tests. The goal is to make the system impossible to break."

---

**This skill must be loaded FIRST in every session. No exceptions.**
