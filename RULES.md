# MultiPhaseSSRVE Domain Rules

## Grain Definition
- **Grain** = continuous region grown from ONE seed (nucleation point)
- Each grain has unique **grain_id** derived from its seed index
- Grains of same phase but different seeds are DISTINCT grains

## Periodic Boundary Conditions

### Growth Rules
1. Grain growing through periodic boundary **maintains same ID** on both sides
2. Two grains (different seeds) meeting at boundary remain **separate grains**
3. Grain ID is **never overwritten** by another grain at boundary

### Phase Rules
- Phase IDs start from 1 (no phase 0)
- Empty cells have grain_id = 0
- Multiple grains can belong to same phase
- Grain merging only within same phase (based on compatibility)

### Infinite Grain Detection
A grain is **infinite** if it has **continuous connection** through opposite periodic boundaries:
- Must have **unbroken path** from Top to Bottom, OR from Left to Right
- Simply touching both boundaries is NOT enough - must be connected internally

**Key distinction:** Pixels on opposite boundaries that are NOT connected through the grain body are finite (periodic copies, not continuous grain).

**Examples:**

```
INFINITE (continuous vertical)  INFINITE (continuous horizontal)  INFINITE (fills all)
┌─────────────┐                 ┌─────────────┐                   ┌─────────────┐
│. . 1 1 1 . .│ < top           │. . . . . . .│                   │1 1 1 1 1 1 1│
│. . 1 1 1 . .│ v               │1 1 1 1 1 1 1│ < continuous      │1 1 1 1 1 1 1│
│. . 1 1 1 . .│ connected       │1 1 1 1 1 1 1│   left to right   │1 1 1 1 1 1 1│
│. . 1 1 1 . .│ v               │. . . . . . .│                   │1 1 1 1 1 1 1│
│. . 1 1 1 . .│ < bottom        │. . . . . . .│                   │1 1 1 1 1 1 1│
└─────────────┘                 └─────────────┘                   └─────────────┘
Continuous path top=bottom      Continuous path left=right        Continuous everywhere

INFINITE (diagonal) 
┌─────────────┐
│. . . . . 1 1│ 
│. . . . 1 1 1│ 
│. . . 1 1 1 .│ 
│. 1 1 1 . . .│
│1 1 1 . . . .│ 
└─────────────┘ 

FINITE (disconnected copies)    FINITE (one boundary only)        FINITE (one boundary only
┌─────────────┐                 ┌─────────────┐                   ┌─────────────┐
│1 1 1 . . . 1│                 │. . 1 1 1 . .│                   │. . . . . . .│
│1 1 1 . . . 1│                 │. . . . . . .│                   │1 . . . . . 1│
│. . . . . . .│                 │. . . . . . .│                   │1 . . . . . 1│
│. . . . . . .│                 │. . . . . . .│                   │1 . . . . . 1│
│1 1 1 . . . 1│                 │. . 1 1 1 . .│                   │. . . . . . .│ 
└─────────────┘                 └─────────────┘                   └─────────────┘

dot (.) means any other grain ID or empty cell
```

**Connectivity Check Algorithm:**
To determine if grain is infinite, check for **continuous internal path**:
1. Start from any pixel on boundary (e.g., top edge)
2. Flood fill / BFS through grain pixels (8-connectivity)
3. If the path reaches the opposite boundary (bottom edge), the grain is INFINITE.
4. If no such path exists, the grain is FINITE (just periodic copies).

**Important:** Simply having pixels on both boundaries is NOT sufficient. The grain must have connected path through its body.

**Consequences:**
- Infinite grain has infinite physical size (due to periodicity)
- `grain_size_deviation` assigns **1.0 (100% deviation)** for infinite grains

