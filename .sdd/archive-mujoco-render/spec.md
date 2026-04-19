# Spec: Explore Pipeline E2E Fix

**Status**: APPROVED (CEO auto-approved 2026-04-13)

## Problem
After "探索", navigation should work immediately. Currently broken: TARE rushes home after finish, terrain replay missing on cancel, visit_count too strict, no explore→navigate transition.

## Acceptance Criteria
1. After explore, "去厨房" works immediately
2. TARE stops at current position (no rush home)  
3. Terrain replay fires on both finish and cancel
4. Rooms visited once are navigable
5. FAR V-Graph exists after explore

## Changes
1. TARE config: kRushHome=false, kNoExplorationReturnHome=false
2. Bridge: load boundary from room_layout.yaml, not hardcoded
3. Bridge: exploration finish triggers terrain replay before declaring done
4. Explore skill: terrain replay in finally block (fires on cancel too)
5. Navigate: visit_count threshold 3→1
