---
name: react
description: React 19.2 hooks, patterns, and best practices. Use when writing, reviewing, or refactoring React components in this project — new hooks (use, useOptimistic, useActionState, useFormStatus, useEffectEvent), Activity component, React Compiler guidance, and forbidden outdated patterns.
---

# React 19.2 — Hooks, Patterns, Best Practices

## What's New in React 19.2

### New Hooks

| Hook | Purpose | Guide |
|------|---------|-------|
| `use()` | Read promises/context in render | references/new-hooks.md |
| `useOptimistic` | Optimistic UI updates | references/new-hooks.md |
| `useActionState` | Form action state management | references/new-hooks.md |
| `useFormStatus` | Form pending state (child components) | references/new-hooks.md |
| `useEffectEvent` | Non-reactive callbacks in effects | references/new-hooks.md |

## Classic Hooks (React 18+)

### State Hooks

| Hook | Purpose | Guide |
|------|---------|-------|
| `useState` | Local component state | references/use-state.md |

→ For global state, see react-state skill

### Effect Hooks

| Hook | Purpose | Guide |
|------|---------|-------|
| `useEffect` | Side effects after paint | references/use-effect.md |
| `useLayoutEffect` | Sync DOM before paint | references/use-layout-effect.md |

### Ref Hooks

| Hook | Purpose | Guide |
|------|---------|-------|
| `useRef` | DOM access, mutable values | references/use-ref.md |
| `useImperativeHandle` | Customize ref API | references/use-imperative-handle.md |

### Performance Hooks (Rare with Compiler)

| Hook | Purpose | Guide |
|------|---------|-------|
| `useMemo` | Memoize expensive values | references/use-memo.md |
| `useCallback` | Memoize functions | references/use-callback.md |

→ React Compiler handles most memoization automatically

### Other Hooks

| Hook | Purpose | Guide |
|------|---------|-------|
| `useId` | Unique IDs for accessibility | references/use-id.md |
| `useSyncExternalStore` | External store subscription | references/use-sync-external-store.md |

### Custom Hooks

→ See references/custom-hooks-patterns.md for patterns
→ See references/templates/custom-hooks.md for implementations

## Activity Component (19.2)

Hide/show components while preserving state:

```tsx
<Activity mode={isActive ? 'visible' : 'hidden'}>
  <TabContent />
</Activity>
```

→ See references/activity-component.md for patterns

## React Compiler (19.1+)

Automatic memoization — useMemo/useCallback mostly obsolete:

- Build-time optimization
- No more manual memoization in most cases
- 2.5× faster interactions reported

→ See references/react-compiler.md for details

## Quick Reference

### use() Hook

```tsx
// Read promise in render (with Suspense)
const data = use(dataPromise)
// Read context conditionally (unique to use())
if (condition) {
  const theme = use(ThemeContext)
}
```

### useOptimistic

```tsx
const [optimisticValue, setOptimistic] = useOptimistic(actualValue)
// Update UI immediately, server updates later
```

### useActionState

```tsx
const [state, action, isPending] = useActionState(asyncFn, initialState)
```

### useEffectEvent (19.2)

```tsx
const onEvent = useEffectEvent(() => {
  // Always has fresh props/state, doesn't trigger re-run
})
useEffect(() => {
  connection.on('event', onEvent)
}, []) // No need to add onEvent to deps
```

## Breaking Changes from 18

| Change | Migration |
|--------|-----------|
| `ref` is a prop | Remove `forwardRef` wrapper |
| Context is provider | Use `<Context value={}>` directly |
| `useFormStatus` | Import from `react-dom` |

## Best Practices

- **Data fetching:** Use `use()` + Suspense, NOT useEffect
- **Forms:** Use Actions + `useActionState`
- **Optimistic UI:** Use `useOptimistic` for instant feedback
- **Tabs/Modals:** Use `<Activity>` to preserve state
- **Effect events:** Use `useEffectEvent` for non-reactive callbacks
- **Memoization:** Let React Compiler handle it

## Performance

- **Virtualization:** Render only visible items for large lists (100+ items)
- **Lazy loading:** Code split routes and heavy components for smaller bundles
- **Profiling:** Measure render performance with DevTools Profiler
- Note: with React Compiler (19.1+), manual memo/useMemo/useCallback optimizations are mostly obsolete. Profile first to verify if optimization is needed.

## Forbidden (Outdated Patterns)

- ❌ `useEffect` for data fetching → use `use()` + Suspense
- ❌ `forwardRef` → use `ref` as prop
- ❌ `<Context.Provider>` → use `<Context value={}>`
- ❌ Manual `useMemo`/`useCallback` everywhere → let Compiler handle it
- ❌ Conditional rendering for state preservation → use `<Activity>`
