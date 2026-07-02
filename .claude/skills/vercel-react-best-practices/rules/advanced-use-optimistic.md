---
title: useOptimistic for Instant UI Updates
impact: MEDIUM
impactDescription: removes hand-rolled optimistic state + rollback logic
tags: advanced, hooks, useOptimistic, transitions, forms, react-19
---

## useOptimistic for Instant UI Updates

`useOptimistic(state, updateFn)` renders an optimistic value immediately, then automatically reverts to the real `state` once the underlying update settles (or fails). It replaces manual "shadow state + reset on error" bookkeeping.

**Incorrect (manual optimistic state and rollback):**

```tsx
function LikeButton({ liked, onToggle }: { liked: boolean; onToggle: () => Promise<void> }) {
  const [optimisticLiked, setOptimisticLiked] = useState(liked)

  const handleClick = async () => {
    setOptimisticLiked(!optimisticLiked)
    try {
      await onToggle()
    } catch {
      setOptimisticLiked(liked) // manual rollback
    }
  }

  return <button onClick={handleClick}>{optimisticLiked ? 'Liked' : 'Like'}</button>
}
```

**Correct (useOptimistic handles rollback):**

```tsx
function LikeButton({ liked, onToggle }: { liked: boolean; onToggle: () => Promise<void> }) {
  const [optimisticLiked, setOptimisticLiked] = useOptimistic(liked)

  const handleClick = () => {
    startTransition(async () => {
      setOptimisticLiked(!optimisticLiked)
      await onToggle() // on throw, optimisticLiked reverts to `liked` automatically
    })
  }

  return <button onClick={handleClick}>{optimisticLiked ? 'Liked' : 'Like'}</button>
}
```

**Must be called inside a transition:** the update function (`setOptimisticLiked` above) has to run inside `startTransition` (or a `useActionState`/form action, which is already a transition). Calling it outside a transition triggers a dev warning and the optimistic value won't revert correctly.

Reference: [https://react.dev/reference/react/useOptimistic](https://react.dev/reference/react/useOptimistic)
