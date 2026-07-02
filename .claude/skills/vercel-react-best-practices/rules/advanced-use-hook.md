---
title: Read Promises and Context with use()
impact: MEDIUM
impactDescription: replaces useEffect data-fetching, allows conditional context reads
tags: advanced, hooks, use, suspense, context, react-19
---

## Read Promises and Context with use()

`use()` reads the value of a promise or context during render. Unlike other hooks, it can be called conditionally or after early returns, and it integrates directly with Suspense instead of needing a separate loading state managed in `useEffect`/`useState`.

**Incorrect (manual fetch + loading state in an effect):**

```tsx
function Comments({ commentsPromise }: { commentsPromise: Promise<Comment[]> }) {
  const [comments, setComments] = useState<Comment[] | null>(null)

  useEffect(() => {
    commentsPromise.then(setComments)
  }, [commentsPromise])

  if (!comments) return <Spinner />
  return <ul>{comments.map(c => <li key={c.id}>{c.text}</li>)}</ul>
}
```

**Correct (use() + Suspense, no extra state or effect):**

```tsx
function Comments({ commentsPromise }: { commentsPromise: Promise<Comment[]> }) {
  const comments = use(commentsPromise)
  return <ul>{comments.map(c => <li key={c.id}>{c.text}</li>)}</ul>
}

function Page({ commentsPromise }: { commentsPromise: Promise<Comment[]> }) {
  return (
    <Suspense fallback={<Spinner />}>
      <Comments commentsPromise={commentsPromise} />
    </Suspense>
  )
}
```

The promise should be created above the `Suspense` boundary (e.g. in a Server Component or a parent that doesn't re-create it on every render) so it isn't recreated each render — see [Strategic Suspense Boundaries](./async-suspense-boundaries.md) for the pattern of sharing one promise across multiple `use()` calls.

**Conditional reads (not possible with useContext):**

```tsx
function Banner({ show }: { show: boolean }) {
  if (!show) return null
  // useContext cannot be called after a conditional return; use() can.
  const theme = use(ThemeContext)
  return <div className={theme}>Banner</div>
}
```

Reference: [https://react.dev/reference/react/use](https://react.dev/reference/react/use)
