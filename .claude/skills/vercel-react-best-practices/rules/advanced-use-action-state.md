---
title: useActionState for Form Action State
impact: MEDIUM
impactDescription: removes manual pending/error state and works without client JS
tags: advanced, hooks, useActionState, forms, actions, react-19
---

## useActionState for Form Action State

`useActionState(action, initialState)` returns `[state, formAction, isPending]`. Pass `formAction` directly as a `<form action={...}>`. It replaces manually wiring `useState` + `useTransition` around a submit handler, and — because it's a real form action — it still works if client JS hasn't loaded yet (progressive enhancement in frameworks that support it).

**Incorrect (manual pending/error state):**

```tsx
function AddToCartForm({ productId }: { productId: string }) {
  const [error, setError] = useState<string | null>(null)
  const [isPending, startTransition] = useTransition()

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    startTransition(async () => {
      const result = await addToCart(productId)
      if (result.error) setError(result.error)
    })
  }

  return (
    <form onSubmit={handleSubmit}>
      <button disabled={isPending}>Add to cart</button>
      {error && <p>{error}</p>}
    </form>
  )
}
```

**Correct (useActionState owns pending + result state):**

```tsx
function AddToCartForm({ productId }: { productId: string }) {
  const [state, formAction, isPending] = useActionState(
    async (_prevState: { error?: string } | null, formData: FormData) => {
      return addToCart(productId)
    },
    null
  )

  return (
    <form action={formAction}>
      <button disabled={isPending}>Add to cart</button>
      {state?.error && <p>{state.error}</p>}
    </form>
  )
}
```

The action receives the previous state as its first argument, so results (validation errors, success messages) can be threaded back into the form without any extra `useState`.

Reference: [https://react.dev/reference/react/useActionState](https://react.dev/reference/react/useActionState)
