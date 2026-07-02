---
title: useFormStatus for Pending State in Child Components
impact: LOW-MEDIUM
impactDescription: avoids prop-drilling pending state through form children
tags: advanced, hooks, useFormStatus, forms, react-19
---

## useFormStatus for Pending State in Child Components

`useFormStatus()` (from `react-dom`) reads the submission status of the nearest parent `<form>` without the form having to pass `pending` down as a prop. It only works in a component **rendered inside** the form — calling it in the same component that renders the `<form>` element itself always returns `pending: false`.

**Incorrect (pending state manually threaded through props):**

```tsx
function SearchForm({ action }: { action: (formData: FormData) => void }) {
  const [isPending, setIsPending] = useState(false)
  return (
    <form action={action}>
      <SubmitButton pending={isPending} />
    </form>
  )
}

function SubmitButton({ pending }: { pending: boolean }) {
  return <button disabled={pending}>{pending ? 'Searching…' : 'Search'}</button>
}
```

**Correct (child reads status directly):**

```tsx
function SearchForm({ action }: { action: (formData: FormData) => void }) {
  return (
    <form action={action}>
      <SubmitButton />
    </form>
  )
}

function SubmitButton() {
  const { pending } = useFormStatus()
  return <button disabled={pending}>{pending ? 'Searching…' : 'Search'}</button>
}
```

**Common mistake — calling it in the form's own component:**

```tsx
// Incorrect: this component renders the <form>, so useFormStatus here
// reports the status of an ancestor form (or nothing), never this one.
function SearchForm({ action }: { action: (formData: FormData) => void }) {
  const { pending } = useFormStatus() // always false
  return <form action={action}>...</form>
}
```

Reference: [https://react.dev/reference/react-dom/hooks/useFormStatus](https://react.dev/reference/react-dom/hooks/useFormStatus)
