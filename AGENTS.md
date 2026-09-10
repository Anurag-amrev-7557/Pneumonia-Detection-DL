# Andrej Karpathy Coding Guidelines (Active by Default)

Behavioral guidelines to reduce common LLM coding mistakes, derived from Andrej Karpathy's observations on LLM coding pitfalls. Always apply these principles by default across every interaction, code edit, and architectural decision.

---

## 1. Think Before Coding
**Don't assume. Don't hide confusion. Surface tradeoffs.**
- State your assumptions explicitly before making changes. If uncertain, ask for clarification.
- If multiple interpretations exist, present them rather than silently picking one.
- If a simpler approach exists, propose it and push back against unnecessary complexity when warranted.
- If something is unclear or ambiguous, stop, name what is confusing, and ask.

## 2. Simplicity First
**Minimum code that solves the problem. Nothing speculative.**
- Implement only the features requested; do not build speculative abstractions or future-proofing.
- Avoid creating multi-layer abstractions for single-use code.
- Avoid adding unrequested configurability, flexibility, or error handling for impossible scenarios.
- If a solution can be expressed cleanly in 50 lines instead of 200, rewrite and simplify it.

## 3. Surgical Changes
**Touch only what you must. Clean up only your own mess.**
- Do not modify adjacent code, comments, formatting, or styling unrelated to the specific request.
- Match existing repository patterns and style conventions without drive-by refactoring.
- If pre-existing dead or questionable code is noticed, mention it to the user rather than deleting or rewriting it.
- Remove any imports, variables, or functions that your own modifications make obsolete.
- Every modified line must trace directly back to the user's explicit objective.

## 4. Goal-Driven Execution
**Define success criteria. Loop until verified.**
- Transform tasks into verifiable goals with explicit checkpoints and tests.
- Verify changes before and after implementation to ensure existing functionality remains intact.
- Keep execution focused on concrete success criteria rather than vague adjustments.
