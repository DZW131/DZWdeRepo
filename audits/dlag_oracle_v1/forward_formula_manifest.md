# Frozen forward and counterfactual formula (DLAG Oracle v1)

The verified UCRF-v1 formula is retained. With frozen Stage3 query/class
weights `W`, the DLAG alpha bank modifies only the real query-logit residual:

```text
q0 = LN(query)
L5 = dot(q0,k5)/sqrt(256)
q5 = update5(q0,L5,v5)
D4 = dot(q5,k4)/sqrt(256)
L4(alpha) = bilinear(L5) + alpha*D4
q4(alpha) = update4(q5,L4(alpha),v4)
D3(alpha) = dot(q4(alpha),k3)/sqrt(256)
L3(alpha) = bilinear(L4(alpha)) + D3(alpha)
C3(alpha)[c,x] = sum_q W[q,c]*sigmoid(L3(alpha)[q,x])
```

There is no tissue-class-map linearization. In particular,
`C4(alpha)` is never constructed as `C5 + alpha*C(D4)`. Alpha `1.0` is
required to reproduce the original Stage3 mixture exactly within floating
point tolerance. All parameters, features, queries before L4, projections,
and class weights remain frozen.
