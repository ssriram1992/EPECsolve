$TITLE "SOLVE AN LCP FROM A CSV FILE"

Sets
indices /i0*i100/
;
alias(indices, indices2);

Variables
Vars (indices)
;

Equations
Eqns(indices)
;

* The values of M and q has to be read in from a csv file, somehow
Parameters
M(indices, indices) "The matrix M in Mx+q"
q(indices) 			"The vector q in Mx+q"
;

Equations
Eqns(indices)
;

Eqns(indices).. sum(indices2, M(indices, indices2)*x(indices2)) + q(indices) =g= 0;

Model LCP/
Eqns.Vars
;

Solve LCP using MCP;

* Should save the solution in a csv file.

