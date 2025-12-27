A system of `linear equations` (or a `linear system`) is a collection of one or more linear equations involving the same variables - say, x1...xn.

A `solution` of the system is a list(s1, s2,....sn) of numbers that makes each equation a true statement when the values s1...sn are substituted for x1...xn respectively

The set of all possible solutions is called the `solution set` of the linear system. Two linear systems are called `equivalent` if they have same solution set.
That is, each solution of the first sytem is a solution of the second system, and each solution of the second system is a solution of the first.

A system of linear equations has
1. no solution, or
2. exactly one solution, or
3. infinitely many solutions.

A system of linear equations is said to be `consistent` if it has either one solution or infinitely many solutions; a system is `inconsistent` if it has no solution.

The essential information of a linear system can be recorded compactly in a rectangular array called a matrix, with the coefficients of each variable aligned in columns, the matrix is called `coefficient matrix` (or `matris of coefficients`) of the system, and is called `augmented matrix` of the system.

An augmented matrix of a system consists of coefficient matrix with an added column containing the constants from the right sides of the equations.

#### Elementary Row Operations
1. (Replacement) Replace one  row by the sum of iteself and a multiple of another row.
2. (Interchange) Interchange two rows.
3. (Scaling) Multiply all entries in a row by a nonzero constant.

Row operations can be applied to any matrix, not merely to one that arises as the augmented matrix of a linear system. Tow matrices are called `row equivalent` if there is a sequence of elementary row operations that transforms one matrix into the other.
If it important to note that row operations are *reversible*. If two rows are interchanged, they can be returned to their original positions by another interchange.

##### If the augmented matrices of two linear systems are now equivalent, then the two systems have the same solution set.