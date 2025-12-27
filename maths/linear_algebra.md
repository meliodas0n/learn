# Linear Algebra and Its Applications

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

## Existence and Uniqueness Questions
Answers to the following two questions will determine the nature of the solution set for a linear system as to determine which possibility is true for a particular system,
(possibility -- no solution or one solution or infinite solution),

1. Is the system consistent; that is, does at least one solution *exist*?
2. If a solution exists, is it the *only one*; that is, is the solution *unique*?

### Numerical Note:
In real-world problems, system of linear equations are solved by a computer, For a square coefficient matrix, computer programs nearly always use the elimination algorithm.
The vast majority of linear algebra problems in business and industry are solved with programs that use *floating point arithmetic*.
Numbers are represented as decimals +-.d1...dp x 10^r, where *r* is an integer and the number *p* of digits to the right of the decimal point is usually between 8 and 16.
Arithmetic with such numbers typically is inexact, because the results must be rounded(or truncated) to the numbner of digits stored.
"Roundoff error" is also introduced when a number such as 1/3 is entered into the computer, since its decimal representation must be approximated by a finite number of digits.
*`Fortunately, inaccuracies in floating point arithmetic seldom cause problems.`*

## Row Reduction and Echelon Forms
A *nonzero* row or column in a matrix means a row or column that contains at least one nonzero entry.
A `leading entry` of a row refers to the leftmost nonzero entry(in a nonzero row).

### Echelon Form
A rectangular matrix is in `echelon form` (or `row echelong form`) if it has the following three properties:
  1. All nonzero rows are above any rows of all zeros.
  2. Each leading entry of a row is in a column to the right of the leading entry of the row above it.
  3. All entries in a column below a leading entry are zeros.
---
If a matrix in echelong form satisfies the following additional conditions, then it is in `reduced echelon form` (or `reduced row echelon form`):
---
  4. The leading entry in each nonzero row is 1.
  5. Each leading 1 is the only nonzero entry in its column.