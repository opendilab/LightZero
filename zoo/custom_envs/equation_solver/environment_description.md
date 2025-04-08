# Introduction

Here we define the Markov Decision Process for our environment.

# Markov Decision Process

![Expression Trees](tree.png)

**States**: Equations like $ax+b = 0$ or $cx+d = -x/b$. We vectorize these using a given equation's expression tree, a representation in which the operations and terms within the equation are represented by nodes and edges. Figure 1 shows the expression trees for $ax+b$ and $ax^2 + bx + c$ as examples. To extract a vector from a given tree, we use preorder traversal, which produces a list of operations and terms of variable length that we pad to a maximum length $L=50$ (chosen heuristically). A typical vector might look like $[b, \text{plus}, \text{times}, a, x]$. Finally, we use a feature dictionary to map operations and symbols to integers $\{\text{add}:1, \text{subtract}:2, \text{multiply}:3, \ldots, x:5, a:6, \ldots\}$. For example, the vector representation of $x+3$ would be $[3, 5, 0, 0, 0, 0, 0]$.

At any point in solving the environment, there will be a left-hand side (*lhs*) and a right-hand side (*rhs*) (e.g., $ax = -b$). The state for our environment will be a concatenation of these: $state = (f(\text{lhs}), f(\text{rhs}))$, where $f$ denotes the featurization mentioned above. When we employ a GNN architecture, we also provide the adjacency matrix (derived from the tree) to the network.

**Actions**: Represented as $(operation, term)$ pairs, such as $(\text{sub}, b)$ or $(\text{div}, a)$. How do we choose which operations and terms to include? For the operations, a simple choice is the arithmetic operations $O = (\text{add}, \text{sub}, \text{mul}, \text{div})$. For terms, one could choose the variables that occur in the equation. If this equation were $ax+b=0$, say, the term set would be $T = (x, a, b)$. The action set in turn would be $A = O \times T = \{(\text{add}, x), (\text{sub}, x), \dots, (\text{div}, b)\}$.

This action formulation is sufficient to solve very simple equations like $ax+b=0$ (the solution actions are $(\text{sub}, b), (\text{div}, a)$, which are in $T$), but not big enough to solve more complex equations like $(ax+b)/(cx+d)+e=0$. You can see that after subtracting $e$ from both sides of this rational equation, the next step is to multiply by the divisor $(cx+d)$—but this $(cx+d)$ is not part of the term set that just contains the variables $T = (a, b, c, d, e)$.

To overcome this limitation, we expand the term set by including all *subexpressions* that appear in the equation. The sub-expressions for $ax+b$ and $(ax+b)/(cx+d)+e=0$ are given below:  
- $ax+b=0 \Rightarrow \{a, x, ax, b\}$  
- $(ax+b)/(cx+d)+e=0 \Rightarrow \{a, x, ax, b, c, d, cx, cx+d, ax+b\}$

This term set is expressive enough to solve the rational equation and all equations we consider in this paper. Importantly, it is also *dynamic*; the list of subexpressions is derived from the equation/state and thus has variable length.

Looping back to the set of operations, we also add some of SymPy's internal functions:  
- $\text{expand}: (x + 1)^2 \rightarrow x^2 + 2x + 1$  
- $\text{factor}: x^2 - 1 \rightarrow (x - 1)(x + 1)$  
- $\text{collect}: a \cdot x + b \cdot x \rightarrow (a + b) \cdot x$  
- $\text{square}: x \rightarrow x^2$  
- $\text{square root}: x^2 \rightarrow x$

Our reasoning here is that these higher-level operations provide structured transformations that simplify equation solving and reduce the number of required steps. Including these operations allows the agent to generalize beyond simple linear equations and efficiently handle more complex algebraic structures, such as rational and polynomial equations, without requiring a massive number of atomic arithmetic operations.

Most of these additional operations do not require an input, so we sub $\text{None}$ into the $(operation, term)$ tuple. The exception is $\text{collect}$, which requires a term. We allow collection only on the variable $x$. We also add an action $(\text{multiply}, -1)$ so that equations of the form $-ax=b$ can be solved. The agent can recreate $(\text{divide}, -a)$ by $(\text{multiply}, -1)$ followed by $(\text{divide}, a)$.

Our final action set is thus:  
- $O_1 = \{\text{add}, \text{subtract}, \text{multiply}, \text{divide}\}$  
- $O_2 = \{\text{expand}, \text{factor}, \text{square}, \text{square root}\}$  
- $\text{Terms} \; T = \{\text{sub-expressions of } \text{lhs}\} \cup \{\text{sub-expressions of } \text{rhs}\}$  
- $\text{Actions} \; A = (O_1 \times T) \cup (O_2 \times \{\text{None}\}) \cup \{(\text{collect}, x)\} \cup \{(\text{multiply}, -1)\}$

The action set $A$ is indexed serially, mapping an integer $i$ selected by the agent to an $(operation, term)$ pair. As $A$’s size varies with the equation state, we cap it at $|A|=50$ and mask illegal actions (e.g., division by zero).

**Rewards**: We encourage the RL agent to simplify the equation by assigning a complexity score $C$ to each equation, defined as the total number of nodes plus edges in the expression tree. For the examples in Figure 1, this would be:  
- $C(ax+b) = N_{\text{nodes}} + N_{\text{edges}} = 5 + 4 = 9$  
- $C(ax^2 + bx + c) = N_{\text{nodes}} + N_{\text{edges}} = 10 + 9 = 19$

The reward is defined as the reduction in the complexity of the equation:  
- $R(\text{action}) = C(\text{equation}) - C(\text{equation after action})$

The intuition here is to encourage the agent to take actions that simplify equations.

**State Transition Function**: We set up our RL problem using Python and use the SymPy package to represent the equation state $ax+b$. It handles the algebraic manipulation required to transition between states. At each step, we keep track of an $\text{lhs} = (ax+b)$ and $\text{rhs} = 0$. We apply the action to both the lhs and rhs. For example, $(\text{subtract}, b)$ results in $(\text{lhs}, \text{rhs}) = (ax, -b)$, and then $(\text{divide}, a)$ results in $(\text{lhs}, \text{rhs}) = (x, -b/a)$. The terminal condition for the environment is when $\text{lhs}=x$ and the $\text{rhs} = -b/a$ when substituted into the original equation $ax+b$ simplifies to $0$. There were some illegal actions, such as division by zero, we needed to prohibit (see Appendix).

**Limitations**: Importantly, this MDP formulation only works on equations that are "closed," in the sense that solving them requires manipulating the terms already present in the equation/in the sub-expression list. By contrast, solving "open" equations requires adding new, out-of-equation terms or clever substitutions. A classic example is the quadratic equation $ax^2 + bx + c = 0$. This can be solved two ways. First, you subtract $c$ from both sides to generate $(\text{lhs}, \text{rhs}) = (ax^2 + bx, -c)$. The next move is to complete the square by adding $(b/2a)^2$ to each side—an instance of "generative" reasoning, since the term $(b/2a)^2$ is *not* in the term set/list of sub-expressions we have defined.

The second way to solve the quadratic is via a change of variables $y \rightarrow x - b/(2a)$, which, after some algebraic simplifications, reduces the equation to $a y^2 + c - b^2/(4a)$, which is now "closed" and can be solved with the action sequence $(\text{sub}, c - b^2/(4a)), (\text{div}, a), (\text{sqrt}, \text{None})$.

Equations that require these more exotic actions (change of variable and completing the square by adding out-of-equation terms) are beyond the scope of the current work.