"""

THE PARTIAL DERIVATIVE

The gradient is a vector of partial derivatives. The partial derivative is the derivative of a function with respect to
one of its arguments, while the other arguments are held constant. For example, if we have a function f(x, y), the
partial derivative of f with respect to x is denoted as ∂f/∂x, and is defined as:

∂f/∂x = lim(Δx→0) (f(x + Δx, y) - f(x, y)) / Δx

The partial derivative of f with respect to y is denoted as ∂f/∂y, and is defined as:

∂f/∂y = lim(Δy→0) (f(x, y + Δy) - f(x, y)) / Δy

The gradient of f is denoted as ∇f, and is defined as:

∇f = (∂f/∂x, ∂f/∂y)


THE PARTIAL DERIVATIVE OF MAX()

The partial derivative of the max function is defined as:

∂max(x, y)/∂x = 1 if x > y, otherwise 0

∂max(x, y)/∂y = 1 if y > x, otherwise 0

For ReLu (Single input or variable), the partial derivative is:

∂max(x, 0)/∂x = 1 if x > 0, otherwise 0


THE CHAIN RULE

The chain rule is a formula for calculating the derivative of a composite function. Composite functions are functions
that are made up of two or more functions, where the output of one function becomes the input of another function.
Which is what forward propagation is all about.

For example, if we have a composite function f(g(x)), the chain rule is defined as:

∂f/∂x = (∂f/∂g) * (∂g/∂x)

For example, if we have a composite function f(g(x, y), h(y, z)), the chain rule is defined as:

∂f/∂y = (∂f/∂g) * (∂g/∂y) + (∂f/∂h) * (∂h/∂y)

For example, if we have a composite function f(g(x, y), h(y, z)), the chain rule is defined as:

∂f/∂x = (∂f/∂g) * (∂g/∂x)

The chain rule can be applied to functions with any number of arguments.

"""