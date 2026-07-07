# micrograd-my-implimentation-
# micrograd (my implementation)

My implementation of a tiny autograd engine, built while following 
[Andrej Karpathy's Neural Networks: Zero to Hero](link) series — 
specifically the micrograd episode.

Based on the architecture from [karpathy/micrograd](https://github.com/karpathy/micrograd).
Written and debugged independently in Jupyter while working through the video.

## What it does
- Implements a `Value` class with automatic differentiation (backpropagation) via a dynamically built computation graph
- `Neuron`, `Layer`, and `MLP` classes built on top of `Value`
- the demo uses graphviz to vsialize how the gradiants flows through and what the calculation for the loss fucntion looks like

## What I learned
- learned how backprop works by debugging everything myself and running into walls numerous times
- got a much better feel for operator overloading using dunder methods(__add__, __mul__, __exp__ etc)
- got to know exactly how loss functions work and how loss is driven down by taking the gradient into account

## Credit
All conceptual credit to Andrej Karpathy — this is a learning exercise, not original work.
