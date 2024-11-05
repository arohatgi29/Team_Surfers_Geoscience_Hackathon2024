# Team_Surfers_Geoscience_Hackathon2024

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/2024-Geoscience-Hackathon-Logo.jpg" >


##### This repository has all the resources and results used in the JSG Open Source Hackathon 2024

We reproduced the paper Raissi, M., Perdikaris, P. and Karniadakis, G.E., 2019. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational physics, 378, pp.686-707

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/Picture1.png" >

## Brief workflow of what we reproduced and produced

```mermaid
graph TD;
    A(Understanding PINNs)-->B(Reproduce Burger's Equation using PINN);
    B-->C(Parameters Sensitivity Analysis);
    C-->D(Used PINNs to solve 1D wave equation)
    D-->E(Used PINNs to solve 2D wave equation)
```

## 1+1D viscous Burgers' equation:


The Burgers' equation is given by:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0
$$

with initial and boundary conditions:

$$
u(x, 0) = -\sin(\pi x), \quad u(-1, t) = u(1, t) = 0
$$

The neural network approximation for \( u(x, t) \) is given by:

$$
NN(x, t; \theta) \approx u(x, t)
$$

### Boundary Loss

The boundary loss is defined as:

$$
L_b(\theta) = \frac{\lambda_1}{N_{b1}} \sum_{j=1}^{N_{b1}} \left( NN(x_j, 0; \theta) + \sin(\pi x_j) \right)^2 + \frac{\lambda_2}{N_{b2}} \sum_{k=1}^{N_{b2}} \left( NN(-1, t_k; \theta) - 0 \right)^2 + \frac{\lambda_3}{N_{b3}} \sum_{l=1}^{N_{b3}} \left( NN(1, t_l; \theta) - 0 \right)^2
$$


### Physics Loss

The physics loss is given by:

$$
L_p(\theta) = \frac{1}{N_p} \sum_{i=1}^{N_p} \left( \left( \frac{\partial NN}{\partial t} + NN \frac{\partial NN}{\partial x} - \nu \frac{\partial^2 NN}{\partial x^2} \right)(x_i, t_i; \theta) \right)^2
$$

## CPU version of reproducibility


```Shell
We used He normal initializer to initialize the weights to the input layer,
Activation function = tanh, L-BFGS-B = optimizer
No of iterations = 5000
```

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/cpu.png" >


## GPU version of reproducibility

GPU version took much less time woith same parameters


<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/GPU.png" >


<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/GPU1.png" >

## Shallow Neural Network

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/shalloe.png" >

## Changing number of layers with 50 Neurons

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/jl_ubmVMNIzsx.gif" >


## 1D Wave Equation using PINNs

The 1D wave equation is given by:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

where:
- \( u(x, t) \) is the displacement at position \( x \) and time \( t \),
- \( c \) is the speed of the wave.

### Initial and Boundary Conditions (Dirichlet's)

The initial and boundary conditions can be specified as:

Initial conditions:

$$
u(x, 0) = f(x), \quad \frac{\partial u}{\partial t}(x, 0) = g(x)
$$

Boundary conditions:

$$
u(0, t) = u(L, t) = 0
$$

where \( f(x) \) is the initial displacement and \( g(x) \) is the initial velocity.

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/1D_output%202.gif" >

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/dirchlet.png" >

## Neumann Boundary Condition for 1D Wave Equation

The Neumann boundary condition specifies the derivative of the function at the boundary rather than the function value itself. For the 1D wave equation:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

The Neumann boundary conditions are given by:

$$
\frac{\partial u}{\partial x}(0, t) = 0, \quad \frac{\partial u}{\partial x}(L, t) = 0
$$

where:
- \( \frac{\partial u}{\partial x} \) is the spatial derivative of \( u \) at the boundaries \( x = 0 \) and \( x = L \),
- \( L \) is the length of the domain.

### Initial Condition

The initial condition can be specified as:

$$
u(x, 0) = f(x), \quad \frac{\partial u}{\partial t}(x, 0) = g(x)
$$

where \( f(x) \) is the initial displacement, and \( g(x) \) is the initial velocity.


<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/1D_output_neumann.gif" >

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/1D_wave_cmap_Neumann.png" >


## 2D Wave Equation using PINNs
## 2D Wave Equation

The 2D wave equation is given by:

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

where:
- \( u(x, y, t) \) is the displacement at position \( (x, y) \) and time \( t \),
- \( c \) is the speed of the wave.

### Initial and Boundary Conditions

The initial and boundary conditions can be specified as:

Initial conditions:

$$
u(x, y, 0) = f(x, y), \quad \frac{\partial u}{\partial t}(x, y, 0) = g(x, y)
$$

Boundary conditions:

$$
u(x, y, t) = 0 \quad \text{on the boundary of the domain}.
$$

where \( f(x, y) \) is the initial displacement and \( g(x, y) \) is the initial velocity.

<img src="https://github.com/arohatgi29/Team_Surfers_Geoscience_Hackathon2024/blob/main/Images/animation_circular_t500.gif" >




















