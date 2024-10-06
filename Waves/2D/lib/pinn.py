import tensorflow as tf
from .layer import GradientLayer

class PINN:
    """
    Build a physics informed neural network (PINN) model for the wave equation.

    Attributes:
        network: keras network model with input (t, x) and output u(t, x).
        c: wave velocity.
        grads: gradient layer.
    """

    def __init__(self, network, c=1):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            c: wave velocity. Default is 1.
        """

        self.network = network
        self.c = c
        self.grads = GradientLayer(self.network)

    def build(self):
 

        # equation input: (t, x, y)
        txy_eqn = tf.keras.layers.Input(shape=(3,))
        # initial condition input: (t=0, x,y)
        txy_ini = tf.keras.layers.Input(shape=(3,))
        # boundary condition input: (t, x, y=-1) or (t, x, y=+1) for y-boundaries, same for x
        txy_bnd = tf.keras.layers.Input(shape=(3,))

        # compute gradients (u, du_dt, du_dx, du_dy, d2u_dt2, d2u_dx2, d2u_dy2)
        u_eqn, _, _, _, d2u_dt2, d2u_dx2, d2u_dy2 = self.grads(txy_eqn)

        # equation output should satisfy the 2D wave equation: u_tt = c^2 (u_xx + u_yy)
        u_eqn = d2u_dt2 - self.c * self.c * (d2u_dx2 + d2u_dy2)

        # initial condition output
        u_ini, du_dt_ini, _, _, _, _, _ = self.grads(txy_ini)

        # boundary condition output
        u_bnd = self.network(txy_bnd)  # Dirichlet boundary condition
        # _, _, _, _, _, _, u_bnd = self.grads(txy_bnd)  # Neumann boundary condition

        # build the PINN model for the 2D wave equation
        return tf.keras.models.Model(
            inputs=[txy_eqn, txy_ini, txy_bnd],
            outputs=[u_eqn, u_ini, du_dt_ini, u_bnd]
        )
