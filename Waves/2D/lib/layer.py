import tensorflow as tf

class GradientLayer(tf.keras.layers.Layer):
    """
    Custom layer to compute 1st and 2nd derivatives for the 2D wave equation.

    Attributes:
        model: keras network model.
    """

    def __init__(self, model, **kwargs):
        """
        Args:
            model: keras network model.
        """
        self.model = model
        super().__init__(**kwargs)

    def call(self, txy):
        """
        Computing 1st and 2nd derivatives for the 2D wave equation.

        Args:
            txy: input variables (t, x, y).

        Returns:
            u: network output.
            du_dt: 1st derivative of t.
            du_dx: 1st derivative of x.
            du_dy: 1st derivative of y.
            d2u_dt2: 2nd derivative of t.
            d2u_dx2: 2nd derivative of x.
            d2u_dy2: 2nd derivative of y.
        """

        with tf.GradientTape() as g:
            g.watch(txy)
            with tf.GradientTape() as gg:
                gg.watch(txy)
                u = self.model(txy)
            du_dtx = gg.batch_jacobian(u, txy)
            du_dt = du_dtx[..., 0]   # Derivative with respect to t
            du_dx = du_dtx[..., 1]   # Derivative with respect to x
            du_dy = du_dtx[..., 2]   # Derivative with respect to y

        d2u_dtx2 = g.batch_jacobian(du_dtx, txy)
        d2u_dt2 = d2u_dtx2[..., 0, 0]  # 2nd derivative with respect to t
        d2u_dx2 = d2u_dtx2[..., 1, 1]  # 2nd derivative with respect to x
        d2u_dy2 = d2u_dtx2[..., 2, 2]  # 2nd derivative with respect to y

        return u, du_dt, du_dx, du_dy, d2u_dt2, d2u_dx2, d2u_dy2
