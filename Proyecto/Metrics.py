import numpy as np

class Metrics:

    @staticmethod
    def IAE(t, e):

        return np.trapz(
            np.abs(e),
            t
        )

    @staticmethod
    def ISE(t, e):

        return np.trapz(
            e**2,
            t
        )

    @staticmethod
    def ITAE(t, e):

        return np.trapz(
            t*np.abs(e),
            t
        )

    @staticmethod
    def RMSE(e):

        return np.sqrt(
            np.mean(e**2)
        )