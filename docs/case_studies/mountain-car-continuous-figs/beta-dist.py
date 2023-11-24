import jax.scipy.stats as jstats
import matplotlib.pyplot as plt
import numpy as np
from jax import grad
from scipy import stats


def main():

    # various beta distribution shapes
    x_vals = np.linspace(0.0, 1.0, 100)
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=0.5, b=0.5), label=f'a={0.5}, b={0.5}')
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=1.0, b=1.0), label=f'a={1.0}, b={1.0}')
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=10.0, b=10.0), label=f'a={10.0}, b={10.0}')
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=5.0, b=20.0), label=f'a={5.0}, b={20.0}')
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=20.0, b=5.0), label=f'a={20.0}, b={5.0}')
    plt.ylabel('Beta PDF')
    plt.xlabel('Action')
    plt.legend()
    plt.show()

    # action distribution
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=2.0, b=2.0), label=f'a={2.0}, b={2.0}')
    plt.axvline(0.1, color='red')
    plt.legend()
    plt.xlabel('Action')
    plt.ylabel('Beta PDF')
    plt.show()

    # beta distribution at x=0.1 for varying values of a
    a_vals = np.linspace(0.1, 2.0, 100)
    pdf_at_x = [stats.beta.pdf(x=0.1, a=a, b=2.0) for a in a_vals]
    plt.plot(a_vals, pdf_at_x)
    plt.xlabel('a')
    plt.ylabel('Beta PDF @ x=0.1, b=2.0')
    plt.xlim((0.1, 2.0))
    plt.show()

    # modified beta distribution to increase the density at x=0.1
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=2.0, b=2.0), label=f'a={2.0}, b={2.0}')
    plt.plot(x_vals, stats.beta.pdf(x_vals, a=0.6, b=2.0), label=f'a={0.6}, b={2.0}')
    plt.axvline(0.1, color='red')
    plt.legend()
    plt.xlabel('Action')
    plt.ylabel('Beta PDF')
    plt.show()

    # 1. Define the function for which we want a gradient.
    def jax_beta_pdf(
            x: float,
            a: float,
            b: float
    ):
        return jstats.beta.pdf(x=x, a=a, b=b, loc=0.0, scale=1.0)

    # 2. Ask JAX for the gradient with respect to the second argument (shape parameter a).
    jax_beta_pdf_grad = grad(jax_beta_pdf, argnums=1)

    # 3. Calculate the gradient that we want.
    print(f'{jax_beta_pdf_grad(0.1, 2.0, b=2.0)}')


if __name__ == '__main__':
    main()
