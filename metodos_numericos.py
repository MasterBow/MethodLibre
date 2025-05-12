# metodos_numericos.py
# (Biblioteca de métodos numéricos)

import numpy as np

# --- MÓDULO: RAÍCES DE ECUACIONES ---

def bisection_method(func, a, b, tol=1e-7, max_iter=100):
    """
    Encuentra una raíz de la función 'func' en el intervalo [a, b]
    usando el método de bisección.

    Args:
        func (callable): La función para la cual encontrar la raíz. Ejemplo: lambda x: x**2 - 2.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        tol (float): Tolerancia para la convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        tuple: (float or None, str)
            - La raíz aproximada si se encuentra.
            - None si no converge o hay error de entrada.
            - Un mensaje indicando el estado de la convergencia o error.
    """
    if func(a) * func(b) >= 0:
        return None, "La función debe tener signos opuestos en los extremos a y b."

    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        c = (a + b) / 2.0
        if func(c) == 0 or (b - a) / 2.0 < tol:
            return c, f"Bisección convergió en {iterations_done} iteraciones."
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2.0, f"Bisección: Máx. iteraciones ({max_iter}) alcanzado. Mejor aprox."

def newton_raphson_method(func, func_derivative, x0, tol=1e-7, max_iter=100):
    """
    Encuentra una raíz de la función 'func' usando el método de Newton-Raphson.

    Args:
        func (callable): La función para la cual encontrar la raíz.
        func_derivative (callable): La derivada de la función 'func'.
        x0 (float): Estimación inicial de la raíz.
        tol (float): Tolerancia para la convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        tuple: (float or None, str)
            - La raíz aproximada si se encuentra.
            - None si no converge o hay error.
            - Un mensaje indicando el estado de la convergencia o error.
    """
    x_n = x0
    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        fx_n = func(x_n)
        dfx_n = func_derivative(x_n)

        if abs(dfx_n) < 1e-12: # Evitar división por cero
            return None, "Newton-Raphson: Derivada cercana a cero. No se puede continuar."

        x_n_plus_1 = x_n - fx_n / dfx_n

        if abs(x_n_plus_1 - x_n) < tol:
            return x_n_plus_1, f"Newton-Raphson convergió en {iterations_done} iteraciones."

        x_n = x_n_plus_1

    return x_n, f"Newton-Raphson: Máx. iteraciones ({max_iter}) alcanzado. Mejor aprox."

# --- MÓDULO: INTEGRACIÓN NUMÉRICA ---

def trapezoidal_rule(func, a, b, n=100):
    """
    Calcula la integral definida de 'func' de 'a' a 'b' usando la regla del trapecio.

    Args:
        func (callable): La función a integrar.
        a (float): Límite inferior de integración.
        b (float): Límite superior de integración.
        n (int): Número de subintervalos (trapecios).

    Returns:
        tuple: (float or None, str)
            - El valor aproximado de la integral.
            - None si hay error de entrada.
            - Un mensaje de error si 'n' no es válido.
    """
    if n <= 0:
        return None, "El número de subintervalos 'n' debe ser positivo."
    h = (b - a) / n
    integral = 0.5 * (func(a) + func(b))
    for i in range(1, n):
        integral += func(a + i * h)
    integral *= h
    return integral, f"Calculado con {n} subintervalos."

def simpsons_1_3_rule(func, a, b, n=100):
    """
    Calcula la integral definida de 'func' de 'a' a 'b' usando la regla de Simpson 1/3.
    Nota: 'n' debe ser un número par.

    Args:
        func (callable): La función a integrar.
        a (float): Límite inferior de integración.
        b (float): Límite superior de integración.
        n (int): Número de subintervalos (debe ser par).

    Returns:
        tuple: (float or None, str)
            - El valor aproximado de la integral.
            - None si hay error de entrada.
            - Un mensaje de error si 'n' no es válido.
    """
    if n <= 0:
        return None, "El número de subintervalos 'n' debe ser positivo."
    if n % 2 != 0:
        return None, "El número de subintervalos 'n' debe ser par para la regla de Simpson 1/3."

    h = (b - a) / n
    integral = func(a) + func(b)

    for i in range(1, n):
        x_i = a + i * h
        if i % 2 == 0: # Puntos pares
            integral += 2 * func(x_i)
        else: # Puntos impares
            integral += 4 * func(x_i)

    integral *= (h / 3.0)
    return integral, f"Calculado con {n} subintervalos (pares)."

# --- MÓDULO: ECUACIONES DIFERENCIALES ORDINARIAS (EDOs) ---

def euler_method(func_deriv, x0, y0, h, x_end):
    """
    Resuelve la EDO dy/dx = func_deriv(x, y) usando el método de Euler.

    Args:
        func_deriv (callable): La función que define la derivada dy/dx, f(x, y).
        x0 (float): Condición inicial para x.
        y0 (float): Condición inicial para y.
        h (float): Tamaño del paso.
        x_end (float): Valor final de x para el cual resolver.

    Returns:
        tuple: (np.array, np.array)
            - Array de valores x.
            - Array de valores y correspondientes.
    """
    if h <= 0:
        raise ValueError("El tamaño del paso h debe ser positivo.")
    if (x_end > x0 and h < 0) or (x_end < x0 and h > 0):
        # Podrías permitir pasos negativos si x_end < x0, pero por simplicidad lo restringimos.
        raise ValueError("Conflicto entre x_end y el signo de h.")

    num_steps = int(abs(x_end - x0) / abs(h))
    # Asegurarse de que el último punto sea exactamente x_end si es posible
    # Esto puede ser más complejo si h no divide exactamente (x_end - x0)
    # Por simplicidad, haremos un número fijo de pasos y el último puede no ser exacto
    # O podemos ajustar el último paso

    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)

    x_values[0] = x0
    y_values[0] = y0

    x_current = x0
    y_current = y0

    for i in range(num_steps):
        # Ajustar el último paso si es necesario para alcanzar x_end exactamente
        current_h = h
        if x_current + current_h > x_end and x_current < x_end and h > 0 : # Para h positivo
            current_h = x_end - x_current
        elif x_current + current_h < x_end and x_current > x_end and h < 0: # Para h negativo
            current_h = x_end - x_current
        
        y_next = y_current + current_h * func_deriv(x_current, y_current)
        x_current += current_h

        x_values[i+1] = x_current
        y_values[i+1] = y_next
        y_current = y_next
        
        if (h > 0 and x_current >= x_end) or (h < 0 and x_current <= x_end):
            # Recortar si se pasó o alcanzó exactamente
            x_values = x_values[:i+2]
            y_values = y_values[:i+2]
            break
            
    return x_values, y_values


def rk4_method(func_deriv, x0, y0, h, x_end):
    """
    Resuelve la EDO dy/dx = func_deriv(x, y) usando el método de Runge-Kutta de 4º orden.

    Args:
        func_deriv (callable): La función que define la derivada dy/dx, f(x, y).
        x0 (float): Condición inicial para x.
        y0 (float): Condición inicial para y.
        h (float): Tamaño del paso.
        x_end (float): Valor final de x para el cual resolver.

    Returns:
        tuple: (np.array, np.array)
            - Array de valores x.
            - Array de valores y correspondientes.
    """
    if h <= 0:
        raise ValueError("El tamaño del paso h debe ser positivo.")
    if (x_end > x0 and h < 0) or (x_end < x0 and h > 0):
        raise ValueError("Conflicto entre x_end y el signo de h.")

    num_steps = int(abs(x_end - x0) / abs(h))

    x_values = np.zeros(num_steps + 1)
    y_values = np.zeros(num_steps + 1)
    
    x_values[0] = x0
    y_values[0] = y0

    x_current = x0
    y_current = y0
    
    for i in range(num_steps):
        current_h = h
        if x_current + current_h > x_end and x_current < x_end and h > 0 :
            current_h = x_end - x_current
        elif x_current + current_h < x_end and x_current > x_end and h < 0:
            current_h = x_end - x_current

        k1 = current_h * func_deriv(x_current, y_current)
        k2 = current_h * func_deriv(x_current + 0.5 * current_h, y_current + 0.5 * k1)
        k3 = current_h * func_deriv(x_current + 0.5 * current_h, y_current + 0.5 * k2)
        k4 = current_h * func_deriv(x_current + current_h, y_current + k3)

        y_next = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        x_current += current_h

        x_values[i+1] = x_current
        y_values[i+1] = y_next
        y_current = y_next

        if (h > 0 and x_current >= x_end) or (h < 0 and x_current <= x_end):
            x_values = x_values[:i+2]
            y_values = y_values[:i+2]
            break
            
    return x_values, y_values


# --- Ejemplo de cómo se podría usar (SOLO PARA PRUEBAS INTERNAS DEL MÓDULO) ---
# Este bloque __main__ no es estrictamente necesario si solo es una biblioteca,
# pero es útil para probar los métodos de forma aislada.
if __name__ == "__main__":
    print("--- Pruebas internas del módulo metodos_numericos.py ---")

    # Prueba de Bisección
    func_prueba_raiz = lambda x: x**2 - 4 # Raíces en -2 y 2
    raiz, msg = bisection_method(func_prueba_raiz, 0, 3, tol=1e-8)
    print(f"Bisección: Raíz = {raiz}, Mensaje: {msg}, f(raiz)={func_prueba_raiz(raiz) if raiz is not None else 'N/A'}")

    # Prueba de Newton-Raphson
    deriv_prueba_raiz = lambda x: 2*x
    raiz_nr, msg_nr = newton_raphson_method(func_prueba_raiz, deriv_prueba_raiz, 1.0, tol=1e-8)
    print(f"Newton-Raphson: Raíz = {raiz_nr}, Mensaje: {msg_nr}, f(raiz)={func_prueba_raiz(raiz_nr) if raiz_nr is not None else 'N/A'}")

    # Prueba de Trapecio
    func_prueba_integral = lambda x: x**3 # Integral de 0 a 1 es 0.25
    integral_val, msg_int = trapezoidal_rule(func_prueba_integral, 0, 1, n=200)
    print(f"Trapecio: Integral = {integral_val}, Mensaje: {msg_int}")

    # Prueba de Euler
    # dy/dx = y, y(0)=1 -> Sol: e^x
    func_prueba_edo = lambda x, y: y
    x_vals_e, y_vals_e = euler_method(func_prueba_edo, 0, 1, 0.01, 1)
    print(f"Euler: y(1) ~ {y_vals_e[-1]:.5f} (Analítico: {np.exp(1):.5f })")

    # Prueba de RK4
    x_vals_rk4, y_vals_rk4 = rk4_method(func_prueba_edo, 0, 1, 0.01, 1)
    print(f"RK4: y(1) ~ {y_vals_rk4[-1]:.5f} (Analítico: {np.exp(1):.5f})")
