# metodos_numericos.py
# (Biblioteca de métodos numéricos)
# metodos_numericos.py

from typing import Callable, Tuple, Optional, List 
import numpy as np

# --- EXCEPCIONES PERSONALIZADAS ---
class NumericalMethodError(Exception):
    """Clase base para errores en métodos numéricos."""
    pass

class InputError(NumericalMethodError):
    """Error por entrada inválida para el método."""
    pass

class ConvergenceError(NumericalMethodError): # Definida, aunque su uso activo dependerá de la lógica específica
    """Error cuando un método no converge como se esperaba (si se considera un error)."""
    pass

# --- MÓDULO: RAÍCES DE ECUACIONES ---

def bisection_method(func, a, b, tol=1e-7, max_iter=100):
    """
    Encuentra una raíz de la función 'func' en el intervalo [a, b]
    usando el método de bisección.

    Args:
        func (callable): La función para la cual encontrar la raíz.
        a (float): Límite inferior del intervalo.
        b (float): Límite superior del intervalo.
        tol (float): Tolerancia para la convergencia.
        max_iter (int): Número máximo de iteraciones.

    Returns:
        tuple: (float, str)
            - La raíz aproximada.
            - Un mensaje indicando el estado de la convergencia.

    Raises:
        InputError: Si los parámetros de entrada no son válidos o no se cumplen las precondiciones.
        NumericalMethodError: Si ocurre un error durante la evaluación de la función.
    """
    if not callable(func):
        raise InputError("La 'func' proporcionada no es una función llamable.")
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise InputError("'a' y 'b' deben ser valores numéricos.")
    if a >= b:
        raise InputError("El límite inferior 'a' debe ser estrictamente menor que 'b'.")
    if tol <= 0:
        raise InputError("La tolerancia 'tol' debe ser positiva.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise InputError("El número máximo de iteraciones 'max_iter' debe ser un entero positivo.")

    try:
        fa = func(a)
        fb = func(b)
    except Exception as e:
        raise NumericalMethodError(f"Error al evaluar la función en los límites del intervalo [{a}, {b}]: {e}")

    if fa * fb >= 0:
        raise InputError("La función debe tener signos opuestos en los extremos a y b para el método de bisección.")

    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        c = (a + b) / 2.0
        try:
            fc = func(c)
        except Exception as e:
            raise NumericalMethodError(f"Error al evaluar la función en el punto c={c}: {e}")

        if fc == 0 or (b - a) / 2.0 < tol:
            return c, f"Bisección convergió en {iterations_done} iteraciones."
        
        if fc * fa < 0:
            b = c
            # fb = fc # Opcional: actualizar fb si se usara más adelante en el bucle
        else:
            a = c
            fa = fc # Necesario para mantener la condición de signo
            
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
        tuple: (float, str)
            - La raíz aproximada.
            - Un mensaje indicando el estado de la convergencia.
    
    Raises:
        InputError: Si los parámetros de entrada no son válidos.
        NumericalMethodError: Si la derivada es cercana a cero o si ocurre un error durante la evaluación.
    """
    if not callable(func) or not callable(func_derivative):
        raise InputError("La 'func' y 'func_derivative' proporcionadas deben ser funciones llamables.")
    if not isinstance(x0, (int, float)):
        raise InputError("La estimación inicial 'x0' debe ser un valor numérico.")
    if tol <= 0:
        raise InputError("La tolerancia 'tol' debe ser positiva.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise InputError("El número máximo de iteraciones 'max_iter' debe ser un entero positivo.")

    x_n = x0
    iterations_done = 0
    for i in range(max_iter):
        iterations_done = i + 1
        try:
            fx_n = func(x_n)
            dfx_n = func_derivative(x_n)
        except Exception as e:
            raise NumericalMethodError(f"Error al evaluar la función o su derivada en x_n={x_n}: {e}")

        if abs(dfx_n) < 1e-12: # Evitar división por cero
            raise NumericalMethodError("Newton-Raphson: Derivada cercana a cero. No se puede continuar.")

        x_n_plus_1 = x_n - fx_n / dfx_n

        if abs(x_n_plus_1 - x_n) < tol:
            return x_n_plus_1, f"Newton-Raphson convergió en {iterations_done} iteraciones."

        x_n = x_n_plus_1

    return x_n, f"Newton-Raphson: Máx. iteraciones ({max_iter}) alcanzado. Mejor aprox."

# --- MÓDULO: INTEGRACIÓN NUMÉRICA ---

def trapezoidal_rule(func, a, b, n=100):
    """
    Calcula la integral definida de 'func' de 'a' a 'b' usando la regla del trapecio.
    Args: ...
    Returns:
        tuple: (float, str) - El valor aproximado de la integral y un mensaje.
    Raises:
        InputError: Si los parámetros de entrada no son válidos.
        NumericalMethodError: Si ocurre un error durante la evaluación de la función.
    """
    if not callable(func):
        raise InputError("La 'func' proporcionada no es una función llamable.")
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise InputError("'a' y 'b' deben ser valores numéricos.")
    # Podríamos permitir a > b y calcular la integral con signo negativo,
    # pero para simplificar la GUI actual, mantendremos a < b como implícito.
    # if a > b: raise InputError("'a' no puede ser mayor que 'b'.")
    if not isinstance(n, int) or n <= 0:
        raise InputError("El número de subintervalos 'n' debe ser un entero positivo.")

    h = (b - a) / n
    try:
        integral = 0.5 * (func(a) + func(b))
        for i in range(1, n):
            integral += func(a + i * h)
    except Exception as e:
        raise NumericalMethodError(f"Error al evaluar la función durante la integración: {e}")
        
    integral *= h
    return integral, f"Calculado con {n} subintervalos."

def simpsons_1_3_rule(func, a, b, n=100):
    """
    Calcula la integral definida de 'func' de 'a' a 'b' usando la regla de Simpson 1/3.
    Args: ...
    Returns:
        tuple: (float, str) - El valor aproximado de la integral y un mensaje.
    Raises:
        InputError: Si los parámetros de entrada no son válidos.
        NumericalMethodError: Si ocurre un error durante la evaluación de la función.
    """
    if not callable(func):
        raise InputError("La 'func' proporcionada no es una función llamable.")
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        raise InputError("'a' y 'b' deben ser valores numéricos.")
    if not isinstance(n, int) or n <= 0:
        raise InputError("El número de subintervalos 'n' debe ser un entero positivo.")
    if n % 2 != 0:
        raise InputError("El número de subintervalos 'n' debe ser par para la regla de Simpson 1/3.")

    h = (b - a) / n
    try:
        integral = func(a) + func(b)
        for i in range(1, n):
            x_i = a + i * h
            if i % 2 == 0: # Puntos pares
                integral += 2 * func(x_i)
            else: # Puntos impares
                integral += 4 * func(x_i)
    except Exception as e:
        raise NumericalMethodError(f"Error al evaluar la función durante la integración: {e}")

    integral *= (h / 3.0)
    return integral, f"Calculado con {n} subintervalos (pares)."

# --- MÓDULO: ECUACIONES DIFERENCIALES ORDINARIAS (EDOs) ---

def euler_method(func_deriv, x0, y0, h, x_end):
    """
    Resuelve la EDO dy/dx = func_deriv(x, y) usando el método de Euler.
    Args: ...
    Returns: (np.array, np.array) - Arrays de valores x e y.
    Raises:
        InputError: Si los parámetros de entrada no son válidos.
        NumericalMethodError: Si ocurre un error durante la evaluación de la función derivada.
    """
    if not callable(func_deriv):
        raise InputError("La 'func_deriv' proporcionada no es una función llamable.")
    if not all(isinstance(val, (int, float)) for val in [x0, y0, h, x_end]):
        raise InputError("x0, y0, h, y x_end deben ser valores numéricos.")
    if h == 0:
        raise InputError("El tamaño del paso h no puede ser cero.")
    if (x_end > x0 and h < 0) or (x_end < x0 and h > 0) or (x_end == x0): # Se agregó x_end == x0
        raise InputError("Conflicto entre x_end, x0 y el signo/valor de h. No hay intervalo para calcular o dirección incorrecta.")
    
    # Reajuste de num_steps para evitar bucles infinitos si h es muy pequeño y x_end está lejos
    if abs(x_end - x0) / abs(h) > 1_000_000: # Límite arbitrario para prevenir demasiados pasos
        raise InputError("La combinación de (x_end - x0) / h resulta en demasiados pasos (>1,000,000). Ajusta los parámetros.")

    num_steps = int(abs(x_end - x0) / abs(h))
    if num_steps == 0 : num_steps = 1 # Asegurar al menos un paso si x_end está muy cerca de x0

    x_values = np.zeros(num_steps + 2) # Un poco más de espacio por si el ajuste del último h añade un punto
    y_values = np.zeros(num_steps + 2)

    x_values[0] = x0
    y_values[0] = y0
    x_current = x0
    y_current = y0
    
    actual_steps = 0
    for i in range(num_steps +1): # +1 para dar oportunidad al ajuste de h
        actual_steps = i
        current_h = h
        # Ajustar el último paso
        if (h > 0 and x_current + h > x_end) or \
           (h < 0 and x_current + h < x_end):
            if abs(x_end - x_current) < 1e-9: # Ya estamos en x_end, no más pasos
                 break
            current_h = x_end - x_current
            if abs(current_h) < 1e-12 : # El paso restante es demasiado pequeño
                break
        
        try:
            y_next = y_current + current_h * func_deriv(x_current, y_current)
        except Exception as e:
            raise NumericalMethodError(f"Error al evaluar la función derivada en (x={x_current}, y={y_current}): {e}")
            
        x_current += current_h
        y_current = y_next

        if i + 1 < len(x_values):
            x_values[i+1] = x_current
            y_values[i+1] = y_current
        else: # Esto no debería ocurrir si num_steps está bien calculado, pero como salvaguarda
            # Podríamos expandir los arrays o detenernos. Por ahora, nos detenemos.
            # print("Advertencia: Se alcanzó el límite de tamaño del array de resultados en Euler.")
            break 

        if (h > 0 and x_current >= x_end - 1e-9) or \
           (h < 0 and x_current <= x_end + 1e-9): # Usar tolerancia para la comparación de flotantes
            break
            
    return x_values[:actual_steps+1], y_values[:actual_steps+1]


def rk4_method(func_deriv, x0, y0, h, x_end):
    """
    Resuelve la EDO dy/dx = func_deriv(x, y) usando el método de Runge-Kutta de 4º orden.
    Args: ...
    Returns: (np.array, np.array) - Arrays de valores x e y.
    Raises:
        InputError: Si los parámetros de entrada no son válidos.
        NumericalMethodError: Si ocurre un error durante la evaluación de la función derivada.
    """
    if not callable(func_deriv):
        raise InputError("La 'func_deriv' proporcionada no es una función llamable.")
    if not all(isinstance(val, (int, float)) for val in [x0, y0, h, x_end]):
        raise InputError("x0, y0, h, y x_end deben ser valores numéricos.")
    if h == 0:
        raise InputError("El tamaño del paso h no puede ser cero.")
    if (x_end > x0 and h < 0) or (x_end < x0 and h > 0) or (x_end == x0):
        raise InputError("Conflicto entre x_end, x0 y el signo/valor de h. No hay intervalo para calcular o dirección incorrecta.")

    if abs(x_end - x0) / abs(h) > 1_000_000:
        raise InputError("La combinación de (x_end - x0) / h resulta en demasiados pasos (>1,000,000). Ajusta los parámetros.")

    num_steps = int(abs(x_end - x0) / abs(h))
    if num_steps == 0 : num_steps = 1

    x_values = np.zeros(num_steps + 2)
    y_values = np.zeros(num_steps + 2)
    
    x_values[0] = x0
    y_values[0] = y0
    x_current = x0
    y_current = y0
    
    actual_steps = 0
    for i in range(num_steps + 1):
        actual_steps = i
        current_h = h
        if (h > 0 and x_current + h > x_end) or \
           (h < 0 and x_current + h < x_end):
            if abs(x_end - x_current) < 1e-9: 
                 break
            current_h = x_end - x_current
            if abs(current_h) < 1e-12 :
                break
        try:
            k1 = current_h * func_deriv(x_current, y_current)
            k2 = current_h * func_deriv(x_current + 0.5 * current_h, y_current + 0.5 * k1)
            k3 = current_h * func_deriv(x_current + 0.5 * current_h, y_current + 0.5 * k2)
            k4 = current_h * func_deriv(x_current + current_h, y_current + k3)
        except Exception as e:
            raise NumericalMethodError(f"Error al evaluar la función derivada en (x={x_current}, y={y_current}) o puntos intermedios: {e}")

        y_next = y_current + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        x_current += current_h

        if i + 1 < len(x_values):
            x_values[i+1] = x_current
            y_values[i+1] = y_next
            y_current = y_next
        else:
            # print("Advertencia: Se alcanzó el límite de tamaño del array de resultados en RK4.")
            break

        if (h > 0 and x_current >= x_end - 1e-9) or \
           (h < 0 and x_current <= x_end + 1e-9):
            break
            
    return x_values[:actual_steps+1], y_values[:actual_steps+1]


# --- Ejemplo de cómo se podría usar (SOLO PARA PRUEBAS INTERNAS DEL MÓDULO) ---
if __name__ == "__main__":
    print("--- Pruebas internas del módulo metodos_numericos.py ---")

    def test_bisection():
        print("\n--- Test Bisección ---")
        func_prueba_raiz = lambda x: x**2 - 4 # Raíces en -2 y 2
        
        # Caso 1: Raíz conocida
        try:
            raiz, msg = bisection_method(func_prueba_raiz, 0, 3, tol=1e-8)
            assert abs(raiz - 2.0) < 1e-7, f"Bisección Caso 1 Falló: raiz={raiz}, msg={msg}"
            print(f"Bisección (0,3) -> Raíz = {raiz:.7f}, Mensaje: {msg}, f(raiz)={func_prueba_raiz(raiz):.2e}")
        except Exception as e:
            print(f"Bisección Caso 1 Falló con excepción: {e}")

        # Caso 2: Sin cambio de signo (espera InputError)
        try:
            bisection_method(func_prueba_raiz, 2.1, 3, tol=1e-8)
            assert False, "Bisección Caso 2 Falló: Se esperaba InputError"
        except InputError as ie:
            print(f"Bisección (2.1,3) -> OK. Error esperado: {ie}")
        except Exception as e:
            assert False, f"Bisección Caso 2 Falló: Se esperaba InputError, se obtuvo {type(e)}"
        
        # Caso 3: a >= b
        try:
            bisection_method(func_prueba_raiz, 3, 0, tol=1e-8)
            assert False, "Bisección Caso 3 Falló: Se esperaba InputError por a>=b"
        except InputError as ie:
            print(f"Bisección (3,0) -> OK. Error esperado: {ie}")
        except Exception as e:
            assert False, f"Bisección Caso 3 Falló: Se esperaba InputError, se obtuvo {type(e)}"
        
        print("Pruebas de Bisección (parcialmente) completadas.")

    def test_newton_raphson():
        print("\n--- Test Newton-Raphson ---")
        func_prueba_raiz = lambda x: x**2 - 4
        deriv_prueba_raiz = lambda x: 2*x
        
        # Caso 1: Raíz conocida
        try:
            raiz_nr, msg_nr = newton_raphson_method(func_prueba_raiz, deriv_prueba_raiz, 1.0, tol=1e-8)
            assert abs(raiz_nr - 2.0) < 1e-7, f"Newton-Raphson Caso 1 Falló: raiz={raiz_nr}, msg={msg_nr}"
            print(f"Newton-Raphson (x0=1.0) -> Raíz = {raiz_nr:.7f}, Mensaje: {msg_nr}, f(raiz)={func_prueba_raiz(raiz_nr):.2e}")
        except Exception as e:
            print(f"Newton-Raphson Caso 1 Falló con excepción: {e}")

        # Caso 2: Derivada cero (espera NumericalMethodError)
        deriv_cero = lambda x: 0
        try:
            newton_raphson_method(func_prueba_raiz, deriv_cero, 1.0, tol=1e-8)
            assert False, "Newton-Raphson Caso 2 Falló: Se esperaba NumericalMethodError"
        except NumericalMethodError as nme:
            print(f"Newton-Raphson (derivada cero) -> OK. Error esperado: {nme}")
        except Exception as e:
            assert False, f"Newton-Raphson Caso 2 Falló: Se esperaba NumericalMethodError, se obtuvo {type(e)}"
        print("Pruebas de Newton-Raphson (parcialmente) completadas.")

    # Prueba de Trapecio
    def test_trapezoidal():
        print("\n--- Test Regla del Trapecio ---")
        func_prueba_integral = lambda x: x**3 # Integral de 0 a 1 es 0.25
        try:
            integral_val, msg_int = trapezoidal_rule(func_prueba_integral, 0, 1, n=200)
            assert abs(integral_val - 0.25) < 1e-3 # La regla del trapecio puede tener error
            print(f"Trapecio: Integral = {integral_val:.7f}, Mensaje: {msg_int}")
        except Exception as e:
            print(f"Trapecio Caso 1 Falló con excepción: {e}")
        
        try:
            trapezoidal_rule(func_prueba_integral, 0, 1, n=0)
            assert False, "Trapecio: Se esperaba InputError por n<=0"
        except InputError as ie:
            print(f"Trapecio (n=0) -> OK. Error esperado: {ie}")
        except Exception as e:
            assert False, f"Trapecio: Se esperaba InputError, se obtuvo {type(e)}"
        print("Pruebas de Trapecio (parcialmente) completadas.")
    
    # Prueba de Simpson
    def test_simpson():
        print("\n--- Test Regla de Simpson 1/3 ---")
        func_prueba_integral = lambda x: x**3
        try:
            integral_val, msg_int = simpsons_1_3_rule(func_prueba_integral, 0, 1, n=100) # n par
            assert abs(integral_val - 0.25) < 1e-6 # Simpson es más preciso
            print(f"Simpson 1/3: Integral = {integral_val:.7f}, Mensaje: {msg_int}")
        except Exception as e:
            print(f"Simpson Caso 1 Falló con excepción: {e}")
        
        try:
            simpsons_1_3_rule(func_prueba_integral, 0, 1, n=99) # n impar
            assert False, "Simpson: Se esperaba InputError por n impar"
        except InputError as ie:
            print(f"Simpson (n=99) -> OK. Error esperado: {ie}")
        except Exception as e:
            assert False, f"Simpson: Se esperaba InputError, se obtuvo {type(e)}"
        print("Pruebas de Simpson (parcialmente) completadas.")

    # Prueba de Euler
    def test_euler():
        print("\n--- Test Método de Euler ---")
        func_prueba_edo = lambda x, y: y # dy/dx = y, y(0)=1 -> Sol: e^x
        try:
            x_vals_e, y_vals_e = euler_method(func_prueba_edo, 0, 1, 0.01, 1)
            analytical_sol_at_1 = np.exp(1)
            assert abs(y_vals_e[-1] - analytical_sol_at_1) < 0.02 
            print(f"Euler: y(1) ~ {y_vals_e[-1]:.5f} (Analítico: {analytical_sol_at_1:.5f })")
        except Exception as e:
            print(f"Euler Caso 1 Falló con excepción: {e}")
        
        try:
            euler_method(func_prueba_edo, 0, 1, 0, 1) # h=0
            assert False, "Euler: Se esperaba InputError por h=0"
        except InputError as ie:
            print(f"Euler (h=0) -> OK. Error esperado: {ie}")
        except Exception as e:
            assert False, f"Euler: Se esperaba InputError, se obtuvo {type(e)}"
        print("Pruebas de Euler (parcialmente) completadas.")

    # Prueba de RK4
    def test_rk4():
        print("\n--- Test Método RK4 ---")
        func_prueba_edo = lambda x, y: y
        try:
            x_vals_rk4, y_vals_rk4 = rk4_method(func_prueba_edo, 0, 1, 0.01, 1)
            analytical_sol_at_1 = np.exp(1)
            assert abs(y_vals_rk4[-1] - analytical_sol_at_1) < 1e-5 # RK4 es más preciso
            print(f"RK4: y(1) ~ {y_vals_rk4[-1]:.5f} (Analítico: {analytical_sol_at_1:.5f})")
        except Exception as e:
            print(f"RK4 Caso 1 Falló con excepción: {e}")
        print("Pruebas de RK4 (parcialmente) completadas.")


    test_bisection()
    test_newton_raphson()
    test_trapezoidal()
    test_simpson()
    test_euler()
    test_rk4()
