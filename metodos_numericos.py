def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    if f(a) * f(b) >= 0:
        raise ValueError("El método de bisección no es aplicable en el intervalo dado.")
    
    iter_count = 0
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        if f(c) == 0:
            return c  # Encontramos la raíz exacta
        elif f(c) * f(a) < 0:
            b = c
        else:
            a = c

        iter_count += 1
        if iter_count > max_iter:
            raise ValueError("El número máximo de iteraciones ha sido alcanzado.")
    
    return (a + b) / 2
