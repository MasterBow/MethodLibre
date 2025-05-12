# main.py

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np # Para funciones matemáticas disponibles en eval() y para EDOs
import math      # Para constantes y funciones matemáticas disponibles en eval()
import metodos_numericos as mn
class App:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Calculadora de Métodos Numéricos")
        self.root.geometry("800x700") # Tamaño inicial

        # Estilo ttk
        style = ttk.Style()
        style.theme_use('clam') # 'clam', 'alt', 'default', 'classic'

        # --- Contenedor Principal ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Notebook para Pestañas ---
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(expand=True, fill=tk.BOTH, pady=10)

        # --- Pestañas ---
        self.tab_roots = ttk.Frame(self.notebook, padding="10")
        self.tab_integration = ttk.Frame(self.notebook, padding="10")
        self.tab_odes = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.tab_roots, text='Raíces de Ecuaciones')
        self.notebook.add(self.tab_integration, text='Integración Numérica')
        self.notebook.add(self.tab_odes, text='Ecuaciones Diferenciales')

        # --- Área Común de Resultados ---
        results_frame = ttk.LabelFrame(main_frame, text="Resultados y Mensajes", padding="10")
        results_frame.pack(expand=True, fill=tk.BOTH, side=tk.BOTTOM, pady=(0,10))
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=8, font=("Arial", 10))
        self.results_text.pack(expand=True, fill=tk.BOTH)
        self.results_text.configure(state='disabled') # Solo lectura al inicio

        # --- Configurar GUI para cada pestaña ---
        self._setup_roots_tab()
        self._setup_integration_tab()
        self._setup_odes_tab()

        # Diccionario de funciones matemáticas seguras para eval()
        self.safe_math_dict = {
            "np": np, "numpy": np,
            "math": math,
            "sin": np.sin, "cos": np.cos, "tan": np.tan,
            "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
            "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
            "exp": np.exp, "log": np.log, "log10": np.log10, "log2": np.log2,
            "sqrt": np.sqrt, "pow": pow,
            "pi": np.pi, "e": np.e,
            "abs": np.abs, "fabs": math.fabs,
            "ceil": np.ceil, "floor": np.floor,
            # Puedes añadir más funciones o constantes seguras aquí
        }


    def _clear_results_text(self):
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.configure(state='disabled')

    def _append_results_text(self, text):
        self.results_text.configure(state='normal')
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.configure(state='disabled')

    def _parse_function(self, func_str, var_names=('x',)):
        """Convierte un string de función en una función callable."""
        try:
            # Crear el diccionario local para eval()
            local_dict = self.safe_math_dict.copy()

            if 'y' in var_names and 'x' in var_names : # Para f(x,y) de EDOs
                func = lambda x, y: eval(func_str, {"__builtins__": {}}, {**local_dict, "x": x, "y": y})
                # Probar la función con valores de muestra para asegurar que está bien definida
                func(1.0, 1.0) 
                return func
            elif 'x' in var_names: # Para f(x)
                func = lambda x: eval(func_str, {"__builtins__": {}}, {**local_dict, "x": x})
                # Probar la función
                func(1.0)
                return func
            else:
                raise ValueError("Nombres de variables no soportados para parse_function.")

        except Exception as e:
            messagebox.showerror("Error de Función", f"Error al interpretar la función '{func_str}':\n{e}\n\n"
                                 "Asegúrate de usar 'np.' para funciones de numpy (ej. np.sin(x)) "
                                 "o funciones de math (ej. math.exp(x)) si están en safe_math_dict. "
                                 "Variables permitidas: " + ", ".join(var_names))
            return None

    # --- Pestaña: Raíces de Ecuaciones ---
    def _setup_roots_tab(self):
        frame = self.tab_roots
        
        # Entradas
        ttk.Label(frame, text="Función f(x):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.roots_func_entry = ttk.Entry(frame, width=40)
        self.roots_func_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        self.roots_func_entry.insert(0, "x**3 - x - 2") # Ejemplo

        ttk.Label(frame, text="Derivada f'(x) (para Newton):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.roots_dfunc_entry = ttk.Entry(frame, width=40)
        self.roots_dfunc_entry.grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        self.roots_dfunc_entry.insert(0, "3*x**2 - 1") # Ejemplo

        ttk.Label(frame, text="a (Bisección):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.roots_a_entry = ttk.Entry(frame, width=10)
        self.roots_a_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.roots_a_entry.insert(0, "1.0")

        ttk.Label(frame, text="b (Bisección):").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.roots_b_entry = ttk.Entry(frame, width=10)
        self.roots_b_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        self.roots_b_entry.insert(0, "2.0")
        
        ttk.Label(frame, text="x0 (Newton):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.roots_x0_entry = ttk.Entry(frame, width=10)
        self.roots_x0_entry.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        self.roots_x0_entry.insert(0, "1.5")

        ttk.Label(frame, text="Tol:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.roots_tol_entry = ttk.Entry(frame, width=10)
        self.roots_tol_entry.grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)
        self.roots_tol_entry.insert(0, "1e-7")

        ttk.Label(frame, text="MaxIter:").grid(row=4, column=2, sticky=tk.W, padx=5, pady=2)
        self.roots_maxiter_entry = ttk.Entry(frame, width=10)
        self.roots_maxiter_entry.grid(row=4, column=3, sticky=tk.W, padx=5, pady=2)
        self.roots_maxiter_entry.insert(0, "100")

        # Botones
        ttk.Button(frame, text="Calcular Bisección", command=self._solve_bisection).grid(row=5, column=0, columnspan=2, pady=10, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Calcular Newton-Raphson", command=self._solve_newton).grid(row=5, column=2, columnspan=2, pady=10, padx=5, sticky=tk.EW)

    def _solve_bisection(self):
        self._clear_results_text()
        try:
            func_str = self.roots_func_entry.get()
            a = float(self.roots_a_entry.get())
            b = float(self.roots_b_entry.get())
            tol = float(self.roots_tol_entry.get())
            max_iter = int(self.roots_maxiter_entry.get())

            user_func = self._parse_function(func_str, var_names=('x',))
            if user_func is None: return

            raiz, mensaje = mn.bisection_method(user_func, a, b, tol, max_iter)

            self._append_results_text(f"--- Método de Bisección ---")
            self._append_results_text(f"Función: {func_str}")
            self._append_results_text(f"Intervalo: [{a}, {b}], Tol: {tol}, MaxIter: {max_iter}")
            self._append_results_text(mensaje)
            if raiz is not None:
                self._append_results_text(f"Raíz encontrada: {raiz:.10f}")
                try:
                    f_raiz = user_func(raiz)
                    self._append_results_text(f"f(raíz): {f_raiz:.3e}")
                except Exception as e_eval:
                    self._append_results_text(f"Error al evaluar f(raíz): {e_eval}")
            else:
                self._append_results_text("No se pudo encontrar la raíz con los parámetros dados.")

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Por favor, ingresa valores numéricos válidos.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            self._append_results_text(f"Error durante el cálculo: {e}")

    def _solve_newton(self):
        self._clear_results_text()
        try:
            func_str = self.roots_func_entry.get()
            dfunc_str = self.roots_dfunc_entry.get()
            x0 = float(self.roots_x0_entry.get())
            tol = float(self.roots_tol_entry.get())
            max_iter = int(self.roots_maxiter_entry.get())

            user_func = self._parse_function(func_str, var_names=('x',))
            user_dfunc = self._parse_function(dfunc_str, var_names=('x',))

            if user_func is None or user_dfunc is None: return

            raiz, mensaje = mn.newton_raphson_method(user_func, user_dfunc, x0, tol, max_iter)

            self._append_results_text(f"--- Método de Newton-Raphson ---")
            self._append_results_text(f"Función f(x): {func_str}")
            self._append_results_text(f"Derivada f'(x): {dfunc_str}")
            self._append_results_text(f"x0: {x0}, Tol: {tol}, MaxIter: {max_iter}")
            self._append_results_text(mensaje)
            if raiz is not None:
                self._append_results_text(f"Raíz encontrada: {raiz:.10f}")
                try:
                    f_raiz = user_func(raiz)
                    self._append_results_text(f"f(raíz): {f_raiz:.3e}")
                except Exception as e_eval:
                    self._append_results_text(f"Error al evaluar f(raíz): {e_eval}")
            else:
                self._append_results_text("No se pudo encontrar la raíz con los parámetros dados.")

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Por favor, ingresa valores numéricos válidos.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            self._append_results_text(f"Error durante el cálculo: {e}")

    # --- Pestaña: Integración Numérica ---
    def _setup_integration_tab(self):
        frame = self.tab_integration
        ttk.Label(frame, text="Función f(x):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.integ_func_entry = ttk.Entry(frame, width=40)
        self.integ_func_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        self.integ_func_entry.insert(0, "x**2")

        ttk.Label(frame, text="Límite inferior a:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.integ_a_entry = ttk.Entry(frame, width=10)
        self.integ_a_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.integ_a_entry.insert(0, "0.0")

        ttk.Label(frame, text="Límite superior b:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.integ_b_entry = ttk.Entry(frame, width=10)
        self.integ_b_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        self.integ_b_entry.insert(0, "1.0")

        ttk.Label(frame, text="Número de subintervalos n:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.integ_n_entry = ttk.Entry(frame, width=10)
        self.integ_n_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.integ_n_entry.insert(0, "100")

        ttk.Button(frame, text="Calcular Trapecio", command=self._solve_trapezoidal).grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky=tk.EW)
        ttk.Button(frame, text="Calcular Simpson 1/3", command=self._solve_simpson).grid(row=3, column=2, columnspan=2, pady=10, padx=5, sticky=tk.EW)

    def _solve_trapezoidal(self):
        self._clear_results_text()
        try:
            func_str = self.integ_func_entry.get()
            a = float(self.integ_a_entry.get())
            b = float(self.integ_b_entry.get())
            n = int(self.integ_n_entry.get())

            user_func = self._parse_function(func_str, var_names=('x',))
            if user_func is None: return

            integral, mensaje = mn.trapezoidal_rule(user_func, a, b, n)

            self._append_results_text(f"--- Regla del Trapecio ---")
            self._append_results_text(f"Función: {func_str}")
            self._append_results_text(f"Intervalo: [{a}, {b}], Subintervalos n: {n}")
            if integral is not None:
                 self._append_results_text(f"Valor de la integral: {integral:.10f}")
            self._append_results_text(f"Mensaje: {mensaje}")

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Por favor, ingresa valores numéricos válidos.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            self._append_results_text(f"Error durante el cálculo: {e}")

    def _solve_simpson(self):
        self._clear_results_text()
        try:
            func_str = self.integ_func_entry.get()
            a = float(self.integ_a_entry.get())
            b = float(self.integ_b_entry.get())
            n = int(self.integ_n_entry.get())

            user_func = self._parse_function(func_str, var_names=('x',))
            if user_func is None: return

            integral, mensaje = mn.simpsons_1_3_rule(user_func, a, b, n)
            
            self._append_results_text(f"--- Regla de Simpson 1/3 ---")
            self._append_results_text(f"Función: {func_str}")
            self._append_results_text(f"Intervalo: [{a}, {b}], Subintervalos n: {n}")
            if integral is not None:
                 self._append_results_text(f"Valor de la integral: {integral:.10f}")
            self._append_results_text(f"Mensaje: {mensaje}")
            if n % 2 != 0 and integral is None: # El método ya maneja esto, pero para claridad
                 self._append_results_text("Recordatorio: 'n' debe ser par para Simpson 1/3.")


        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Por favor, ingresa valores numéricos válidos.\n{ve}")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            self._append_results_text(f"Error durante el cálculo: {e}")

    # --- Pestaña: Ecuaciones Diferenciales ---
    def _setup_odes_tab(self):
        frame = self.tab_odes
        
        # Frame para controles y frame para gráfico
        controls_frame = ttk.Frame(frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        graph_frame = ttk.LabelFrame(frame, text="Gráfico de Solución")
        graph_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=5)

        ttk.Label(controls_frame, text="Función dy/dx = f(x,y):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ode_func_entry = ttk.Entry(controls_frame, width=35)
        self.ode_func_entry.grid(row=0, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        self.ode_func_entry.insert(0, "x + y")

        ttk.Label(controls_frame, text="x0:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ode_x0_entry = ttk.Entry(controls_frame, width=10)
        self.ode_x0_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.ode_x0_entry.insert(0, "0.0")

        ttk.Label(controls_frame, text="y0:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.ode_y0_entry = ttk.Entry(controls_frame, width=10)
        self.ode_y0_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        self.ode_y0_entry.insert(0, "1.0")

        ttk.Label(controls_frame, text="Paso h:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.ode_h_entry = ttk.Entry(controls_frame, width=10)
        self.ode_h_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        self.ode_h_entry.insert(0, "0.1")

        ttk.Label(controls_frame, text="x_final:").grid(row=2, column=2, sticky=tk.W, padx=5, pady=2)
        self.ode_xend_entry = ttk.Entry(controls_frame, width=10)
        self.ode_xend_entry.grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)
        self.ode_xend_entry.insert(0, "1.0")

        ttk.Button(controls_frame, text="Calcular Euler", command=self._solve_euler).grid(row=3, column=0, columnspan=2, pady=10, padx=5, sticky=tk.EW)
        ttk.Button(controls_frame, text="Calcular RK4", command=self._solve_rk4).grid(row=3, column=2, columnspan=2, pady=10, padx=5, sticky=tk.EW)

        # Configuración del gráfico Matplotlib
        self.fig_ode, self.ax_ode = plt.subplots(figsize=(6,4))
        self.ax_ode.set_xlabel("x")
        self.ax_ode.set_ylabel("y")
        self.ax_ode.grid(True)
        
        self.canvas_ode = FigureCanvasTkAgg(self.fig_ode, master=graph_frame)
        self.canvas_ode_widget = self.canvas_ode.get_tk_widget()
        self.canvas_ode_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas_ode.draw()


    def _solve_ode_common(self, method_name, method_func):
        self._clear_results_text()
        self.ax_ode.clear() # Limpiar gráfico anterior
        self.ax_ode.set_xlabel("x") # Reestablecer etiquetas
        self.ax_ode.set_ylabel("y")
        self.ax_ode.grid(True)

        try:
            func_str = self.ode_func_entry.get()
            x0 = float(self.ode_x0_entry.get())
            y0 = float(self.ode_y0_entry.get())
            h = float(self.ode_h_entry.get())
            x_end = float(self.ode_xend_entry.get())

            user_func = self._parse_function(func_str, var_names=('x', 'y'))
            if user_func is None: 
                self.canvas_ode.draw() # Redibujar canvas vacío
                return

            x_values, y_values = method_func(user_func, x0, y0, h, x_end)

            self._append_results_text(f"--- Método {method_name} para EDOs ---")
            self._append_results_text(f"Función dy/dx: {func_str}")
            self._append_results_text(f"Condiciones: y({x0})={y0}, h={h}, x_final={x_end}")
            
            if x_values is not None and y_values is not None and len(x_values) > 0:
                self._append_results_text(f"Solución en x_final={x_values[-1]:.4f}: y={y_values[-1]:.7f}")
                self._append_results_text(f"Número de pasos: {len(x_values)-1}")

                # Graficar
                self.ax_ode.plot(x_values, y_values, marker='o', linestyle='-', markersize=3, label=method_name)
                self.ax_ode.legend()
            else:
                self._append_results_text("No se pudo calcular la solución.")
            
            self.canvas_ode.draw()

        except ValueError as ve:
            messagebox.showerror("Error de Entrada", f"Por favor, ingresa valores numéricos válidos.\n{ve}")
            self.canvas_ode.draw() # Redibujar
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {e}")
            self._append_results_text(f"Error durante el cálculo: {e}")
            self.canvas_ode.draw() # Redibujar

    def _solve_euler(self):
        self._solve_ode_common("Euler", mn.euler_method)

    def _solve_rk4(self):
        self._solve_ode_common("Runge-Kutta RK4", mn.rk4_method)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
