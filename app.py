import numpy as np
import matplotlib.pyplot as plt

def web_server_model(A, F, B, I, Y, R, S, C):
    p = B / F
    A_prime = A / p
    
    T_s = I + (Y + B/R) + (B/S) + (B/C)
    T = T_s / (1 - A * T_s)
    
    return T

def simulate_web_server(config, num_simulations=1000, max_A=100):
    A_values = np.linspace(0.1, max_A, 100)  # Empezamos desde 0.1 para evitar división por cero
    T_values = []
    
    for A in A_values:
        T_samples = []
        for _ in range(num_simulations):
            F = np.random.exponential(config['F'])
            T = web_server_model(A, F, config['B'], config['I'], config['Y'], config['R'], config['S'], config['C'])
            if np.isfinite(T):
                T_samples.append(T)
        if T_samples:
            T_values.append(np.mean(T_samples))
        else:
            T_values.append(np.inf)
    
    return A_values, T_values

# Configuración inicial
config_initial = {
    'F': 5275,  # Tamaño promedio de archivo (bytes)
    'B': 2000,  # Tamaño del buffer (bytes)
    'I': 0.001,  # Tiempo de inicialización (segundos)
    'Y': 0.001,  # Tiempo de servidor estático (segundos)
    'R': 10e6,  # Tasa de servidor dinámico (bytes/segundo)
    'S': 1.5e6 / 8,  # Ancho de banda de red del servidor (bytes/segundo) - T1
    'C': 707e3 / 8   # Ancho de banda de red del cliente (bytes/segundo)
}

# Simulación de la configuración inicial
A_initial, T_initial = simulate_web_server(config_initial)

# Gráfica de la configuración inicial
plt.figure(figsize=(10, 6))
plt.plot(A_initial, T_initial)
plt.title('Tiempo de respuesta vs. Tasa de llegada (Configuración inicial)')
plt.xlabel('Tasa de llegada (solicitudes/segundo)')
plt.ylabel('Tiempo de respuesta (segundos)')
plt.ylim(0, 10)
plt.show()

# Calcular la capacidad máxima y otros datos relevantes
def calculate_metrics(A_values, T_values):
    M = A_values[np.argmax(np.array(T_values) > 10) - 1]
    avg_response_time = np.mean(T_values)
    max_response_time = np.max(T_values)
    return M, avg_response_time, max_response_time

# Calcular métricas para la configuración inicial
M_initial, avg_T_initial, max_T_initial = calculate_metrics(A_initial, T_initial)

print("\n--- Resultados de la simulación ---")
print("\nConfiguración inicial:")
print(f"Capacidad máxima: {M_initial:.2f} solicitudes/segundo")
print(f"Tiempo de respuesta promedio: {avg_T_initial:.4f} segundos")
print(f"Tiempo de respuesta máximo: {max_T_initial:.4f} segundos")
print(f"\nParámetros de configuración:")
for key, value in config_initial.items():
    print(f"  {key}: {value}")

# Mejora 1: Aumentar la velocidad del servidor
config_improved1 = config_initial.copy()
config_improved1['R'] = 20e6  # Duplicar la tasa de servidor dinámico

# Mejora 2: Aumentar el ancho de banda de la red
config_improved2 = config_initial.copy()
config_improved2['S'] = 3e6 / 8  # Duplicar el ancho de banda de la red (T1 a T2)

# Simulación de las mejoras
A_improved1, T_improved1 = simulate_web_server(config_improved1)
A_improved2, T_improved2 = simulate_web_server(config_improved2)

# Calcular métricas para las mejoras
M_improved1, avg_T_improved1, max_T_improved1 = calculate_metrics(A_improved1, T_improved1)
M_improved2, avg_T_improved2, max_T_improved2 = calculate_metrics(A_improved2, T_improved2)

print("\nMejora 1 (Aumento de velocidad del servidor):")
print(f"Capacidad máxima: {M_improved1:.2f} solicitudes/segundo")
print(f"Tiempo de respuesta promedio: {avg_T_improved1:.4f} segundos")
print(f"Tiempo de respuesta máximo: {max_T_improved1:.4f} segundos")
print(f"Mejora en capacidad: {((M_improved1 - M_initial) / M_initial * 100):.2f}%")

print("\nMejora 2 (Aumento de ancho de banda de red):")
print(f"Capacidad máxima: {M_improved2:.2f} solicitudes/segundo")
print(f"Tiempo de respuesta promedio: {avg_T_improved2:.4f} segundos")
print(f"Tiempo de respuesta máximo: {max_T_improved2:.4f} segundos")
print(f"Mejora en capacidad: {((M_improved2 - M_initial) / M_initial * 100):.2f}%")

print("\nComparación de mejoras:")
if M_improved1 > M_improved2:
    print("La Mejora 1 (velocidad del servidor) proporciona una mayor capacidad máxima.")
elif M_improved2 > M_improved1:
    print("La Mejora 2 (ancho de banda de red) proporciona una mayor capacidad máxima.")
else:
    print("Ambas mejoras proporcionan la misma capacidad máxima.")

print("\nRecomendación:")
if M_improved2 > M_improved1:
    print("Se recomienda implementar la Mejora 2 (aumento de ancho de banda de red) para obtener el mejor rendimiento.")
else:
    print("Se recomienda implementar la Mejora 1 (aumento de velocidad del servidor) para obtener el mejor rendimiento.")

# Gráfica comparativa
plt.figure(figsize=(10, 6))
plt.plot(A_initial, T_initial, label='Inicial')
plt.plot(A_improved1, T_improved1, label='Mejora 1: Velocidad del servidor')
plt.plot(A_improved2, T_improved2, label='Mejora 2: Ancho de banda de red')
plt.title('Comparación de rendimiento')
plt.xlabel('Tasa de llegada (solicitudes/segundo)')
plt.ylabel('Tiempo de respuesta (segundos)')
plt.ylim(0, 10)
plt.legend()
plt.show()