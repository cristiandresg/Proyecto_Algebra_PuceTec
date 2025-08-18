import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Estudiante:
    def __init__(self, nombre, apellido, fecha_nacimiento, nivel_educativo):
        self.nombre = nombre
        self.apellido = apellido
        self.fecha_nacimiento = fecha_nacimiento
        self.edad = self.calcular_edad()
        self.nivel_educativo = nivel_educativo
                
    def calcular_edad(self):
        hoy = datetime.date.today()
        edad = hoy.year - self.fecha_nacimiento.year
        if hoy < self.fecha_nacimiento.replace(year=hoy.year):
            edad -= 1
        return edad

     
    def determinar_nivel(self):
        rangos_edad = [
            (3, 4, "Kinder"),
            (5, 5, "Jardín de Infantes"),
            (6, 12, "Primaria"),
            (13, 17, "Secundaria"),
            (18, 25, "Universidad o Instituto"),
            (27, 99, "Otros estudios universitarios")
            
        ]
        for edad_min, edad_max, nivel in rangos_edad:
            if edad_min <= self.edad <= edad_max:
                return nivel
        return "No aplica a ningún nivel educativo"


    def __str__(self):
        return (f"Nombre: {self.nombre} {self.apellido} \n"
                f"Fecha de Nacimiento: {self.fecha_nacimiento.strftime('%d/%m/%Y')}\n"
                f"Edad: {self.edad} años\n"
                f"Nivel Educativo: {self.nivel_educativo}\n")

    def obtener_caracteristicas(self):
        niveles_map = {
            "Kinder": 1,
            "Jardín de Infantes": 2,
            "Primaria": 3,
            "Secundaria": 4,
            "Universidad o Instituto": 5,
            "Otros estudios universitarios": 6,
            "No aplica a ningún nivel educativo": 0
        }
        nivel_numerico = niveles_map.get(self.nivel_educativo, 0)
        return [self.edad, nivel_numerico]

# Se Mueve ingresar_estudiante fuera de la clase
def ingresar_estudiante():
    print("\nIngrese los datos del estudiante:")
    nombre = input(" Nombre: ").strip().title()
    apellido = input(" Apellido : ").strip().title()

    while True:
        fecha_str = input(" Fecha de nacimiento (DD/MM/AAAA): ").strip()
        try:
            dia, mes, anio = map(int, fecha_str.split('/'))
            if not (1 <= dia <= 31 and 1 <= mes <= 12 and 1900 <= anio <= datetime.date.today().year):
                print("Día, mes o año fuera de rango válido. Intente nuevamente.")
                continue
            fecha_nacimiento = datetime.date(anio, mes, dia)
            hoy = datetime.date.today()
            if fecha_nacimiento > hoy:
                print("La fecha de nacimiento no puede ser futura. Intente nuevamente.")
                continue
            if fecha_nacimiento.year < hoy.year - 100:
                print("La fecha de nacimiento parece ser demasiado antigua. Intente nuevamente.")
                continue
            break
        except ValueError:
            print("Formato de fecha incorrecto. Use DD/MM/AAAA con números válidos.")

    print("\nSeleccione el nivel educativo:")
    niveles = [
        "Kinder",
        "Jardín de Infantes",
        "Primaria",
        "Secundaria",
        "Universidad o Instituto",
        "Otros estudios universitarios",
        "No aplica a ningún nivel educativo"
    ]
    for i, nivel in enumerate(niveles, 1):
        print(f"{i}. {nivel}")
    while True:
        opcion = input("Ingrese el número correspondiente: ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(niveles):
            nivel_educativo = niveles[int(opcion) - 1]
            break
        else:
            print("Opción inválida. Intente nuevamente.")
    return Estudiante(nombre, apellido, fecha_nacimiento, nivel_educativo)

    
def buscar_por_edad(estudiantes):
    print("\n--- Buscar estudiantes por rango de edad ---")
    try:
        edad_min = int(input("Edad mínima: ").strip())
        edad_max = int(input("Edad máxima: ").strip())
        if edad_min > edad_max or edad_min < 0:
            print("Rango inválido. La edad mínima debe ser menor o igual a la máxima y mayor o igual a 0.")
            return
        encontrados = [e for e in estudiantes if edad_min <= e.edad <= edad_max]
        if encontrados:
            print(f"\nEstudiantes entre {edad_min} y {edad_max} años:")
            for est in encontrados:
                print(est)
        else:
            print(f"\nNo se encontraron estudiantes en el rango de {edad_min} a {edad_max} años.")
    except ValueError:
        print("Por favor ingrese valores numéricos válidos para las edades.")
    
def aplicar_pca(estudiantes):
    if len(estudiantes) < 3:
        print("\nSe necesitan al menos 3 estudiantes para un análisis PCA significativo.")
        return
    
    datos = np.array([e.obtener_caracteristicas() for e in estudiantes])
    nombres = [f"{e.nombre} {e.apellido}" for e in estudiantes]
    edades = [e.edad for e in estudiantes]
    niveles_educativos = [e.nivel_educativo for e in estudiantes]
    caracteristicas = ['Edad', 'Nivel Educativo (codificado)']
    
    # Calcular promedio de edades global y por nivel educativo
    promedio_edad_global = np.mean(edades)
    niveles_unicos = list(set(niveles_educativos))
    promedios_por_nivel = {
        nivel: np.mean([e.edad for e in estudiantes if e.nivel_educativo == nivel])
        for nivel in niveles_unicos
    }
    
    print(f"\n--- Aplicando Análisis de Componentes Principales (PCA) ---")
    print(f"Datos originales: {len(estudiantes)} estudiantes x {len(caracteristicas)} características.")
    print(f"Promedio de edad global: {promedio_edad_global:.1f} años")
    print("Promedio de edad por nivel educativo:")
    for nivel, promedio in promedios_por_nivel.items():
        print(f"  {nivel}: {promedio:.1f} años")
    
    try:
        scaler = StandardScaler()
        datos_escalados = scaler.fit_transform(datos)
        if np.any(np.isnan(datos_escalados)) or np.std(datos_escalados, axis=0).min() == 0:
            print("Error: Los datos no tienen suficiente variabilidad para PCA.")
            return
        
        n_componentes = min(2, datos_escalados.shape[1], len(estudiantes)-1)
        pca = PCA(n_components=n_componentes)
        componentes_principales = pca.fit_transform(datos_escalados)
        
        # Proyectar el promedio de edades en el espacio PCA
        promedio_datos = np.array([[promedio_edad_global, np.mean([e.obtener_caracteristicas()[1] for e in estudiantes])]])
        promedio_datos_escalado = scaler.transform(promedio_datos)
        promedio_proyectado = pca.transform(promedio_datos_escalado)
        
        print(f"\nDimensionalidad reducida a {n_componentes} componentes principales.")
        print("Varianza explicada por cada componente:")
        for i, var in enumerate(pca.explained_variance_ratio_):
            print(f"  PC{i+1}: {var:.2%} de la varianza total")
        
        plt.figure(figsize=(10, 8))
        
        if n_componentes == 1:
            # Gráfico 1D
            scatter = plt.scatter(componentes_principales[:, 0], np.zeros(len(componentes_principales)), 
                                 c=edades, cmap='viridis', alpha=0.7)
            plt.plot(componentes_principales[:, 0], np.zeros(len(componentes_principales)), 'r-', alpha=0.5, label='Trayectoria')
            # Marcar promedio de edades
            plt.scatter(promedio_proyectado[0, 0], 0, color='red', marker='*', s=200, label='Promedio de Edades')
            plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} Var.)')
            plt.title('Proyección 1D de Estudiantes (PCA)')
        else:
            # Gráfico 2D
            scatter = plt.scatter(componentes_principales[:, 0], componentes_principales[:, 1], 
                                 c=edades, cmap='viridis', alpha=0.7)
            plt.plot(componentes_principales[:, 0], componentes_principales[:, 1], 'r-', alpha=0.5, label='Trayectoria')
            # Marcar promedio de edades
            plt.scatter(promedio_proyectado[0, 0], promedio_proyectado[0, 1], color='red', marker='*', s=200, label='Promedio de Edades')
            # Líneas a centroides por nivel educativo
            for nivel in niveles_unicos:
                indices = [i for i, niv in enumerate(niveles_educativos) if niv == nivel]
                if indices:
                    puntos_nivel = componentes_principales[indices]
                    centroide_nivel = np.mean(puntos_nivel, axis=0)
                    for idx in indices:
                        plt.plot([componentes_principales[idx, 0], centroide_nivel[0]], 
                                 [componentes_principales[idx, 1], centroide_nivel[1]], 
                                 'k--', alpha=0.3)
                    plt.scatter(centroide_nivel[0], centroide_nivel[1], marker='D', s=100, label=f'Centroide {nivel}')
            plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} Var.)')
            plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} Var.)')
            plt.title('Proyección 2D de Estudiantes (PCA)')
        
        plt.grid(True)
        
        # Agregar barra de colores para las edades
        plt.colorbar(scatter, label='Edad (años)')
        
        # Anotaciones de nombres (si hay 20 o menos estudiantes)
        if len(estudiantes) <= 20:
            for i, nombre in enumerate(nombres):
                etiqueta = f"{nombre} ({edades[i]} años)"
                if n_componentes == 1:
                    plt.annotate(etiqueta, (componentes_principales[i, 0], 0), fontsize=8)
                else:
                    plt.annotate(etiqueta, (componentes_principales[i, 0], componentes_principales[i, 1]), fontsize=8)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error al aplicar PCA: {str(e)}")

def mostrar_estudiantes(estudiantes):
    if not estudiantes:
        print("\nNo hay estudiantes registrados.")
    else:
        print("\n--- Lista de estudiantes registrados ---")
        for i, est in enumerate(estudiantes, 1):
            print(f"{i}. {est}")

def buscar_por_nivel_educativo(estudiantes):
    print("\n--- Buscar estudiantes por nivel educativo ---")
    niveles = [
        "Kinder",
        "Jardín de Infantes",
        "Primaria",
        "Secundaria",
        "Universidad o Instituto",
        "Otros estudios universitarios",
        "No aplica a ningún nivel educativo"
    ]
    for i, nivel in enumerate(niveles, 1):
        print(f"{i}. {nivel}")
    while True:
        opcion = input("Ingrese el número correspondiente al nivel: ").strip()
        if opcion.isdigit() and 1 <= int(opcion) <= len(niveles):
            nivel_seleccionado = niveles[int(opcion) - 1]
            break
        else:
            print("Opción inválida. Intente nuevamente.")
    encontrados = [e for e in estudiantes if e.nivel_educativo == nivel_seleccionado]
    if encontrados:
        print(f"\nEstudiantes en el nivel '{nivel_seleccionado}':")
        for est in encontrados:
            print(est)
    else:
        print(f"\nNo se encontraron estudiantes en el nivel '{nivel_seleccionado}'.")

def grafico_edad_vs_nivel(estudiantes):
    if not estudiantes:
        print("No hay estudiantes para graficar.")
        return
    edades = [e.edad for e in estudiantes]
    niveles_codificados = [e.obtener_caracteristicas()[1] for e in estudiantes]
    plt.figure()
    plt.scatter(edades, niveles_codificados, c=edades, cmap='viridis')
    plt.xlabel("Edad")
    plt.ylabel("Nivel Educativo (codificado)")
    plt.title("Edad vs Nivel Educativo")
    plt.colorbar(label="Edad (años)")
    plt.show()

def menu():
    estudiantes = []
    while True:
        print("\n SISTEMA DE CONSULTA Y ANÁLISIS DE ESTUDIANTES ")
        print("\n MENÚ PRINCIPAL ")
        print("1. Ingresar nuevo estudiante")
        print("2. Mostrar estudiantes registrados")
        print("3. Buscar por rango de edad")
        print("4. Buscar por nivel educativo")
        print("5. Aplicar Análisis de Componentes Principales (PCA)")
        print("6. Salir")
        opcion = input("Seleccione una opción: ").strip()
        if opcion == "1":
            estudiantes.append(ingresar_estudiante())
            print("\nEstudiante registrado exitosamente")
        elif opcion == "2":
            mostrar_estudiantes(estudiantes)
        elif opcion == "3":
            buscar_por_edad(estudiantes)
        elif opcion == "4":
            buscar_por_nivel_educativo(estudiantes)
        elif opcion == "5":
            aplicar_pca(estudiantes)
        elif opcion == "6":
            print("\nSaliendo del programa...VUELVA PRONTO! ")
            break
        else:
            print("\nOpción no válida. Intente nuevamente.") 

if __name__ == "__main__":
    print("SISTEMA DE CONSULTA Y ANÁLISIS DE EDADES DE ESTUDIANTES")
    menu()