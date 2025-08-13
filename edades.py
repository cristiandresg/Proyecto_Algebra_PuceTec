import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class Estudiante:
    def __init__(self, nombre, segundo_nombre, apellido_paterno, apellido_materno, fecha_nacimiento):
        self.nombre = nombre
        self.segundo_nombre = segundo_nombre
        self.apellido_paterno = apellido_paterno
        self.apellido_materno = apellido_materno
        self.fecha_nacimiento = fecha_nacimiento
        self.edad = self.calcular_edad()
        self.nivel_educativo = self.determinar_nivel()

    def calcular_edad(self):
        hoy = datetime.date.today()
        edad = hoy.year - self.fecha_nacimiento.year
        if (hoy.month, hoy.day) < (self.fecha_nacimiento.month, self.fecha_nacimiento.day):
            edad -= 1
        return edad

    def determinar_nivel(self):
        rangos_edad = [
            (3, 4, "Kinder"),
            (5, 5, "Jardín de Infantes"),
            (6, 12, "Primaria"),
            (13, 17, "Secundaria"),
            (18, 25, "Universidad o Instituto"),
            (27, 35, "Otros estudios universitarios")
        ]
        for edad_min, edad_max, nivel in rangos_edad:
            if edad_min <= self.edad <= edad_max:
                return nivel
        return "No aplica a ningún nivel educativo"


    def __str__(self):
        return (f"Nombre: {self.nombre} {self.segundo_nombre} {self.apellido_paterno} {self.apellido_materno}\n"
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

def ingresar_estudiante():
    print("\nIngrese los datos del estudiante:")
    nombre = input("Primer nombre: ").strip().title()
    segundo_nombre = input("Segundo nombre: ").strip().title()
    apellido_paterno = input("Apellido paterno: ").strip().title()
    apellido_materno = input("Apellido materno: ").strip().title()

    while True:
        try:
            fecha_str = input("Fecha de nacimiento (DD/MM/AAAA): ")
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
        except ValueError as e:
            if "invalid literal" in str(e):
                print("Formato de fecha incorrecto. Use DD/MM/AAAA con números válidos.")
            else:
                print("Fecha inválida (día o mes no válido). Intente nuevamente.")

    return Estudiante(nombre, segundo_nombre, apellido_paterno, apellido_materno, fecha_nacimiento)

def mostrar_estudiantes(estudiantes):
    if not estudiantes:
        print("\nNo hay estudiantes registrados.")
        return

    print("\nLista de Estudiantes:")
    for i, estudiante in enumerate(estudiantes, 1):
        print(f"Estudiante #{i}")
        print(estudiante)

def buscar_por_edad(estudiantes):
    try:
        edad = int(input("\nIngrese la edad a buscar: "))
        encontrados = [e for e in estudiantes if e.edad == edad]
        if not encontrados:
            print(f"No se encontraron estudiantes con {edad} años.")
        else:
            print(f"\nEstudiantes con {edad} años:")
            for estudiante in encontrados:
                print(estudiante)
    except ValueError:
        print("Edad debe ser un número entero.")

def buscar_por_inicial(estudiantes):
    inicial = input("\nIngrese la inicial del apellido paterno: ").strip().upper()
    if len(inicial) != 1 or not inicial.isalpha():
        print("Debe ingresar una sola letra.")
        return
    encontrados = [e for e in estudiantes if e.apellido_paterno.startswith(inicial)]
    if not encontrados:
        print(f"No se encontraron estudiantes con apellido paterno que comience con '{inicial}'.")
    else:
        print(f"\nEstudiantes con apellido paterno que comienza con '{inicial}':")
        for estudiante in encontrados:
            print(estudiante)

def buscar_por_fecha(estudiantes):
    try:
        fecha_str = input("\nIngrese la fecha de nacimiento a buscar (DD/MM/AAAA): ")
        dia, mes, anio = map(int, fecha_str.split('/'))
        fecha_busqueda = datetime.date(anio, mes, dia)
        encontrados = [e for e in estudiantes if e.fecha_nacimiento == fecha_busqueda]
        if not encontrados:
            print(f"No se encontraron estudiantes nacidos el {fecha_busqueda.strftime('%d/%m/%Y')}.")
        else:
            print(f"\nEstudiantes nacidos el {fecha_busqueda.strftime('%d/%m/%Y')}:")
            for estudiante in encontrados:
                print(estudiante)
    except ValueError:
        print("Formato de fecha incorrecto. Use DD/MM/AAAA.")

def aplicar_pca(estudiantes):
    if len(estudiantes) < 3:
        print("\nSe necesitan al menos 3 estudiantes para un análisis PCA significativo.")
        return
    
    datos = np.array([e.obtener_caracteristicas() for e in estudiantes])
    nombres = [f"{e.nombre} {e.apellido_paterno}" for e in estudiantes]
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
                if n_componentes == 1:
                    plt.annotate(nombre, (componentes_principales[i, 0], 0), fontsize=8)
                else:
                    plt.annotate(nombre, (componentes_principales[i, 0], componentes_principales[i, 1]), fontsize=8)
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error al aplicar PCA: {str(e)}")

def menu():
    estudiantes = []
    while True:
        print("\n---- SISTEMA DE CONSULTA Y ANÁLISIS DE ESTUDIANTES ----")
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Ingresar nuevo estudiante")
        print("2. Mostrar todos los estudiantes")
        print("3. Buscar por edad")
        print("4. Buscar por inicial del apellido paterno")
        print("5. Buscar por fecha de nacimiento")
        print("6. Aplicar Análisis de Componentes Principales (PCA)")
        print("7. Salir")
        opcion = input("Seleccione una opción: ").strip()
        if opcion == "1":
            estudiantes.append(ingresar_estudiante())
            print("\nEstudiante registrado exitosamente!")
        elif opcion == "2":
            mostrar_estudiantes(estudiantes)
        elif opcion == "3":
            buscar_por_edad(estudiantes)
        elif opcion == "4":
            buscar_por_inicial(estudiantes)
        elif opcion == "5":
            buscar_por_fecha(estudiantes)
        elif opcion == "6":
            aplicar_pca(estudiantes)
        elif opcion == "7":
            print("\nSaliendo del programa...")
            break
        else:
            print("\nOpción no válida. Intente nuevamente.")

if __name__ == "__main__":
    print("SISTEMA DE CONSULTA Y ANÁLISIS DE EDADES DE ESTUDIANTES")
    menu()