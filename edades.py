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
        if 3 <= self.edad <= 4:
            return "Kinder"
        elif self.edad == 5:
            return "Jardín de Infantes"
        elif 6 <= self.edad <= 12:
            return "Primaria"
        elif 13 <= self.edad <= 17:
            return "Secundaria"
        elif 18 <= self.edad <= 25:
            return "Universidad o Instituto"
        elif 27 <= self.edad <= 35:
            return "Otros estudios universitarios"
        else:
            return "No aplica a ningún nivel educativo"

    def __str__(self):
        return ("Nombre: {self.nombre} {self.segundo_nombre} {self.apellido_paterno} {self.apellido_materno}\n"
                "Fecha de Nacimiento: {self.fecha_nacimiento.strftime('%d/%m/%Y')}\n"
                "Edad: {self.edad} años\n"
                "Nivel Educativo: {self.nivel_educativo}\n")

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
            print("Formato de fecha incorrecto. Use DD/MM/AAAA. Intente nuevamente.")

    return Estudiante(nombre, segundo_nombre, apellido_paterno, apellido_materno, fecha_nacimiento)

def mostrar_estudiantes(estudiantes):
    if not estudiantes:
        print("\nNo hay estudiantes registrados.")
        return

    print("\nLista de Estudiantes:")
    for i, estudiante in enumerate(estudiantes, 1):
        print("Estudiante #{i}")
        print(estudiante)

def buscar_por_edad(estudiantes):
    try:
        edad = int(input("\nIngrese la edad a buscar: "))
        encontrados = [e for e in estudiantes if e.edad == edad]

        if not encontrados:
            print("No se encontraron estudiantes con {edad} años.")
        else:
            print("\nEstudiantes con {edad} años:")
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
        print("No se encontraron estudiantes con apellido paterno que comience con '{inicial}'.")
    else:
        print("\nEstudiantes con apellido paterno que comienza con '{inicial}':")
        for estudiante in encontrados:
            print(estudiante)

def buscar_por_fecha(estudiantes):
    try:
        fecha_str = input("\nIngrese la fecha de nacimiento a buscar (DD/MM/AAAA): ")
        dia, mes, anio = map(int, fecha_str.split('/'))
        fecha_busqueda = datetime.date(anio, mes, dia)

        encontrados = [e for e in estudiantes if e.fecha_nacimiento == fecha_busqueda]

        if not encontrados:
            print("No se encontraron estudiantes nacidos el {fecha_busqueda.strftime('%d/%m/%Y')}.")
        else:
            print("\nEstudiantes nacidos el {fecha_busqueda.strftime('%d/%m/%Y')}:")
            for estudiante in encontrados:
                print(estudiante)
    except ValueError:
        print("Formato de fecha incorrecto. Use DD/MM/AAAA.")

def aplicar_pca(estudiantes):
    if len(estudiantes) < 3:
        print("\nSe necesitan al menos 3 estudiantes para un análisis PCA significativo.")
        return

    datos = np.array([e.obtener_caracteristicas() for e in estudiantes])
    nombres = ["{e.nombre} {e.apellido_paterno}" for e in estudiantes]
    caracteristicas = ['Edad', 'Nivel Educativo (codificado)']

    print("\n--- Aplicando Análisis de Componentes Principales (PCA) ---")
    print("Datos originales: {len(estudiantes)} estudiantes x {len(caracteristicas)} características.")

    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos)
    
    n_componentes = min(2, datos_escalados.shape[1], len(estudiantes)-1)
    pca = PCA(n_components=n_componentes)
    componentes_principales = pca.fit_transform(datos_escalados)

    print("\nDimensionalidad reducida a {n_componentes} componentes principales.")
    print("Varianza explicada por cada componente:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print("  PC{i+1}: {var:.2%} de la varianza total")

    plt.figure(figsize=(8, 6))
    
    if n_componentes == 1:
        plt.scatter(componentes_principales[:, 0], np.zeros(len(componentes_principales)), 
                   alpha=0.7)
        plt.xlabel('Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} Var.)')
        plt.title('Proyección 1D de Estudiantes (PCA)')
    else:
        plt.scatter(componentes_principales[:, 0], componentes_principales[:, 1], alpha=0.7)
        plt.xlabel('Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} Var.)')
        plt.ylabel('Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} Var.)')
        plt.title('Proyección 2D de Estudiantes (PCA)')
    
    plt.grid(True)
    
    if len(estudiantes) <= 20:
        for i, nombre in enumerate(nombres):
            if n_componentes == 1:
                plt.annotate(nombre, (componentes_principales[i, 0], 0), fontsize=8)
            else:
                plt.annotate(nombre, (componentes_principales[i, 0], componentes_principales[i, 1]), 
                           fontsize=8)

    plt.tight_layout()
    plt.show()

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