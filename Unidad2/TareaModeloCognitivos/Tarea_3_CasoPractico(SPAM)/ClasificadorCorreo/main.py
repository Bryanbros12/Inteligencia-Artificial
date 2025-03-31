import os

def menu():
    print("\n=== Clasificador de Correos ===")
    print("1. Entrenar el modelo")
    print("2. Clasificar un correo")
    print("3. Interfaz gráfica")
    print("4. Salir")

while True:
    menu()
    opcion = input("Seleccione una opción: ")

    if opcion == "1":
        os.system("python entrenar.py")
    elif opcion == "2":
        os.system("python clasificar.py")
    elif opcion == "3":
        os.system("python gui.py")
    elif opcion == "4":
        print("Saliendo...")
        break
    else:
        print("Opción no válida. Intente de nuevo.")
