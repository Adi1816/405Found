import tkinter as tk
import subprocess

# Define the function to execute a Python script
def execute_script(script_name):
    try:
        subprocess.run(["python", script_name])
    except Exception as e:
        print(f"Error executing {script_name}: {e}")

# Create the main window
root = tk.Tk()
root.title("Python Script Executor")

# Define the function to execute p1.py
def execute_p1():
    execute_script("facerecognize.py")

# Define the function to execute p2.py
def execute_p2():
    execute_script("facedetection.py")

# Define the function to execute p3.py
def execute_p3():
    execute_script("csvv.py")


# Create buttons to execute the scripts
button_p1 = tk.Button(root, text="Load a new Face", command=execute_p1)
button_p2 = tk.Button(root, text="Display Your Face/Mark Your Attendance", command=execute_p2)
button_p3 = tk.Button(root, text="Get the Attendance File", command=execute_p3)

# Pack the buttons into the main window
button_p1.pack()
button_p2.pack()
button_p3.pack()

# Start the GUI main loop
root.mainloop()
