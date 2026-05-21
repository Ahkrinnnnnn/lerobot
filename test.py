# test_gui.py
from pynput import keyboard
import tkinter as tk

def on_press(key):
    print(f"Pressed: {key}")
    if hasattr(key, 'char') and key.char == 'q':
        root.quit()
        return False

def start_listener():
    print("Starting keyboard listener...")
    print("Press 'q' to quit")
    with keyboard.Listener(on_press=on_press, suppress=True) as listener:
        listener.join()

root = tk.Tk()
root.title("Keyboard Test")
root.geometry("300x100")

label = tk.Label(root, text="Click Start, then press keys.\nPress 'q' to exit.")
label.pack(pady=20)

btn = tk.Button(root, text="Start Listener", command=start_listener)
btn.pack()

root.mainloop()