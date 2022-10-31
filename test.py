from tkinter import *

root = Tk()

items = {'python': 1, 'perl': 2, 'C': 3, 'R': 4}
vars = []
for i in range(len(items)):
    vars.append(IntVar())
print(vars)
for var in vars:
    print(var.get())
for key, value in items.items():
    Checkbutton(root, text=key, onvalue=value, variable=vars[list(items.keys()).index(key)]).grid(sticky=W)


def show():
    for new_var in vars:
        print(new_var.get())


Button(root, text='Show', command=show).grid(sticky=W)

root.mainloop()
