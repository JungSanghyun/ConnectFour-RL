# -*- coding: utf-8 -*-
import tkinter as tk


class ConnectFourViewer(tk.Tk):
    def __init__(self, width, height, block_size=60):
        super(ConnectFourViewer, self).__init__()
        self.block_size = block_size
        self.w = width
        self.h = height

        self.title('ConnectFour')
        self.geometry(f'{self.block_size * self.w}x{self.block_size * self.h}')

        self.canvas = tk.Canvas(self, width=block_size * self.w, height=block_size * self.h)
        self.canvas.pack()
        self._draw_background()

    def _get_oval_position(self, i, j):
        center_x = j * self.block_size + self.block_size // 2
        center_y = self.block_size * self.h - (i + 1) * self.block_size + self.block_size // 2
        radius = self.block_size * 2 // 5
        return center_x, center_y, radius

    def _draw_background(self):
        self.canvas.configure(bg='blue')
        self.ovals = [[None for _ in range(self.w)] for _ in range(self.h)]
        for i in range(self.h):
            for j in range(self.w):
                x, y, r = self._get_oval_position(i, j)
                self.ovals[i][j] = self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')

    def render(self, state):
        for i in range(self.h):
            for j in range(self.w):
                if state[i][j] == 1:
                    self.canvas.itemconfig(self.ovals[i][j], fill="yellow")
                elif state[i][j] == 2:
                    self.canvas.itemconfig(self.ovals[i][j], fill="red")
        self.update()

    def reset(self):
        self.canvas.delete("all")
        self._draw_background()

    def close(self):
        self.destroy()
