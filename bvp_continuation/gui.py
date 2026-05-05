"""gui.py — Tkinter GUI for the BVP continuation solver."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .examples import get_example_names, load_example
from .export_utils import export_all_steps_csv, export_final_solution_csv
from .io_utils import load_problem, save_problem
from .problem import BVPProblem, ContinuationSettings
from .solver import ContinuationResult, solve_by_continuation


AUTHOR_INFO_TEXT = """Студент факультета ВМК МГУ
Орлов Герман Евгеньевич
Группа: 313
Дата рождения: 08.10.2005
Email: o.german.e@yandex.ru

Проект выполнен в рамках технологического практикума по Python."""

PROJECT_INFO_TEXT = """Данная программа предназначена для численного решения краевых задач методом продолжения по параметру.

Метод продолжения по параметру позволяет решать сложную задачу постепенно. Сначала рассматривается более простая задача при начальном значении параметра, затем параметр постепенно изменяется до конечного значения. Решение, найденное на предыдущем шаге, используется как начальное приближение для следующего шага.

Программа позволяет:
— задавать системы обыкновенных дифференциальных уравнений первого порядка;
— вводить граничные условия;
— задавать начальное приближение;
— настраивать параметры метода продолжения;
— строить графики итогового и промежуточных решений;
— сохранять и загружать задачи в формате JSON;
— экспортировать результаты вычислений в CSV.

Проект является некоммерческим и учебным."""

HELP_TEXT = """1. Выберите встроенный пример из списка Example и нажмите Load, либо введите свою задачу вручную.

2. В поле Variables укажите переменные системы через запятую.

Пример:
y1, y2

3. В поле Equations вводятся только правые части уравнений.

Если задача имеет вид:
y1' = y2
y2' = -mu * sin(y1)

то в поле Equations нужно написать:
y2
-mu * sin(y1)

4. В поле Boundary conditions вводятся граничные условия в виде выражений, которые должны быть равны нулю.

Пример:
y1_a - 0
y1_b - 1

Здесь y1_a означает значение y1 на левом конце отрезка, а y1_b — значение y1 на правом конце отрезка.

5. В поле Initial guess задаётся начальное приближение.

Пример:
y1 = t
y2 = 1

6. В блоке Continuation settings задаются параметры метода:
parameter — имя параметра продолжения, обычно mu;
start — начальное значение параметра;
end — конечное значение параметра;
steps — количество шагов продолжения;
mesh — начальное количество точек сетки;
tol — допустимая погрешность вычислений.

7. Кнопка Validate проверяет корректность введённой задачи.

8. Кнопка Solve запускает численное решение задачи.

9. В блоке Plot settings можно выбрать режим отображения графика:
Final solution — итоговое решение;
Continuation path — промежуточные решения при разных значениях параметра;
Initial guess vs final — сравнение начального приближения и итогового решения.

10. Кнопка Save JSON сохраняет текущую задачу в файл.

11. Кнопка Load JSON загружает задачу из файла.

12. Кнопка Export final сохраняет итоговое решение в CSV.

13. Кнопка Export all сохраняет все промежуточные решения по шагам параметра в CSV."""


def _split_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_variables(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_initial_guess(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _split_lines(text):
        if "=" not in line:
            raise ValueError(
                f"Initial guess line must have the format variable = expression: {line}"
            )
        name, expression = line.split("=", 1)
        result[name.strip()] = expression.strip()
    return result


def _initial_guess_to_text(initial_guess: dict[str, str]) -> str:
    return "\n".join(f"{name} = {expression}" for name, expression in initial_guess.items())


def _center_window(window: tk.Toplevel | tk.Tk, width: int, height: int) -> None:
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = max((screen_width - width) // 2, 0)
    y = max((screen_height - height) // 2, 0)
    window.geometry(f"{width}x{height}+{x}+{y}")


def _show_text_window(parent: tk.Misc, title: str, text: str, width: int = 720, height: int = 520) -> None:
    window = tk.Toplevel(parent)
    window.title(title)
    _center_window(window, width, height)
    window.transient(parent)
    window.grab_set()

    frame = ttk.Frame(window, padding=12)
    frame.pack(fill="both", expand=True)
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    text_box = tk.Text(frame, wrap="word", height=20)
    text_box.grid(row=0, column=0, sticky="nsew")
    text_box.insert("1.0", text)
    text_box.configure(state="disabled")

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_box.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    text_box.configure(yscrollcommand=scrollbar.set)

    ttk.Button(frame, text="Закрыть", command=window.destroy).grid(row=1, column=0, columnspan=2, sticky="e", pady=(10, 0))


def _show_author_window(parent: tk.Misc) -> None:
    window = tk.Toplevel(parent)
    window.title("Об авторе")
    _center_window(window, 620, 360)
    window.transient(parent)
    window.grab_set()

    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    photo_placeholder = ttk.LabelFrame(frame, text="Фотография", padding=12)
    photo_placeholder.grid(row=0, column=0, sticky="n", padx=(0, 16))
    placeholder = ttk.Label(photo_placeholder, text="Место\nдля фотографии\nавтора", anchor="center", justify="center", width=18)
    placeholder.pack(ipadx=14, ipady=42)

    text_box = tk.Text(frame, wrap="word", height=12, width=46)
    text_box.grid(row=0, column=1, sticky="nsew")
    text_box.insert("1.0", AUTHOR_INFO_TEXT)
    text_box.configure(state="disabled")

    ttk.Button(frame, text="Закрыть", command=window.destroy).grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))


def _show_project_window(parent: tk.Misc) -> None:
    _show_text_window(parent, "О проекте", PROJECT_INFO_TEXT)


def _show_help_window(parent: tk.Misc) -> None:
    _show_text_window(parent, "Справка", HELP_TEXT, width=760, height=640)


class WelcomeWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Parameter Continuation Method for BVP")
        self.should_start_app = False
        _center_window(self, 560, 390)
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=24)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)

        title = ttk.Label(frame, text="Метод продолжения по параметру", font=("Arial", 18, "bold"), anchor="center")
        title.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        subtitle = ttk.Label(
            frame,
            text="Программа для численного решения краевых задач",
            anchor="center",
            justify="center",
        )
        subtitle.grid(row=1, column=0, sticky="ew", pady=(0, 22))

        ttk.Button(frame, text="Запустить программу", command=self._start_program).grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Button(frame, text="Об авторе", command=lambda: _show_author_window(self)).grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Button(frame, text="О проекте", command=lambda: _show_project_window(self)).grid(row=4, column=0, sticky="ew", pady=5)
        ttk.Button(frame, text="Справка", command=lambda: _show_help_window(self)).grid(row=5, column=0, sticky="ew", pady=5)
        ttk.Button(frame, text="Выход", command=self.destroy).grid(row=6, column=0, sticky="ew", pady=(16, 0))

    def _start_program(self) -> None:
        self.should_start_app = True
        self.destroy()


class BVPContinuationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Parameter Continuation Method for BVP")
        self.geometry("1320x860")
        self.current_result: ContinuationResult | None = None
        self.solver_thread: threading.Thread | None = None
        self.info_menu: tk.Menu | None = None
        self._build_ui()
        self._load_example_into_form("linear")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(self, padding=(10, 8, 10, 0))
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_bar.columnconfigure(0, weight=1)
        ttk.Label(top_bar, text="Parameter Continuation Method for BVP", font=("Arial", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(top_bar, text="⚙", width=3, command=self._show_info_menu).grid(row=0, column=1, sticky="e")

        left = ttk.Frame(self, padding=10)
        left.grid(row=1, column=0, sticky="nsew")
        left.columnconfigure(1, weight=1)

        right = ttk.Frame(self, padding=10)
        right.grid(row=1, column=1, sticky="nsew")
        right.rowconfigure(0, weight=4)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_input_panel(left)
        self._build_output_panel(right)

    def _build_input_panel(self, left: ttk.Frame) -> None:
        row = 0
        ttk.Label(left, text="Example").grid(row=row, column=0, sticky="w")
        self.example_var = tk.StringVar(value="linear")
        self.example_box = ttk.Combobox(left, textvariable=self.example_var, values=get_example_names(), state="readonly", width=34)
        self.example_box.grid(row=row, column=1, sticky="ew", padx=5, pady=3)
        ttk.Button(left, text="Load", command=self._on_load_example).grid(row=row, column=2, sticky="ew", pady=3)
        row += 1

        ttk.Label(left, text="Name").grid(row=row, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.name_var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Description").grid(row=row, column=0, sticky="nw")
        self.description_text = tk.Text(left, height=3, width=54)
        self.description_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="a").grid(row=row, column=0, sticky="w")
        self.a_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.a_var, width=12).grid(row=row, column=1, sticky="w", padx=5, pady=3)
        ttk.Label(left, text="b").grid(row=row, column=1, sticky="e")
        self.b_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.b_var, width=12).grid(row=row, column=2, sticky="w", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Variables").grid(row=row, column=0, sticky="w")
        self.variables_var = tk.StringVar()
        ttk.Entry(left, textvariable=self.variables_var).grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Equations").grid(row=row, column=0, sticky="nw")
        self.equations_text = tk.Text(left, height=5, width=54)
        self.equations_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Boundary conditions").grid(row=row, column=0, sticky="nw")
        self.boundary_text = tk.Text(left, height=5, width=54)
        self.boundary_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Initial guess").grid(row=row, column=0, sticky="nw")
        self.guess_text = tk.Text(left, height=5, width=54)
        self.guess_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        ttk.Label(left, text="Parameters JSON").grid(row=row, column=0, sticky="nw")
        self.parameters_text = tk.Text(left, height=3, width=54)
        self.parameters_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)
        row += 1

        continuation_frame = ttk.LabelFrame(left, text="Continuation", padding=8)
        continuation_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        for idx in range(6):
            continuation_frame.columnconfigure(idx, weight=1)

        self.parameter_var = tk.StringVar(value="mu")
        self.mu_start_var = tk.StringVar(value="0")
        self.mu_end_var = tk.StringVar(value="1")
        self.steps_var = tk.StringVar(value="20")
        self.mesh_points_var = tk.StringVar(value="80")
        self.tolerance_var = tk.StringVar(value="1e-4")

        labels_and_vars = [("parameter", self.parameter_var), ("start", self.mu_start_var), ("end", self.mu_end_var), ("steps", self.steps_var), ("mesh", self.mesh_points_var), ("tol", self.tolerance_var)]
        for column, (label, variable) in enumerate(labels_and_vars):
            ttk.Label(continuation_frame, text=label).grid(row=0, column=column)
            ttk.Entry(continuation_frame, textvariable=variable, width=9).grid(row=1, column=column, padx=3)
        row += 1

        button_frame = ttk.Frame(left)
        button_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        for idx in range(6):
            button_frame.columnconfigure(idx, weight=1)
        ttk.Button(button_frame, text="Validate", command=self._on_validate).grid(row=0, column=0, sticky="ew", padx=3)
        ttk.Button(button_frame, text="Solve", command=self._on_solve).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(button_frame, text="Save JSON", command=self._on_save_json).grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(button_frame, text="Load JSON", command=self._on_load_json).grid(row=0, column=3, sticky="ew", padx=3)
        ttk.Button(button_frame, text="Export final", command=self._on_export_final).grid(row=0, column=4, sticky="ew", padx=3)
        ttk.Button(button_frame, text="Export all", command=self._on_export_all).grid(row=0, column=5, sticky="ew", padx=3)
        row += 1

        plot_frame = ttk.LabelFrame(left, text="Plot settings", padding=8)
        plot_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=6)
        plot_frame.columnconfigure(1, weight=1)
        self.plot_mode_var = tk.StringVar(value="Final solution")
        self.variable_var = tk.StringVar(value="")
        ttk.Label(plot_frame, text="Mode").grid(row=0, column=0, sticky="w")
        self.plot_mode_box = ttk.Combobox(plot_frame, textvariable=self.plot_mode_var, values=["Final solution", "Continuation path", "Initial guess vs final"], state="readonly")
        self.plot_mode_box.grid(row=0, column=1, sticky="ew", padx=5)
        self.plot_mode_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw_current_result())
        ttk.Label(plot_frame, text="Variable").grid(row=1, column=0, sticky="w")
        self.variable_box = ttk.Combobox(plot_frame, textvariable=self.variable_var, values=[], state="readonly")
        self.variable_box.grid(row=1, column=1, sticky="ew", padx=5, pady=3)
        self.variable_box.bind("<<ComboboxSelected>>", lambda _event: self._redraw_current_result())
        row += 1

        ttk.Label(left, text="Log").grid(row=row, column=0, sticky="nw")
        self.log_text = tk.Text(left, height=7, width=54)
        self.log_text.grid(row=row, column=1, columnspan=2, sticky="ew", padx=5, pady=3)

    def _build_output_panel(self, right: ttk.Frame) -> None:
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Solution")
        self.axis.set_xlabel("t")
        self.axis.grid(True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        table_frame = ttk.LabelFrame(right, text="Continuation steps", padding=6)
        table_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        columns = ("mu", "success", "iterations", "nodes", "max_bc_residual")
        self.steps_table = ttk.Treeview(table_frame, columns=columns, show="headings", height=7)
        headings = {"mu": "mu", "success": "success", "iterations": "iterations", "nodes": "nodes", "max_bc_residual": "max boundary residual"}
        widths = {"mu": 90, "success": 80, "iterations": 90, "nodes": 80, "max_bc_residual": 160}
        for column in columns:
            self.steps_table.heading(column, text=headings[column])
            self.steps_table.column(column, width=widths[column], anchor="center")
        self.steps_table.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.steps_table.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.steps_table.configure(yscrollcommand=scrollbar.set)

    def _show_info_menu(self) -> None:
        if self.info_menu is None:
            self.info_menu = tk.Menu(self, tearoff=False)
            self.info_menu.add_command(label="Об авторе", command=lambda: _show_author_window(self))
            self.info_menu.add_command(label="О проекте", command=lambda: _show_project_window(self))
            self.info_menu.add_command(label="Справка", command=lambda: _show_help_window(self))
        self.info_menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())

    def _log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _on_load_example(self) -> None:
        self._load_example_into_form(self.example_var.get())

    def _load_example_into_form(self, name: str) -> None:
        problem = load_example(name)
        self._problem_to_form(problem)
        self._log(f"Loaded example: {problem.name}")

    def _problem_to_form(self, problem: BVPProblem) -> None:
        self.name_var.set(problem.name)
        self.description_text.delete("1.0", "end")
        self.description_text.insert("1.0", problem.description)
        self.a_var.set(str(problem.a))
        self.b_var.set(str(problem.b))
        self.variables_var.set(", ".join(problem.variables))
        self.equations_text.delete("1.0", "end")
        self.equations_text.insert("1.0", "\n".join(problem.equations))
        self.boundary_text.delete("1.0", "end")
        self.boundary_text.insert("1.0", "\n".join(problem.boundary_conditions))
        self.guess_text.delete("1.0", "end")
        self.guess_text.insert("1.0", _initial_guess_to_text(problem.initial_guess))
        self.parameters_text.delete("1.0", "end")
        self.parameters_text.insert("1.0", json.dumps(problem.parameters, indent=4))
        self.parameter_var.set(problem.continuation.parameter)
        self.mu_start_var.set(str(problem.continuation.start))
        self.mu_end_var.set(str(problem.continuation.end))
        self.steps_var.set(str(problem.continuation.steps))
        self.mesh_points_var.set(str(problem.continuation.mesh_points))
        self.tolerance_var.set(str(problem.continuation.tolerance))
        self._update_variable_selector(problem.variables)

    def _update_variable_selector(self, variables: list[str]) -> None:
        self.variable_box.configure(values=variables)
        self.variable_var.set(variables[0] if variables else "")

    def _form_to_problem(self) -> BVPProblem:
        parameters_text = self.parameters_text.get("1.0", "end").strip()
        parameters = json.loads(parameters_text) if parameters_text else {}
        problem = BVPProblem(
            name=self.name_var.get().strip() or "Untitled BVP",
            description=self.description_text.get("1.0", "end").strip(),
            a=float(self.a_var.get()),
            b=float(self.b_var.get()),
            variables=_parse_variables(self.variables_var.get()),
            equations=_split_lines(self.equations_text.get("1.0", "end")),
            boundary_conditions=_split_lines(self.boundary_text.get("1.0", "end")),
            parameters={str(k): float(v) for k, v in parameters.items()},
            initial_guess=_parse_initial_guess(self.guess_text.get("1.0", "end")),
            continuation=ContinuationSettings(
                parameter=self.parameter_var.get().strip() or "mu",
                start=float(self.mu_start_var.get()),
                end=float(self.mu_end_var.get()),
                steps=int(self.steps_var.get()),
                mesh_points=int(self.mesh_points_var.get()),
                tolerance=float(self.tolerance_var.get()),
            ),
        )
        problem.validate_basic_structure()
        self._update_variable_selector(problem.variables)
        return problem

    def _on_validate(self) -> None:
        try:
            problem = self._form_to_problem()
            from .parser import compile_problem
            compile_problem(problem)
        except Exception as exc:
            messagebox.showerror("Validation error", str(exc))
            self._log(f"Validation error: {exc}")
            return
        messagebox.showinfo("Validation", "Problem is valid.")
        self._log("Problem is valid.")

    def _on_solve(self) -> None:
        if self.solver_thread is not None and self.solver_thread.is_alive():
            messagebox.showinfo("Solver", "Solver is already running.")
            return
        try:
            problem = self._form_to_problem()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            self._log(f"Input error: {exc}")
            return
        self._log("Solving started...")
        def worker() -> None:
            try:
                result = solve_by_continuation(problem)
            except Exception as exc:
                self.after(0, self._on_solver_error, exc)
                return
            self.after(0, self._on_solver_success, result)
        self.solver_thread = threading.Thread(target=worker, daemon=True)
        self.solver_thread.start()

    def _on_solver_error(self, exc: Exception) -> None:
        messagebox.showerror("Solver error", str(exc))
        self._log(f"Solver error: {exc}")

    def _on_solver_success(self, result: ContinuationResult) -> None:
        self.current_result = result
        self._update_variable_selector(result.problem.variables)
        self._log("Solving finished.")
        final_residuals = result.final_solution.boundary_residuals
        self._log("Final boundary residuals: " + ", ".join(f"{value:.3e}" for value in final_residuals))
        self._fill_steps_table(result)
        self._plot_result(result)

    def _fill_steps_table(self, result: ContinuationResult) -> None:
        for item in self.steps_table.get_children():
            self.steps_table.delete(item)
        for step in result.steps:
            self.steps_table.insert("", "end", values=(f"{step.mu:.6g}", "yes" if step.success else "no", step.iterations, step.nodes, f"{step.max_boundary_residual:.3e}"))

    def _selected_variable_index(self, result: ContinuationResult) -> int:
        variable = self.variable_var.get()
        if variable in result.problem.variables:
            return result.problem.variables.index(variable)
        return 0

    def _redraw_current_result(self) -> None:
        if self.current_result is not None:
            self._plot_result(self.current_result)

    def _plot_result(self, result: ContinuationResult) -> None:
        mode = self.plot_mode_var.get()
        self.axis.clear()
        if mode == "Continuation path":
            self._plot_continuation_path(result)
        elif mode == "Initial guess vs final":
            self._plot_initial_vs_final(result)
        else:
            self._plot_final_solution(result)
        self.axis.grid(True)
        self.axis.legend()
        self.figure.tight_layout()
        self.canvas.draw()

    def _plot_final_solution(self, result: ContinuationResult) -> None:
        for idx, variable in enumerate(result.problem.variables):
            self.axis.plot(result.x, result.y[idx], label=variable)
        self.axis.set_title(f"Final solution, mu={result.final_mu:.6g}")
        self.axis.set_xlabel("t")
        self.axis.set_ylabel("solution")

    def _plot_continuation_path(self, result: ContinuationResult) -> None:
        variable_index = self._selected_variable_index(result)
        variable = result.problem.variables[variable_index]
        max_lines = 8
        total = len(result.solutions)
        indices = list(range(total)) if total <= max_lines else sorted(set(np.linspace(0, total - 1, max_lines, dtype=int).tolist()))
        for index in indices:
            solution = result.solutions[index]
            self.axis.plot(solution.x, solution.y[variable_index], label=f"mu={solution.mu:.3g}")
        self.axis.set_title(f"Continuation path for {variable}")
        self.axis.set_xlabel("t")
        self.axis.set_ylabel(variable)

    def _plot_initial_vs_final(self, result: ContinuationResult) -> None:
        variable_index = self._selected_variable_index(result)
        variable = result.problem.variables[variable_index]
        self.axis.plot(result.initial_x, result.initial_y[variable_index], linestyle="--", label=f"initial {variable}")
        self.axis.plot(result.x, result.y[variable_index], label=f"final {variable}")
        self.axis.set_title(f"Initial guess vs final solution for {variable}")
        self.axis.set_xlabel("t")
        self.axis.set_ylabel(variable)

    def _on_save_json(self) -> None:
        try:
            problem = self._form_to_problem()
        except Exception as exc:
            messagebox.showerror("Input error", str(exc))
            return
        path = filedialog.asksaveasfilename(title="Save problem", defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        save_problem(problem, path)
        self._log(f"Saved problem to {path}")

    def _on_load_json(self) -> None:
        path = filedialog.askopenfilename(title="Load problem", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            problem = load_problem(path)
            problem.validate_basic_structure()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))
            self._log(f"Load error: {exc}")
            return
        self._problem_to_form(problem)
        self._log(f"Loaded problem from {path}")

    def _on_export_final(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("Export", "Solve a problem first.")
            return
        path = filedialog.asksaveasfilename(title="Export final solution", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        export_final_solution_csv(self.current_result, path)
        self._log(f"Exported final solution to {path}")

    def _on_export_all(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("Export", "Solve a problem first.")
            return
        path = filedialog.asksaveasfilename(title="Export all continuation steps", defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        export_all_steps_csv(self.current_result, path)
        self._log(f"Exported all continuation steps to {path}")


def run_gui() -> None:
    welcome = WelcomeWindow()
    welcome.mainloop()
    if not welcome.should_start_app:
        return
    app = BVPContinuationApp()
    app.mainloop()
