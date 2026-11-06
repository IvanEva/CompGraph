#version 450

// Данные, которые CPU присылает перед каждым вызовом отрисовки (push constants).
// Раскладка должна совпадать со структурой PushConstants в application.cpp.
layout(push_constant) uniform PushConstants {
	mat4 mvp;   // Projection * View * Model
	vec4 tint;  // цвет, выбранный в интерфейсе
} pc;

// Атрибуты вершины (см. описание вершинного ввода в createPipeline()).
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

// То, что уйдёт во фрагментный шейдер (интерполируется по треугольнику).
layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_local_position;

void main() {
	gl_Position = pc.mvp * vec4(in_position, 1.0);

	// Цвет вершины умножается на выбранный в интерфейсе цвет.
	out_color = in_color * pc.tint.rgb;

	// Положение в локальных координатах фигуры -- нужно для затенения граней.
	out_local_position = in_position;
}
