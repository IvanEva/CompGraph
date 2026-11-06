#version 450

// Набор 0 -- данные сцены (одинаковые для всех объектов).
// Раскладка std140, должна совпадать со структурой SceneData в application.cpp.
layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_proj;       // Projection * View
	vec4 camera_position; // xyz -- положение камеры в мире
	vec4 ambient;         // rgb -- цвет фонового света (уже умноженный на интенсивность)
	vec4 dir_direction;   // xyz -- куда светит направленный источник
	vec4 dir_color;       // rgb -- его цвет * интенсивность
	ivec4 counts;         // x -- число точечных, y -- число прожекторных источников
} scene;

// Набор 1 -- данные модели (свои у каждого объекта).
layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;   // transpose(inverse(model)); в mat3-части лежит то, что нужно для нормалей
	vec4 albedo;          // rgb -- цвет материала (рассеянный)
	vec4 specular;        // rgb -- цвет блика, a -- shininess
} object;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;

void main() {
	vec4 world = object.model * vec4(in_position, 1.0);
	out_world_position = world.xyz;

	// Нормали нельзя умножать на model: при неравномерном масштабе они перестанут
	// быть перпендикулярны поверхности. Для них используется нормальная матрица.
	out_world_normal = mat3(object.normal_matrix) * in_normal;

	gl_Position = scene.view_proj * world;
}
