#version 450

// Набор 0 -- данные сцены (одинаковые для всех объектов).
layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	ivec4 counts;
} scene;

// Набор 1 -- свойства модели. Это ДИНАМИЧЕСКИЙ uniform-буфер: в нём лежит массив таких структур
// (по одной на объект), а какую читать -- задаёт сдвиг в vkCmdBindDescriptorSets.
layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;
	vec4 tint;      // rgb -- множитель цвета текстуры
	vec4 specular;  // rgb -- цвет блика, a -- shininess
	vec4 params;    // x -- сила свечения (emissive), yz -- масштаб текстурных координат
} object;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_uv;

void main() {
	vec4 world = object.model * vec4(in_position, 1.0);
	out_world_position = world.xyz;
	out_world_normal = mat3(object.normal_matrix) * in_normal;

	// Масштаб > 1 повторяет текстуру несколько раз по поверхности (сэмплер в режиме REPEAT).
	out_uv = in_uv * object.params.yz;

	gl_Position = scene.view_proj * world;
}
