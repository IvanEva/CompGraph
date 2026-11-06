#version 450

layout(set = 0, binding = 0, std140) uniform SceneData {
	mat4 view_proj;
	vec4 camera_position;
	vec4 ambient;
	vec4 dir_direction;
	vec4 dir_color;
	ivec4 counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelData {
	mat4 model;
	mat4 normal_matrix;
	vec4 albedo;
	vec4 specular;
} object;

// Массивы источников лежат в storage-буферах: их длина не зашита в шейдер,
// реальное число лежит в scene.counts.
struct PointLight {
	vec4 position; // xyz
	vec4 color;    // rgb * интенсивность
};

struct SpotLight {
	vec4 position;  // xyz
	vec4 direction; // xyz -- единичный вектор, куда направлен конус
	vec4 color;     // rgb * интенсивность
	vec4 cone;      // x -- cos внутреннего угла, y -- cos внешнего угла
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
	PointLight lights[];
} point_lights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
	SpotLight lights[];
} spot_lights;

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

// Модель Блинна-Фонга для одного источника.
//   to_light -- единичный вектор от точки к источнику
//   radiance -- цвет света, дошедшего до точки (с учётом затухания)
vec3 blinnPhong(vec3 normal, vec3 to_view, vec3 to_light, vec3 radiance) {
	float diffuse = max(dot(normal, to_light), 0.0);

	// Блик считается по "половинному вектору" между направлениями на свет и на камеру.
	vec3 half_vector = normalize(to_light + to_view);
	float specular = 0.0;
	if (diffuse > 0.0) {
		specular = pow(max(dot(normal, half_vector), 0.0), object.specular.a);
	}

	return radiance * (object.albedo.rgb * diffuse + object.specular.rgb * specular);
}

void main() {
	vec3 normal = normalize(in_world_normal);
	vec3 to_view = normalize(scene.camera_position.xyz - in_world_position);

	// Фоновый свет: одинаково освещает всё.
	vec3 color = scene.ambient.rgb * object.albedo.rgb;

	// Направленный источник (как Солнце): направление одно для всех точек, затухания нет.
	color += blinnPhong(normal, to_view, normalize(-scene.dir_direction.xyz), scene.dir_color.rgb);

	// Точечные источники: свет убывает как 1/d^2 (закон обратных квадратов).
	for (int i = 0; i < scene.counts.x; ++i) {
		vec3 delta = point_lights.lights[i].position.xyz - in_world_position;
		float dist2 = max(dot(delta, delta), 0.0001);
		float attenuation = 1.0 / dist2;
		color += blinnPhong(normal, to_view, delta * inversesqrt(dist2),
		                    point_lights.lights[i].color.rgb * attenuation);
	}

	// Прожекторы: то же затухание плюс конус. Внутри внутреннего угла свет полный,
	// между внутренним и внешним плавно спадает до нуля, снаружи света нет.
	for (int i = 0; i < scene.counts.y; ++i) {
		vec3 delta = spot_lights.lights[i].position.xyz - in_world_position;
		float dist2 = max(dot(delta, delta), 0.0001);
		vec3 to_light = delta * inversesqrt(dist2);

		// cos угла между осью конуса и направлением от источника к точке.
		float cos_angle = dot(-to_light, spot_lights.lights[i].direction.xyz);
		float cone = smoothstep(spot_lights.lights[i].cone.y, spot_lights.lights[i].cone.x, cos_angle);

		color += blinnPhong(normal, to_view, to_light,
		                    spot_lights.lights[i].color.rgb * (cone / dist2));
	}

	out_color = vec4(color, 1.0);
}
