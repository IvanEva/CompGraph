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
	vec4 tint;
	vec4 specular;
	vec4 params;
} object;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 cone;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLights {
	PointLight lights[];
} point_lights;

layout(set = 0, binding = 2, std430) readonly buffer SpotLights {
	SpotLight lights[];
} spot_lights;

// Набор 2 -- материал: три текстуры (каждая -- пара "изображение + сэмплер").
layout(set = 2, binding = 0) uniform sampler2D albedo_map;    // основной цвет
layout(set = 2, binding = 1) uniform sampler2D specular_map;  // где поверхность блестит
layout(set = 2, binding = 2) uniform sampler2D emissive_map;  // где поверхность светится сама

// Push constants: быстрые данные, которые меняются от объекта к объекту.
layout(push_constant) uniform PushConstants {
	uint flags;   // бит 0 -- включить нетривиальное сэмплирование
	float time;   // время в секундах
} pc;

const uint FLAG_ANIMATED_SAMPLING = 1u;

layout(location = 0) in vec3 in_world_position;
layout(location = 1) in vec3 in_world_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec4 out_color;

// Нетривиальное сэмплирование: одно и то же изображение читается ДВАЖДЫ с разными
// текстурными координатами, результаты смешиваются, а цвет модулируется по времени.
vec3 sampleAlbedoAnimated(vec2 uv) {
	float t = pc.time;

	// Слой A: текстура плывёт вдоль U.
	vec2 uv_a = uv + vec2(0.08 * t, 0.0);

	// Слой B: другой масштаб (x1.7), поворот на 35 градусов и обратное смещение вдоль V.
	float angle = radians(35.0);
	mat2 rotation = mat2(cos(angle), sin(angle), -sin(angle), cos(angle));
	vec2 uv_b = rotation * (uv * 1.7) - vec2(0.0, 0.05 * t);

	vec3 layer_a = texture(albedo_map, uv_a).rgb;
	vec3 layer_b = texture(albedo_map, uv_b).rgb;

	// Доля слоя B медленно колеблется между 0 и 1.
	vec3 blended = mix(layer_a, layer_b, 0.5 + 0.5 * sin(0.8 * t));

	// Модуляция цвета: каждый канал пульсирует со своим сдвигом фазы (120 градусов).
	vec3 pulse = 0.8 + 0.4 * vec3(sin(t), sin(t + 2.094), sin(t + 4.189));
	return blended * pulse;
}

vec3 blinnPhong(vec3 normal, vec3 to_view, vec3 to_light, vec3 radiance,
                vec3 albedo, vec3 specular_color) {
	float diffuse = max(dot(normal, to_light), 0.0);

	vec3 half_vector = normalize(to_light + to_view);
	float specular = 0.0;
	if (diffuse > 0.0) {
		specular = pow(max(dot(normal, half_vector), 0.0), object.specular.a);
	}

	return radiance * (albedo * diffuse + specular_color * specular);
}

void main() {
	// Если флаг выключен -- обычное чтение текстуры одной выборкой.
	// Условие зависит только от push constant (одинаково для всех пикселей) -- это безопасно
	// для texture(): GPU вычисляет производные по соседним пикселям квада.
	vec3 albedo;
	if ((pc.flags & FLAG_ANIMATED_SAMPLING) != 0u) {
		albedo = sampleAlbedoAnimated(in_uv);
	} else {
		albedo = texture(albedo_map, in_uv).rgb;
	}
	albedo *= object.tint.rgb;

	// Specular-текстура работает как маска блеска: белое -- блестит, чёрное -- матовое.
	vec3 specular_color = object.specular.rgb * texture(specular_map, in_uv).r;

	vec3 normal = normalize(in_world_normal);
	vec3 to_view = normalize(scene.camera_position.xyz - in_world_position);

	vec3 color = scene.ambient.rgb * albedo;

	color += blinnPhong(normal, to_view, normalize(-scene.dir_direction.xyz),
	                    scene.dir_color.rgb, albedo, specular_color);

	for (int i = 0; i < scene.counts.x; ++i) {
		vec3 delta = point_lights.lights[i].position.xyz - in_world_position;
		float dist2 = max(dot(delta, delta), 0.0001);
		color += blinnPhong(normal, to_view, delta * inversesqrt(dist2),
		                    point_lights.lights[i].color.rgb / dist2, albedo, specular_color);
	}

	for (int i = 0; i < scene.counts.y; ++i) {
		vec3 delta = spot_lights.lights[i].position.xyz - in_world_position;
		float dist2 = max(dot(delta, delta), 0.0001);
		vec3 to_light = delta * inversesqrt(dist2);

		float cos_angle = dot(-to_light, spot_lights.lights[i].direction.xyz);
		float cone = smoothstep(spot_lights.lights[i].cone.y, spot_lights.lights[i].cone.x, cos_angle);

		color += blinnPhong(normal, to_view, to_light,
		                    spot_lights.lights[i].color.rgb * (cone / dist2), albedo, specular_color);
	}

	// Свечение прибавляется ПОСЛЕ расчёта освещения: оно не зависит от источников света,
	// поэтому в темноте светящиеся места остаются яркими. Если emissive-текстуры нет,
	// вместо неё привязана текстура из одного чёрного пикселя -- прибавится 0.
	color += texture(emissive_map, in_uv).rgb * object.params.x;

	out_color = vec4(color, 1.0);
}
