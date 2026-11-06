#version 450

layout(location = 0) in vec3 in_color;
layout(location = 1) in vec3 in_local_position;

layout(location = 0) out vec4 out_color;

// Направление "света" в локальных координатах фигуры (свет вращается вместе с ней).
const vec3 light_direction = normalize(vec3(0.5, 0.8, 0.3));

void main() {
	// Затенение граней, только чтобы форма фигуры читалась на картинке (заданием не требуется).
	// dFdx/dFdy -- на сколько меняется положение при шаге на один пиксель по экрану.
	// Оба вектора лежат в плоскости грани, значит их векторное произведение -- нормаль грани.
	// У всех пикселей одной плоской грани нормаль одинакова, поэтому грань получается однотонной.
	vec3 normal = normalize(cross(dFdx(in_local_position), dFdy(in_local_position)));
	float shade = 0.5 + 0.5 * abs(dot(normal, light_direction));

	out_color = vec4(in_color * shade, 1.0);
}
