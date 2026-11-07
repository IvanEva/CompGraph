#include "mesh.hpp"

#include <algorithm>

namespace {

// Добавляет в сетку выпуклую грань, заданную точками по кругу (в любом направлении обхода).
// Вычисляет нормаль грани и текстурные координаты каждой вершины.
void addFace(Mesh& mesh, std::vector<glm::vec3> ring) {
	// Центр грани. Фигура выпуклая и стоит в начале координат,
	// поэтому вектор "от начала координат к центру грани" смотрит наружу.
	glm::vec3 center(0.0f);
	for (const glm::vec3& p : ring) {
		center += p;
	}
	center /= float(ring.size());

	// Нормаль по правилу правой руки. Если она смотрит внутрь, разворачиваем обход,
	// чтобы снаружи вершины шли против часовой стрелки (лицевая сторона).
	glm::vec3 normal = glm::cross(ring[1] - ring[0], ring[2] - ring[0]);
	if (glm::dot(normal, center) < 0.0f) {
		std::reverse(ring.begin(), ring.end());
		normal = -normal;
	}
	normal = glm::normalize(normal);

	// Оси текстурного пространства лежат В ПЛОСКОСТИ грани: tangent вдоль первого ребра,
	// bitangent перпендикулярен ему и нормали. Координата точки вдоль этих осей и есть (u, v).
	const glm::vec3 tangent = glm::normalize(ring[1] - ring[0]);
	const glm::vec3 bitangent = glm::cross(normal, tangent);

	// Ребро фигуры ~0.94, самая широкая грань (шестиугольник) ~1.9 в поперечнике.
	// Делим на 2, чтобы любая грань целиком уместилась в [0, 1], и сдвигаем центр в (0.5, 0.5).
	// Масштаб один на все грани, значит и размер текселя на всех гранях одинаков.
	constexpr float uv_scale = 0.5f;

	const uint32_t first = uint32_t(mesh.vertices.size());
	for (const glm::vec3& p : ring) {
		const glm::vec3 local = p - center;
		const glm::vec2 uv = glm::vec2(0.5f) + uv_scale * glm::vec2(glm::dot(local, tangent),
		                                                             glm::dot(local, bitangent));
		mesh.vertices.push_back({p, normal, uv});
	}

	// Режем грань на треугольники "веером" от первой вершины.
	for (uint32_t k = 1; k + 1 < ring.size(); ++k) {
		mesh.indices.push_back(first);
		mesh.indices.push_back(first + k);
		mesh.indices.push_back(first + k + 1);
	}
}

} // namespace

Mesh makeTruncatedTetrahedron() {
	// Правильный тетраэдр (четыре угла куба [-1, 1]^3 через один), каждая вершина срезана
	// на расстоянии 1/3 длины ребра.
	const glm::vec3 tetra[4] = {
		{ 1.0f,  1.0f,  1.0f},
		{ 1.0f, -1.0f, -1.0f},
		{-1.0f,  1.0f, -1.0f},
		{-1.0f, -1.0f,  1.0f},
	};

	// point[i][j] -- точка на ребре от вершины i к вершине j, на 1/3 длины от i.
	glm::vec3 point[4][4] = {};
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (i != j) {
				point[i][j] = tetra[i] + (tetra[j] - tetra[i]) / 3.0f;
			}
		}
	}

	Mesh mesh;

	// Четыре треугольные грани -- по одной на каждую срезанную вершину i.
	for (int i = 0; i < 4; ++i) {
		std::vector<glm::vec3> ring;
		for (int j = 0; j < 4; ++j) {
			if (j != i) {
				ring.push_back(point[i][j]);
			}
		}
		addFace(mesh, ring);
	}

	// Четыре шестиугольные грани -- по одной на каждую исходную грань тетраэдра
	// (грань напротив вершины k состоит из трёх остальных вершин a, b, c).
	for (int k = 0; k < 4; ++k) {
		int v[3];
		int n = 0;
		for (int i = 0; i < 4; ++i) {
			if (i != k) {
				v[n++] = i;
			}
		}
		const int a = v[0];
		const int b = v[1];
		const int c = v[2];

		addFace(mesh, {
			point[a][b], point[b][a],
			point[b][c], point[c][b],
			point[c][a], point[a][c],
		});
	}

	return mesh;
}

Mesh makeQuad() {
	const glm::vec3 normal(0.0f, 1.0f, 0.0f);

	// Порядок вершин -- против часовой стрелки при взгляде сверху (с +Y).
	Mesh mesh;
	mesh.vertices = {
		{{-0.5f, 0.0f,  0.5f}, normal, {0.0f, 0.0f}},
		{{ 0.5f, 0.0f,  0.5f}, normal, {1.0f, 0.0f}},
		{{ 0.5f, 0.0f, -0.5f}, normal, {1.0f, 1.0f}},
		{{-0.5f, 0.0f, -0.5f}, normal, {0.0f, 1.0f}},
	};
	mesh.indices = {0, 1, 2, 0, 2, 3};
	return mesh;
}
