#include "mesh.hpp"

#include <algorithm>

namespace {

// Добавляет в сетку выпуклую грань, заданную вершинами по кругу (в любом направлении обхода).
// Грань режется на треугольники "веером" от первой вершины. Направление обхода
// выбирается так, чтобы при взгляде на грань снаружи вершины шли ПРОТИВ часовой стрелки:
// именно такие треугольники считаются лицевыми, остальные отсекаются (back-face culling).
void addFace(Mesh& mesh, std::vector<uint32_t> ring) {
	// Центр грани. Фигура выпуклая и стоит в начале координат,
	// поэтому вектор "от начала координат к центру грани" смотрит наружу.
	glm::vec3 center(0.0f);
	for (uint32_t i : ring) {
		center += mesh.vertices[i].position;
	}
	center /= float(ring.size());

	const glm::vec3& p0 = mesh.vertices[ring[0]].position;
	const glm::vec3& p1 = mesh.vertices[ring[1]].position;
	const glm::vec3& p2 = mesh.vertices[ring[2]].position;

	// Нормаль первого треугольника по правилу правой руки.
	// Если она смотрит внутрь фигуры, обход надо развернуть.
	const glm::vec3 normal = glm::cross(p1 - p0, p2 - p0);
	if (glm::dot(normal, center) < 0.0f) {
		std::reverse(ring.begin(), ring.end());
	}

	for (size_t k = 1; k + 1 < ring.size(); ++k) {
		mesh.indices.push_back(ring[0]);
		mesh.indices.push_back(ring[k]);
		mesh.indices.push_back(ring[k + 1]);
	}
}

} // namespace

Mesh makeTruncatedTetrahedron() {
	// Берём правильный тетраэдр (вершины -- четыре угла куба [-1, 1]^3 через один)
	// и "срезаем" каждую его вершину на расстоянии 1/3 длины ребра.
	const glm::vec3 tetra[4] = {
		{ 1.0f,  1.0f,  1.0f},
		{ 1.0f, -1.0f, -1.0f},
		{-1.0f,  1.0f, -1.0f},
		{-1.0f, -1.0f,  1.0f},
	};

	Mesh mesh;

	// point[i][j] -- индекс вершины на ребре от вершины i к вершине j,
	// лежащей на 1/3 длины ребра от i. Всего 4 * 3 = 12 вершин.
	uint32_t point[4][4] = {};

	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			if (i == j) {
				continue;
			}

			const glm::vec3 p = tetra[i] + (tetra[j] - tetra[i]) / 3.0f;

			// Цвет вершины считаем процедурно из её положения в локальных координатах:
			// x, y, z из диапазона [-1, 1] переводим в r, g, b из [0, 1].
			point[i][j] = uint32_t(mesh.vertices.size());
			mesh.vertices.push_back({p, 0.5f + 0.5f * p});
		}
	}

	// Четыре треугольные грани -- по одной на каждую срезанную вершину i.
	for (int i = 0; i < 4; ++i) {
		std::vector<uint32_t> ring;
		for (int j = 0; j < 4; ++j) {
			if (j != i) {
				ring.push_back(point[i][j]);
			}
		}
		addFace(mesh, ring);
	}

	// Четыре шестиугольные грани -- по одной на каждую исходную грань тетраэдра.
	// Грань напротив вершины k состоит из трёх остальных вершин a, b, c.
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

		// Идём по контуру a -> b -> c -> a; на каждом исходном ребре у шестиугольника
		// две вершины: ближняя к началу ребра и ближняя к концу.
		addFace(mesh, {
			point[a][b], point[b][a],
			point[b][c], point[c][b],
			point[c][a], point[a][c],
		});
	}

	return mesh;
}
