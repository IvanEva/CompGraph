#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// Одна вершина в том виде, в котором она лежит в вершинном буфере:
// позиция и цвет подряд, без отступов (24 байта).
struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

// Сетка треугольников: вершины + индексы (каждые 3 индекса -- один треугольник).
struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

// Вариант 4: усечённый правильный тетраэдр.
// 12 вершин, 8 граней (4 правильных треугольника и 4 правильных шестиугольника),
// 18 рёбер одинаковой длины. Центр фигуры в начале координат, все координаты в [-1, 1].
Mesh makeTruncatedTetrahedron();
