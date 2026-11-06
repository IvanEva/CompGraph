#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// Одна вершина в вершинном буфере: позиция и нормаль подряд (24 байта).
// Цвета у вершины больше нет -- цвет задаётся материалом (см. Lab 2).
struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
};

// Сетка треугольников: вершины + индексы (каждые 3 индекса -- один треугольник).
struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

// Вариант 4: усечённый правильный тетраэдр.
// 8 граней (4 треугольника и 4 шестиугольника), у каждой грани свои вершины
// (36 штук), потому что у соседних граней нормали разные -- освещение "плоское".
Mesh makeTruncatedTetrahedron();
