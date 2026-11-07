#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include "mesh.hpp"
#include "texture.hpp"
#include "vk_utils.hpp"

namespace application {

namespace {

using graphics::internal::context;

// ---------------------------------------------------------------------------
// Константы
// ---------------------------------------------------------------------------

constexpr float near_plane = 0.1f;
constexpr float far_plane = 100.0f;

constexpr int max_point_lights = 16;
constexpr int max_spot_lights = 16;

// Сколько объектов помещается в динамический uniform-буфер.
constexpr int max_objects = 8;

// Биты флагов в push constants (те же значения, что в shaders/lit.frag).
constexpr uint32_t flag_animated_sampling = 1u << 0;

// ---------------------------------------------------------------------------
// Данные для GPU. Раскладка ОБЯЗАНА совпадать с блоками в shaders/lit.vert и lit.frag.
// ---------------------------------------------------------------------------

// Набор 0, binding 0 (uniform-буфер): общие данные сцены.
struct SceneData {
	glm::mat4 view_proj;
	glm::vec4 camera_position;
	glm::vec4 ambient;
	glm::vec4 dir_direction;
	glm::vec4 dir_color;
	glm::ivec4 counts;          // x -- точечных, y -- прожекторов
};

// Набор 0, binding 1 и 2 (storage-буферы): массивы источников.
struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 color;
};

struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 color;
	glm::vec4 cone;             // x -- cos внутреннего угла, y -- cos внешнего
};

// Набор 1, binding 0 (ДИНАМИЧЕСКИЙ uniform-буфер): свойства одной модели.
// В буфере лежит массив таких структур, каждая со сдвигом кратным minUniformBufferOffsetAlignment.
struct ModelData {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 tint;             // множитель цвета текстуры
	glm::vec4 specular;         // rgb -- цвет блика, a -- shininess
	glm::vec4 params;           // x -- сила свечения, yz -- масштаб текстурных координат
};

// Push constants (фрагментный шейдер). Должна совпадать с блоком PushConstants в lit.frag.
struct PushConstants {
	uint32_t flags;
	float time;
};
static_assert(sizeof(PushConstants) <= 128, "Vulkan guarantees only 128 bytes of push constants");

// ---------------------------------------------------------------------------
// Материалы: набор текстур + свой набор дескрипторов
// ---------------------------------------------------------------------------

// Описание материала: имя в интерфейсе и файлы текстур (пустая строка -- текстуры нет).
struct MaterialDescription {
	const char* name;
	const char* albedo;
	const char* specular;
	const char* emissive;
};

const MaterialDescription material_descriptions[] = {
	{"Lava rock",     "lava_albedo.png",    "lava_specular.png",    "lava_emissive.png"},
	{"Bricks",        "bricks_albedo.png",  "bricks_specular.png",  ""},
	{"Checker tiles", "checker_albedo.png", "checker_specular.png", ""},
	{"Wood planks",   "wood_albedo.png",    "wood_specular.png",    ""},
};
constexpr int material_count = int(sizeof(material_descriptions) / sizeof(material_descriptions[0]));

struct Material {
	Texture albedo;
	Texture specular;
	Texture emissive;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------------------
// Состояние, которым управляет пользователь
// ---------------------------------------------------------------------------

struct PointLightState {
	glm::vec3 position = {2.0f, 3.0f, 3.0f};
	glm::vec3 color = {1.0f, 1.0f, 1.0f};
	float intensity = 30.0f;
};

struct SpotLightState {
	glm::vec3 position = {0.0f, 4.0f, 3.0f};
	glm::vec3 direction = {0.0f, -1.0f, -0.8f};
	glm::vec3 color = {1.0f, 1.0f, 1.0f};
	float intensity = 60.0f;
	float inner_degrees = 12.0f;
	float outer_degrees = 22.0f;
};

// Какую сетку рисует объект (индексы в массиве mesh_ranges).
enum MeshId { mesh_tetrahedron = 0, mesh_quad = 1 };

struct ObjectState {
	std::string name;
	int mesh = mesh_tetrahedron;
	int material = 0;                               // индекс в material_descriptions

	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation_degrees = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};

	float spin_speed_degrees = 0.0f;                // собственное вращение, градусов в секунду
	float spin_degrees = 0.0f;

	glm::vec3 tint = {1.0f, 1.0f, 1.0f};
	float shininess = 48.0f;
	float emissive_strength = 1.0f;
	glm::vec2 uv_scale = {1.0f, 1.0f};              // сколько раз текстура повторяется по объекту
	bool animated_sampling = false;                 // нетривиальное сэмплирование (через push constants)
};

struct State {
	// Камера
	glm::vec3 camera_position = {0.0f, 1.2f, 7.0f};
	float camera_yaw_degrees = 0.0f;
	float camera_pitch_degrees = -8.0f;
	float fov_degrees = 45.0f;

	// Фоновый и направленный свет
	glm::vec3 ambient_color = {1.0f, 1.0f, 1.0f};
	float ambient_intensity = 0.06f;
	glm::vec3 dir_direction = {-0.4f, -0.8f, -0.5f};
	glm::vec3 dir_color = {1.0f, 1.0f, 1.0f};
	float dir_intensity = 0.35f;

	std::vector<PointLightState> point_lights = {PointLightState{}};
	std::vector<SpotLightState> spot_lights;

	float time = 0.0f;   // время в секундах для анимированного сэмплирования

	// Объекты сцены. Порядок = порядок записей в динамическом uniform-буфере.
	std::vector<ObjectState> objects;

	State() {
		ObjectState lava;
		lava.name = "Tetrahedron (lava)";
		lava.mesh = mesh_tetrahedron;
		lava.material = 0;
		lava.position = {-1.5f, 0.0f, 0.0f};
		lava.spin_speed_degrees = 25.0f;
		lava.emissive_strength = 1.5f;
		objects.push_back(lava);

		ObjectState wood;
		wood.name = "Tetrahedron (wood)";
		wood.mesh = mesh_tetrahedron;
		wood.material = 3;
		wood.position = {1.7f, -0.2f, -0.5f};
		wood.scale = {0.8f, 0.8f, 0.8f};
		wood.spin_speed_degrees = -35.0f;
		wood.animated_sampling = true;               // у этого объекта включено особое сэмплирование
		objects.push_back(wood);

		ObjectState floor;
		floor.name = "Floor";
		floor.mesh = mesh_quad;
		floor.material = 2;
		floor.position = {0.0f, -1.3f, 0.0f};
		floor.scale = {12.0f, 1.0f, 12.0f};
		floor.uv_scale = {6.0f, 6.0f};
		floor.shininess = 80.0f;
		objects.push_back(floor);

		ObjectState wall;
		wall.name = "Wall";
		wall.mesh = mesh_quad;
		wall.material = 1;
		wall.position = {0.0f, 1.7f, -4.0f};
		wall.rotation_degrees = {90.0f, 0.0f, 0.0f};    // квадрат лежал в XZ, ставим его "стоймя"
		wall.scale = {12.0f, 1.0f, 6.0f};
		wall.uv_scale = {3.0f, 1.5f};
		wall.shininess = 20.0f;
		objects.push_back(wall);
	}
};

// Часть общего вершинного/индексного буфера, принадлежащая одной сетке.
struct MeshRange {
	uint32_t index_count = 0;
	uint32_t first_index = 0;      // с какого индекса в общем индексном буфере начинается сетка
	int32_t vertex_offset = 0;     // сколько вершин лежит в общем буфере перед сеткой
};

// ---------------------------------------------------------------------------
// Глобальное состояние приложения
// ---------------------------------------------------------------------------

State state;

double last_time = 0.0;
bool has_last_time = false;

// Вершины и индексы ВСЕХ сеток лежат подряд в двух буферах в видеопамяти.
vkutil::Buffer vertex_buffer;
vkutil::Buffer index_buffer;
MeshRange mesh_ranges[2];

vkutil::Buffer scene_buffer;
vkutil::Buffer point_light_buffer;
vkutil::Buffer spot_light_buffer;
vkutil::Buffer model_buffer;            // динамический uniform: max_objects структур ModelData
VkDeviceSize model_stride = 0;          // расстояние между соседними структурами в model_buffer

VkSampler sampler = VK_NULL_HANDLE;
Texture black_texture;                  // заглушка 1x1 вместо отсутствующих specular/emissive
Material materials[material_count];

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// ---------------------------------------------------------------------------
// Создание ресурсов
// ---------------------------------------------------------------------------

// Кладёт сетки подряд в общие массивы и загружает их в видеопамять через staging-буфер.
bool createGeometry() {
	const Mesh meshes[2] = {makeTruncatedTetrahedron(), makeQuad()};

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for (int i = 0; i < 2; ++i) {
		mesh_ranges[i].index_count = uint32_t(meshes[i].indices.size());
		mesh_ranges[i].first_index = uint32_t(indices.size());
		mesh_ranges[i].vertex_offset = int32_t(vertices.size());

		// Индексы в каждой сетке считаются от её первой вершины; при отрисовке к ним
		// добавляется vertex_offset, поэтому здесь индексы переписывать не нужно.
		vertices.insert(vertices.end(), meshes[i].vertices.begin(), meshes[i].vertices.end());
		indices.insert(indices.end(), meshes[i].indices.begin(), meshes[i].indices.end());
	}

	return vkutil::createDeviceBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
	                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) &&
	       vkutil::createDeviceBuffer(indices.data(), indices.size() * sizeof(uint32_t),
	                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer);
}

bool createUniformBuffers() {
	// Сдвиг между структурами в динамическом буфере обязан быть кратен этому пределу устройства.
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelData) + alignment - 1) / alignment * alignment;

	return vkutil::createHostBuffer(nullptr, sizeof(SceneData),
	                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) &&
	       vkutil::createHostBuffer(nullptr, sizeof(GpuPointLight) * max_point_lights,
	                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) &&
	       vkutil::createHostBuffer(nullptr, sizeof(GpuSpotLight) * max_spot_lights,
	                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer) &&
	       vkutil::createHostBuffer(nullptr, model_stride * max_objects,
	                                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer);
}

// Загружает все текстуры материалов из папки textures/.
bool createTextures() {
	if (!createSampler(sampler)) {
		return false;
	}

	// Чёрный пиксель: если у материала нет specular или emissive, используем его --
	// блеск (specular) и свечение (emissive) тогда просто равны нулю.
	if (!createSolidTexture(0, 0, 0, 255, false, black_texture)) {
		return false;
	}

	const std::string directory = TEXTURE_DIR;

	for (int i = 0; i < material_count; ++i) {
		const MaterialDescription& description = material_descriptions[i];
		Material& material = materials[i];

		// albedo и emissive -- цвета (sRGB), specular -- просто числа (UNORM).
		if (!loadTextureFromFile(directory + description.albedo, true, material.albedo)) {
			return false;
		}
		if (description.specular[0] != '\0' &&
		    !loadTextureFromFile(directory + description.specular, false, material.specular)) {
			return false;
		}
		if (description.emissive[0] != '\0' &&
		    !loadTextureFromFile(directory + description.emissive, true, material.emissive)) {
			return false;
		}
	}

	return true;
}

// Дескрипторы: набор 0 -- сцена, набор 1 -- свойства моделей (динамический UBO),
// набор 2 -- материал (по одному набору на каждый материал).
bool createDescriptors() {
	const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, both, nullptr},
		{1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
	};

	// Тип UNIFORM_BUFFER_DYNAMIC: сдвиг внутри буфера указывается при привязке набора.
	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, both, nullptr},
	};

	// Тип COMBINED_IMAGE_SAMPLER: изображение и сэмплер в одном дескрипторе.
	const VkDescriptorSetLayoutBinding material_bindings[] = {
		{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
		{2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
	};

	auto createLayout = [](const VkDescriptorSetLayoutBinding* bindings, uint32_t count,
	                       VkDescriptorSetLayout& layout) {
		const VkDescriptorSetLayoutCreateInfo info = {
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = count,
			.pBindings = bindings,
		};
		return vkCreateDescriptorSetLayout(context.device, &info, nullptr, &layout) == VK_SUCCESS;
	};

	if (!createLayout(scene_bindings, 3, scene_set_layout) ||
	    !createLayout(model_bindings, 1, model_set_layout) ||
	    !createLayout(material_bindings, 3, material_set_layout)) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	// Пул: 2 + material_count наборов; считаем, сколько дескрипторов каждого типа понадобится.
	const VkDescriptorPoolSize pool_sizes[] = {
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
		{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
		{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
		{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * material_count},
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2 + material_count,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	// Выделяем наборы: сцена, модель и по одному набору на материал.
	VkDescriptorSetLayout layouts[2 + material_count];
	layouts[0] = scene_set_layout;
	layouts[1] = model_set_layout;
	for (int i = 0; i < material_count; ++i) {
		layouts[2 + i] = material_set_layout;
	}

	VkDescriptorSet sets[2 + material_count] = {};
	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2 + material_count,
		.pSetLayouts = layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}
	scene_set = sets[0];
	model_set = sets[1];
	for (int i = 0; i < material_count; ++i) {
		materials[i].set = sets[2 + i];
	}

	// --- Привязываем буферы к наборам 0 и 1 ---
	const VkDescriptorBufferInfo scene_info = {scene_buffer.buffer, 0, sizeof(SceneData)};
	const VkDescriptorBufferInfo point_info = {point_light_buffer.buffer, 0, sizeof(GpuPointLight) * max_point_lights};
	const VkDescriptorBufferInfo spot_info = {spot_light_buffer.buffer, 0, sizeof(GpuSpotLight) * max_spot_lights};
	// Для динамического буфера range -- размер ОДНОЙ структуры. Сдвиг к нужной даётся при привязке набора.
	const VkDescriptorBufferInfo model_info = {model_buffer.buffer, 0, sizeof(ModelData)};

	std::vector<VkWriteDescriptorSet> writes = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 0, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &scene_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 1, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &point_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set, .dstBinding = 2, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &spot_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = model_set, .dstBinding = 0, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &model_info,
		},
	};

	// --- Привязываем текстуры к наборам материалов (3 дескриптора на материал) ---
	// Массив должен пережить vkUpdateDescriptorSets, поэтому объявлен здесь.
	std::vector<VkDescriptorImageInfo> image_infos(size_t(material_count) * 3);

	for (int i = 0; i < material_count; ++i) {
		// Если у материала нет specular/emissive -- подставляем чёрную заглушку.
		const VkImageView views[3] = {
			materials[i].albedo.view,
			materials[i].specular.view != VK_NULL_HANDLE ? materials[i].specular.view : black_texture.view,
			materials[i].emissive.view != VK_NULL_HANDLE ? materials[i].emissive.view : black_texture.view,
		};

		for (uint32_t binding = 0; binding < 3; ++binding) {
			VkDescriptorImageInfo& image_info = image_infos[size_t(i) * 3 + binding];
			image_info = {
				.sampler = sampler,
				.imageView = views[binding],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,   // в этом состоянии лежит текстура
			};

			writes.push_back({
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = materials[i].set, .dstBinding = binding, .descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_info,
			});
		}
	}

	vkUpdateDescriptorSets(context.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
	return true;
}

// Собирает графический конвейер: шейдеры + все настройки фиксированных стадий.
bool createPipeline() {
	// Порядок наборов = номера set в шейдере: 0 -- сцена, 1 -- модель, 2 -- материал.
	const VkDescriptorSetLayout set_layouts[] = {scene_set_layout, model_set_layout, material_set_layout};

	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 3,
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!vkutil::createShaderModule(std::string(SHADER_DIR) + "lit.vert.spv", vert) ||
	    !vkutil::createShaderModule(std::string(SHADER_DIR) + "lit.frag.spv", frag)) {
		vkDestroyShaderModule(context.device, vert, nullptr);
		vkDestroyShaderModule(context.device, frag, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag,
			.pName = "main",
		},
	};

	// Вершина: позиция (location 0), нормаль (1), текстурные координаты (2).
	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)},
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
		{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv)},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = sizeof(dynamic_states) / sizeof(dynamic_states[0]),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = sizeof(stages) / sizeof(stages[0]),
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                  &pipeline_info, nullptr, &pipeline);

	vkDestroyShaderModule(context.device, vert, nullptr);
	vkDestroyShaderModule(context.device, frag, nullptr);

	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		return false;
	}

	return true;
}

// ---------------------------------------------------------------------------
// Математика сцены
// ---------------------------------------------------------------------------

glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
	const float length = glm::length(v);
	return length > 1e-6f ? v / length : fallback;
}

// Матрица модели объекта: сначала масштаб, потом повороты, потом перенос (справа налево).
glm::mat4 modelMatrix(const ObjectState& object) {
	const glm::vec3 spin_axis = glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));

	glm::mat4 model = glm::translate(glm::mat4(1.0f), object.position);
	model = glm::rotate(model, glm::radians(object.spin_degrees), spin_axis);
	model = glm::rotate(model, glm::radians(object.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::rotate(model, glm::radians(object.rotation_degrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(object.rotation_degrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::scale(model, object.scale);
	return model;
}

glm::mat4 viewMatrix() {
	glm::mat4 camera = glm::translate(glm::mat4(1.0f), state.camera_position);
	camera = glm::rotate(camera, glm::radians(state.camera_yaw_degrees), glm::vec3(0.0f, 1.0f, 0.0f));
	camera = glm::rotate(camera, glm::radians(state.camera_pitch_degrees), glm::vec3(1.0f, 0.0f, 0.0f));
	return glm::inverse(camera);
}

glm::mat4 projectionMatrix(float aspect) {
	glm::mat4 projection = glm::perspective(glm::radians(state.fov_degrees), aspect, near_plane, far_plane);
	projection[1][1] *= -1.0f;   // ось Y экрана в Vulkan направлена вниз
	return projection;
}

// Заполняет все GPU-буферы данными из текущего состояния.
void uploadFrameData(float aspect) {
	// --- Свойства объектов: все структуры подряд в одном динамическом буфере ---
	std::vector<uint8_t> model_bytes(size_t(model_stride) * max_objects, 0);
	const int object_count = int(std::min<size_t>(state.objects.size(), max_objects));

	for (int i = 0; i < object_count; ++i) {
		const ObjectState& object = state.objects[i];
		const glm::mat4 model = modelMatrix(object);

		const ModelData data = {
			.model = model,
			.normal_matrix = glm::mat4(glm::inverseTranspose(glm::mat3(model))),
			.tint = glm::vec4(object.tint, 1.0f),
			.specular = glm::vec4(1.0f, 1.0f, 1.0f, object.shininess),
			.params = glm::vec4(object.emissive_strength, object.uv_scale.x, object.uv_scale.y, 0.0f),
		};
		std::memcpy(model_bytes.data() + size_t(i) * model_stride, &data, sizeof(data));
	}
	vkutil::updateHostBuffer(model_buffer, model_bytes.data(), model_bytes.size());

	// --- Источники ---
	GpuPointLight point_lights[max_point_lights] = {};
	const int point_count = int(std::min<size_t>(state.point_lights.size(), max_point_lights));
	for (int i = 0; i < point_count; ++i) {
		const PointLightState& light = state.point_lights[i];
		point_lights[i].position = glm::vec4(light.position, 1.0f);
		point_lights[i].color = glm::vec4(light.color * light.intensity, 1.0f);
	}
	vkutil::updateHostBuffer(point_light_buffer, point_lights, sizeof(point_lights));

	GpuSpotLight spot_lights[max_spot_lights] = {};
	const int spot_count = int(std::min<size_t>(state.spot_lights.size(), max_spot_lights));
	for (int i = 0; i < spot_count; ++i) {
		const SpotLightState& light = state.spot_lights[i];
		const float outer = glm::radians(light.outer_degrees);
		const float inner = glm::radians(std::min(light.inner_degrees, light.outer_degrees));
		spot_lights[i].position = glm::vec4(light.position, 1.0f);
		spot_lights[i].direction = glm::vec4(safeNormalize(light.direction, {0.0f, -1.0f, 0.0f}), 0.0f);
		spot_lights[i].color = glm::vec4(light.color * light.intensity, 1.0f);
		spot_lights[i].cone = glm::vec4(std::cos(inner), std::cos(outer), 0.0f, 0.0f);
	}
	vkutil::updateHostBuffer(spot_light_buffer, spot_lights, sizeof(spot_lights));

	// --- Данные сцены ---
	const SceneData scene_data = {
		.view_proj = projectionMatrix(aspect) * viewMatrix(),
		.camera_position = glm::vec4(state.camera_position, 1.0f),
		.ambient = glm::vec4(state.ambient_color * state.ambient_intensity, 1.0f),
		.dir_direction = glm::vec4(safeNormalize(state.dir_direction, {0.0f, -1.0f, 0.0f}), 0.0f),
		.dir_color = glm::vec4(state.dir_color * state.dir_intensity, 1.0f),
		.counts = glm::ivec4(point_count, spot_count, 0, 0),
	};
	vkutil::updateHostBuffer(scene_buffer, &scene_data, sizeof(scene_data));
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------

// Выпадающий список материалов. Возвращает true, если выбор изменился.
bool materialCombo(const char* label, int& current) {
	bool changed = false;
	if (ImGui::BeginCombo(label, material_descriptions[current].name)) {
		for (int i = 0; i < material_count; ++i) {
			const bool selected = (i == current);
			if (ImGui::Selectable(material_descriptions[i].name, selected)) {
				current = i;
				changed = true;
			}
			if (selected) {
				ImGui::SetItemDefaultFocus();
			}
		}
		ImGui::EndCombo();
	}
	return changed;
}

void drawUserInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(420.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Controls");

	if (ImGui::CollapsingHeader("Camera")) {
		ImGui::DragFloat3("Camera position", glm::value_ptr(state.camera_position), 0.02f);
		ImGui::SliderFloat("Yaw (deg)", &state.camera_yaw_degrees, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch (deg)", &state.camera_pitch_degrees, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV (deg)", &state.fov_degrees, 20.0f, 90.0f);
	}

	if (ImGui::CollapsingHeader("Objects", ImGuiTreeNodeFlags_DefaultOpen)) {
		for (int i = 0; i < int(state.objects.size()); ++i) {
			ObjectState& object = state.objects[i];
			ImGui::PushID(i);

			if (ImGui::TreeNodeEx(object.name.c_str(), i == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
				materialCombo("Material", object.material);
				ImGui::DragFloat3("Position", glm::value_ptr(object.position), 0.02f);
				ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(object.rotation_degrees), 0.5f);
				ImGui::DragFloat3("Scale", glm::value_ptr(object.scale), 0.01f, 0.05f, 20.0f, "%.2f",
				                  ImGuiSliderFlags_AlwaysClamp);
				ImGui::DragFloat2("UV tiling", glm::value_ptr(object.uv_scale), 0.05f, 0.1f, 20.0f);
				ImGui::SliderFloat("Spin (deg/s)", &object.spin_speed_degrees, -180.0f, 180.0f);
				ImGui::ColorEdit3("Tint", glm::value_ptr(object.tint));
				ImGui::SliderFloat("Shininess", &object.shininess, 1.0f, 256.0f, "%.0f",
				                   ImGuiSliderFlags_Logarithmic);
				ImGui::SliderFloat("Emissive", &object.emissive_strength, 0.0f, 4.0f);
				ImGui::Checkbox("Animated sampling", &object.animated_sampling);
				ImGui::TreePop();
			}

			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Ambient and directional light")) {
		ImGui::ColorEdit3("Ambient color", glm::value_ptr(state.ambient_color));
		ImGui::SliderFloat("Ambient intensity", &state.ambient_intensity, 0.0f, 1.0f);
		ImGui::DragFloat3("Dir. direction", glm::value_ptr(state.dir_direction), 0.01f);
		ImGui::ColorEdit3("Dir. color", glm::value_ptr(state.dir_color));
		ImGui::SliderFloat("Dir. intensity", &state.dir_intensity, 0.0f, 3.0f);
	}

	if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("%d / %d", int(state.point_lights.size()), max_point_lights);
		ImGui::SameLine();
		if (ImGui::Button("Add point light") && state.point_lights.size() < size_t(max_point_lights)) {
			state.point_lights.emplace_back();
		}

		int remove_index = -1;
		for (int i = 0; i < int(state.point_lights.size()); ++i) {
			PointLightState& light = state.point_lights[i];
			ImGui::PushID(100 + i);
			ImGui::SeparatorText(("Point " + std::to_string(i)).c_str());
			ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.02f);
			ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 150.0f);
			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::PopID();
		}
		if (remove_index >= 0) {
			state.point_lights.erase(state.point_lights.begin() + remove_index);
		}
	}

	if (ImGui::CollapsingHeader("Spot lights")) {
		ImGui::Text("%d / %d", int(state.spot_lights.size()), max_spot_lights);
		ImGui::SameLine();
		if (ImGui::Button("Add spot light") && state.spot_lights.size() < size_t(max_spot_lights)) {
			state.spot_lights.emplace_back();
		}

		int remove_index = -1;
		for (int i = 0; i < int(state.spot_lights.size()); ++i) {
			SpotLightState& light = state.spot_lights[i];
			ImGui::PushID(1000 + i);
			ImGui::SeparatorText(("Spot " + std::to_string(i)).c_str());
			ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.02f);
			ImGui::DragFloat3("Direction", glm::value_ptr(light.direction), 0.01f);
			ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 300.0f);
			ImGui::SliderFloat("Inner angle (deg)", &light.inner_degrees, 0.0f, 89.0f);
			ImGui::SliderFloat("Outer angle (deg)", &light.outer_degrees, 1.0f, 89.0f);
			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::PopID();
		}
		if (remove_index >= 0) {
			state.spot_lights.erase(state.spot_lights.begin() + remove_index);
		}
	}

	ImGui::Separator();
	if (ImGui::Button("Reset all")) {
		state = State{};
	}

	ImGui::End();
}

} // namespace

// ---------------------------------------------------------------------------
// Интерфейс модуля (объявлен в application.hpp)
// ---------------------------------------------------------------------------

bool initialize() {
	ImGui::GetIO().IniFilename = nullptr;

	// Порядок важен: пул команд -> геометрия и буферы -> текстуры -> дескрипторы -> конвейер.
	if (!vkutil::initialize() || !createGeometry() || !createUniformBuffers() ||
	    !createTextures() || !createDescriptors() || !createPipeline()) {
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

	for (Material& material : materials) {
		destroyTexture(material.albedo);
		destroyTexture(material.specular);
		destroyTexture(material.emissive);
	}
	destroyTexture(black_texture);
	destroySampler(sampler);

	vkutil::destroyBuffer(model_buffer);
	vkutil::destroyBuffer(spot_light_buffer);
	vkutil::destroyBuffer(point_light_buffer);
	vkutil::destroyBuffer(scene_buffer);
	vkutil::destroyBuffer(index_buffer);
	vkutil::destroyBuffer(vertex_buffer);

	vkutil::shutdown();
}

void update(double time) {
	float dt = 0.0f;
	if (has_last_time) {
		dt = std::min(float(time - last_time), 0.1f);
	}
	last_time = time;
	has_last_time = true;

	state.time += dt;
	for (ObjectState& object : state.objects) {
		object.spin_degrees += dt * object.spin_speed_degrees;
	}

	drawUserInterface();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkExtent2D extent = context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	uploadFrameData(aspect);

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkClearValue clear_values[2] = {};
	clear_values[0].color = {{0.01f, 0.012f, 0.02f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	const VkRenderPassBeginInfo render_pass_info = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(extent.width),
		.height = float(extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// Данные сцены общие для всех объектов: привязываем набор 0 один раз.
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	// Вершины и индексы всех сеток лежат в общих буферах, привязываем их один раз.
	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	const int object_count = int(std::min<size_t>(state.objects.size(), max_objects));
	for (int i = 0; i < object_count; ++i) {
		const ObjectState& object = state.objects[i];
		const MeshRange& range = mesh_ranges[object.mesh];

		// Наборы 1 и 2 одной командой (firstSet = 1):
		//   набор 1 -- динамический буфер: dynamicOffset выбирает i-ую структуру ModelData;
		//   набор 2 -- материал объекта (его текстуры).
		const VkDescriptorSet sets[] = {model_set, materials[object.material].set};
		const uint32_t dynamic_offset = uint32_t(model_stride * VkDeviceSize(i));
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 2, sets, 1, &dynamic_offset);

		// Push constants: быстрая передача флагов и времени прямо в команде.
		const PushConstants constants = {
			.flags = object.animated_sampling ? flag_animated_sampling : 0u,
			.time = state.time,
		};
		vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
		                   sizeof(PushConstants), &constants);

		vkCmdDrawIndexed(cmd, range.index_count, 1, range.first_index, range.vertex_offset, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
