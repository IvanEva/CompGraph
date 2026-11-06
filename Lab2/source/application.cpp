#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include "mesh.hpp"

namespace application {

namespace {

using graphics::internal::context;

// ---------------------------------------------------------------------------
// Константы
// ---------------------------------------------------------------------------

constexpr float near_plane = 0.1f;
constexpr float far_plane = 100.0f;

// Максимум источников каждого типа. Storage-буферы создаются сразу на это число,
// поэтому их не нужно пересоздавать, когда пользователь добавляет или удаляет источник.
constexpr int max_point_lights = 16;
constexpr int max_spot_lights = 16;

// ---------------------------------------------------------------------------
// Данные для GPU. Раскладка ОБЯЗАНА совпадать с блоками в shaders/lit.vert и lit.frag.
// Везде используем vec4 вместо vec3: в std140/std430 vec3 занимает 16 байт,
// и с vec4 не нужно считать отступы вручную.
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

// Набор 0, binding 1 (storage-буфер): массив точечных источников.
struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 color;
};

// Набор 0, binding 2 (storage-буфер): массив прожекторов.
struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 color;
	glm::vec4 cone;             // x -- cos внутреннего угла, y -- cos внешнего
};

// Набор 1, binding 0 (uniform-буфер): данные модели.
struct ModelData {
	glm::mat4 model;
	glm::mat4 normal_matrix;
	glm::vec4 albedo;
	glm::vec4 specular;         // rgb -- цвет блика, a -- shininess
};

// ---------------------------------------------------------------------------
// Состояние, которым управляет пользователь
// ---------------------------------------------------------------------------

struct PointLightState {
	glm::vec3 position = {2.0f, 2.0f, 3.0f};
	glm::vec3 color = {1.0f, 1.0f, 1.0f};
	float intensity = 12.0f;
};

struct SpotLightState {
	glm::vec3 position = {0.0f, 3.0f, 3.0f};
	glm::vec3 direction = {0.0f, -1.0f, -0.8f};   // куда смотрит конус
	glm::vec3 color = {1.0f, 1.0f, 1.0f};
	float intensity = 30.0f;
	float inner_degrees = 12.0f;                  // внутри -- полная яркость
	float outer_degrees = 22.0f;                  // снаружи -- темнота
};

struct State {
	// Камера
	glm::vec3 camera_position = {0.0f, 0.0f, 6.0f};
	float camera_yaw_degrees = 0.0f;              // поворот вокруг оси Y
	float camera_pitch_degrees = 0.0f;            // поворот вокруг оси X
	float fov_degrees = 45.0f;

	// Объект
	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation_degrees = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};
	bool spinning = true;
	float spin_speed_degrees = 30.0f;
	float spin_degrees = 0.0f;

	// Материал
	glm::vec3 albedo = {0.85f, 0.45f, 0.15f};
	glm::vec3 specular = {1.0f, 1.0f, 1.0f};
	float shininess = 48.0f;

	// Фоновый и направленный свет
	glm::vec3 ambient_color = {1.0f, 1.0f, 1.0f};
	float ambient_intensity = 0.08f;
	glm::vec3 dir_direction = {-0.4f, -0.8f, -0.5f};   // куда светит
	glm::vec3 dir_color = {1.0f, 1.0f, 1.0f};
	float dir_intensity = 0.7f;

	// Динамические массивы источников
	std::vector<PointLightState> point_lights = {PointLightState{}};
	std::vector<SpotLightState> spot_lights;
};

// Буфер Vulkan вместе с выделенной под него памятью (память выделяет VMA).
struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
};

// ---------------------------------------------------------------------------
// Глобальное состояние приложения
// ---------------------------------------------------------------------------

State state;

double last_time = 0.0;
bool has_last_time = false;

Buffer vertex_buffer;
Buffer index_buffer;
uint32_t index_count = 0;

Buffer scene_buffer;         // SceneData (uniform)
Buffer point_light_buffer;   // GpuPointLight[max_point_lights] (storage)
Buffer spot_light_buffer;    // GpuSpotLight[max_spot_lights] (storage)
Buffer model_buffer;         // ModelData (uniform)

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// ---------------------------------------------------------------------------
// Вспомогательные функции Vulkan
// ---------------------------------------------------------------------------

// Создаёт буфер в памяти, доступной процессору. Если data == nullptr, буфер остаётся
// незаполненным (его будем заполнять каждый кадр через updateBuffer).
bool createBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	if (data != nullptr &&
	    vmaCopyMemoryToAllocation(context.allocator, data, out.allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to copy data into Vulkan buffer\n";
		return false;
	}

	return true;
}

// Перезаписывает начало буфера новыми данными (вызывается каждый кадр).
// Кадр у нас один "в полёте", предыдущий уже закончился, так что гонки с GPU нет.
void updateBuffer(const Buffer& buffer, const void* data, VkDeviceSize size) {
	vmaCopyMemoryToAllocation(context.allocator, data, buffer.allocation, 0, size);
}

// Читает скомпилированный шейдер (SPIR-V) из файла и создаёт VkShaderModule.
bool createShaderModule(const std::string& path, VkShaderModule& module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	if (size <= 0 || size % 4 != 0) {
		std::cerr << "Shader file has invalid size: " << path << '\n';
		return false;
	}

	std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), size);
	if (!file) {
		std::cerr << "Failed to read shader file: " << path << '\n';
		return false;
	}

	const VkShaderModuleCreateInfo module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(size),
		.pCode = code.data(),
	};

	if (vkCreateShaderModule(context.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << path << '\n';
		return false;
	}

	return true;
}

// Создаёт все GPU-буферы. Буферы источников выделяются сразу на максимальный размер.
bool createBuffers(const Mesh& mesh) {
	return createBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
	                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) &&
	       createBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
	                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) &&
	       createBuffer(nullptr, sizeof(SceneData),
	                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer) &&
	       createBuffer(nullptr, sizeof(GpuPointLight) * max_point_lights,
	                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_light_buffer) &&
	       createBuffer(nullptr, sizeof(GpuSpotLight) * max_spot_lights,
	                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_light_buffer) &&
	       createBuffer(nullptr, sizeof(ModelData),
	                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer);
}

// Дескрипторы: два набора (set 0 -- сцена, set 1 -- модель) и пул, из которого они выделяются.
bool createDescriptors() {
	// Набор 0: uniform-буфер сцены + два storage-буфера с источниками.
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	// Набор 1: uniform-буфер модели.
	const VkDescriptorSetLayoutBinding model_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(scene_bindings) / sizeof(scene_bindings[0]),
		.pBindings = scene_bindings,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = sizeof(model_bindings) / sizeof(model_bindings[0]),
		.pBindings = model_bindings,
	};

	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &scene_set_layout) != VK_SUCCESS ||
	    vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layouts\n";
		return false;
	}

	// Пул: всего 2 набора; в них суммарно 2 uniform-буфера и 2 storage-буфера.
	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = sizeof(pool_sizes) / sizeof(pool_sizes[0]),
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout layouts[] = {scene_set_layout, model_set_layout};
	VkDescriptorSet sets[2] = {};
	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &allocate_info, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}
	scene_set = sets[0];
	model_set = sets[1];

	// Привязываем к наборам конкретные буферы. Делается один раз: буферы не пересоздаются,
	// меняется только их содержимое.
	const VkDescriptorBufferInfo scene_info = {scene_buffer.buffer, 0, sizeof(SceneData)};
	const VkDescriptorBufferInfo point_info = {point_light_buffer.buffer, 0, sizeof(GpuPointLight) * max_point_lights};
	const VkDescriptorBufferInfo spot_info = {spot_light_buffer.buffer, 0, sizeof(GpuSpotLight) * max_spot_lights};
	const VkDescriptorBufferInfo model_info = {model_buffer.buffer, 0, sizeof(ModelData)};

	const VkWriteDescriptorSet writes[] = {
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
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &model_info,
		},
	};
	vkUpdateDescriptorSets(context.device, sizeof(writes) / sizeof(writes[0]), writes, 0, nullptr);

	return true;
}

// Собирает графический конвейер: шейдеры + все настройки фиксированных стадий.
bool createPipeline() {
	// Layout конвейера: порядок наборов здесь = номера set в шейдере (0 -- сцена, 1 -- модель).
	const VkDescriptorSetLayout set_layouts[] = {scene_set_layout, model_set_layout};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!createShaderModule(std::string(SHADER_DIR) + "lit.vert.spv", vert) ||
	    !createShaderModule(std::string(SHADER_DIR) + "lit.frag.spv", frag)) {
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

	// Вершинный буфер: позиция (location 0) и нормаль (location 1).
	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
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

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

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

// Нормализует вектор, а если он нулевой, возвращает запасное направление.
glm::vec3 safeNormalize(const glm::vec3& v, const glm::vec3& fallback) {
	const float length = glm::length(v);
	return length > 1e-6f ? v / length : fallback;
}

// Матрица модели: сначала масштаб, потом повороты, потом перенос (читаем справа налево).
glm::mat4 modelMatrix() {
	const glm::vec3 spin_axis = glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));

	glm::mat4 model = glm::translate(glm::mat4(1.0f), state.position);
	model = glm::rotate(model, glm::radians(state.spin_degrees), spin_axis);
	model = glm::rotate(model, glm::radians(state.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::rotate(model, glm::radians(state.rotation_degrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(state.rotation_degrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::scale(model, state.scale);
	return model;
}

// Матрица вида. Сначала строим матрицу САМОЙ КАМЕРЫ как обычного объекта в мире
// (переместить, повернуть), а вид -- это обратная к ней: она переносит мир так,
// чтобы камера оказалась в начале координат и смотрела вдоль -Z.
glm::mat4 viewMatrix() {
	glm::mat4 camera = glm::translate(glm::mat4(1.0f), state.camera_position);
	camera = glm::rotate(camera, glm::radians(state.camera_yaw_degrees), glm::vec3(0.0f, 1.0f, 0.0f));
	camera = glm::rotate(camera, glm::radians(state.camera_pitch_degrees), glm::vec3(1.0f, 0.0f, 0.0f));
	return glm::inverse(camera);
}

glm::mat4 projectionMatrix(float aspect) {
	glm::mat4 projection = glm::perspective(glm::radians(state.fov_degrees), aspect, near_plane, far_plane);

	// В Vulkan ось Y экрана направлена вниз, поэтому переворачиваем её.
	projection[1][1] *= -1.0f;

	return projection;
}

// Заполняет все GPU-буферы данными из текущего состояния.
void uploadFrameData(float aspect) {
	const glm::mat4 model = modelMatrix();

	// --- Данные модели (set 1) ---
	// Нормальная матрица = transpose(inverse(model)). Обычный model для нормалей не годится:
	// при неравномерном масштабе нормаль перестаёт быть перпендикулярна поверхности.
	// (Матрицу mat3 кладём в mat4: в std140 столбцы mat3 занимали бы по 16 байт, так проще.)
	const ModelData model_data = {
		.model = model,
		.normal_matrix = glm::mat4(glm::inverseTranspose(glm::mat3(model))),
		.albedo = glm::vec4(state.albedo, 1.0f),
		.specular = glm::vec4(state.specular, state.shininess),
	};
	updateBuffer(model_buffer, &model_data, sizeof(model_data));

	// --- Источники (set 0, binding 1 и 2) ---
	GpuPointLight point_lights[max_point_lights] = {};
	const int point_count = int(std::min<size_t>(state.point_lights.size(), max_point_lights));
	for (int i = 0; i < point_count; ++i) {
		const PointLightState& light = state.point_lights[i];
		point_lights[i].position = glm::vec4(light.position, 1.0f);
		point_lights[i].color = glm::vec4(light.color * light.intensity, 1.0f);
	}
	updateBuffer(point_light_buffer, point_lights, sizeof(point_lights));

	GpuSpotLight spot_lights[max_spot_lights] = {};
	const int spot_count = int(std::min<size_t>(state.spot_lights.size(), max_spot_lights));
	for (int i = 0; i < spot_count; ++i) {
		const SpotLightState& light = state.spot_lights[i];
		// В шейдер передаём косинусы углов: сравнивать с dot() дешевле, чем считать acos.
		const float outer = glm::radians(light.outer_degrees);
		const float inner = glm::radians(std::min(light.inner_degrees, light.outer_degrees));
		spot_lights[i].position = glm::vec4(light.position, 1.0f);
		spot_lights[i].direction = glm::vec4(safeNormalize(light.direction, {0.0f, -1.0f, 0.0f}), 0.0f);
		spot_lights[i].color = glm::vec4(light.color * light.intensity, 1.0f);
		spot_lights[i].cone = glm::vec4(std::cos(inner), std::cos(outer), 0.0f, 0.0f);
	}
	updateBuffer(spot_light_buffer, spot_lights, sizeof(spot_lights));

	// --- Данные сцены (set 0, binding 0) ---
	const SceneData scene_data = {
		.view_proj = projectionMatrix(aspect) * viewMatrix(),
		.camera_position = glm::vec4(state.camera_position, 1.0f),
		.ambient = glm::vec4(state.ambient_color * state.ambient_intensity, 1.0f),
		.dir_direction = glm::vec4(safeNormalize(state.dir_direction, {0.0f, -1.0f, 0.0f}), 0.0f),
		.dir_color = glm::vec4(state.dir_color * state.dir_intensity, 1.0f),
		.counts = glm::ivec4(point_count, spot_count, 0, 0),
	};
	updateBuffer(scene_buffer, &scene_data, sizeof(scene_data));
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------

void drawUserInterface() {
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(400.0f, 600.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Controls");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Camera position", glm::value_ptr(state.camera_position), 0.02f);
		ImGui::SliderFloat("Yaw (deg)", &state.camera_yaw_degrees, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch (deg)", &state.camera_pitch_degrees, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV (deg)", &state.fov_degrees, 20.0f, 90.0f);
	}

	if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", glm::value_ptr(state.position), 0.01f);
		ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(state.rotation_degrees), 0.5f);
		ImGui::DragFloat3("Scale", glm::value_ptr(state.scale), 0.01f, 0.05f, 5.0f, "%.2f",
		                  ImGuiSliderFlags_AlwaysClamp);
		ImGui::Checkbox("Auto spin", &state.spinning);
		ImGui::SliderFloat("Spin (deg/s)", &state.spin_speed_degrees, -180.0f, 180.0f);
	}

	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Albedo", glm::value_ptr(state.albedo));
		ImGui::ColorEdit3("Specular", glm::value_ptr(state.specular));
		ImGui::SliderFloat("Shininess", &state.shininess, 1.0f, 256.0f, "%.0f",
		                   ImGuiSliderFlags_Logarithmic);
	}

	if (ImGui::CollapsingHeader("Ambient and directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
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
			ImGui::PushID(i);   // у одинаковых подписей разных источников должны быть разные id
			ImGui::SeparatorText(("Point " + std::to_string(i)).c_str());
			ImGui::DragFloat3("Position", glm::value_ptr(light.position), 0.02f);
			ImGui::ColorEdit3("Color", glm::value_ptr(light.color));
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 100.0f);
			if (ImGui::Button("Remove")) {
				remove_index = i;
			}
			ImGui::PopID();
		}
		if (remove_index >= 0) {
			state.point_lights.erase(state.point_lights.begin() + remove_index);
		}
	}

	if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
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
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 200.0f);
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

	const Mesh mesh = makeTruncatedTetrahedron();
	index_count = uint32_t(mesh.indices.size());

	// Порядок важен: дескрипторам нужны буферы, конвейеру -- layout'ы наборов.
	if (!createBuffers(mesh) || !createDescriptors() || !createPipeline()) {
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	// Наборы дескрипторов освобождаются вместе с пулом.
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);

	vmaDestroyBuffer(context.allocator, model_buffer.buffer, model_buffer.allocation);
	vmaDestroyBuffer(context.allocator, spot_light_buffer.buffer, spot_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, point_light_buffer.buffer, point_light_buffer.allocation);
	vmaDestroyBuffer(context.allocator, scene_buffer.buffer, scene_buffer.allocation);
	vmaDestroyBuffer(context.allocator, index_buffer.buffer, index_buffer.allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer.buffer, vertex_buffer.allocation);
}

void update(double time) {
	float dt = 0.0f;
	if (has_last_time) {
		dt = std::min(float(time - last_time), 0.1f);
	}
	last_time = time;
	has_last_time = true;

	if (state.spinning) {
		state.spin_degrees += dt * state.spin_speed_degrees;
	}

	drawUserInterface();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkExtent2D extent = context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	// Обновляем содержимое буферов до записи команд.
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

	// Привязываем оба набора разом: firstSet = 0, значит sets[0] -> set 0, sets[1] -> set 1.
	const VkDescriptorSet sets[] = {scene_set, model_set};
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 2, sets, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
