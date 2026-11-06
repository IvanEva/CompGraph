#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>

#include "mesh.hpp"

namespace application {

namespace {

using graphics::internal::context;

// ---------------------------------------------------------------------------
// Константы сцены
// ---------------------------------------------------------------------------

constexpr float camera_distance = 6.0f;   // камера стоит в точке (0, 0, 6) и смотрит в начало координат
constexpr float near_plane = 0.1f;
constexpr float far_plane = 100.0f;
constexpr float ortho_half_height = 2.5f; // половина высоты видимой области для ортографической проекции

// ---------------------------------------------------------------------------
// Типы
// ---------------------------------------------------------------------------

// Данные, которые уходят в вершинный шейдер через push constants.
// Раскладка совпадает с блоком PushConstants в shaders/object.vert.
struct PushConstants {
	glm::mat4 mvp;
	glm::vec4 tint;
};
static_assert(sizeof(PushConstants) <= 128, "Vulkan guarantees only 128 bytes of push constants");

// Буфер Vulkan вместе с выделенной под него памятью (память выделяет VMA).
struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
};

// Всё, чем управляет пользователь через интерфейс.
struct State {
	// Проекция
	bool perspective = true;
	float fov_degrees = 45.0f;

	// Преобразования фигуры
	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation_degrees = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};

	// Цвет, на который умножается цвет каждой вершины
	glm::vec3 tint = {1.0f, 1.0f, 1.0f};

	// Анимация
	bool playing = true;
	float speed = 1.0f;                // общий множитель скорости
	float radius = 1.5f;               // радиус орбиты
	float height = 0.8f;               // амплитуда колебаний вверх-вниз
	int wave_count = 3;                // сколько волн помещается в один оборот
	float spin_speed_degrees = 60.0f;  // собственное вращение, градусов в секунду
	float anim_time = 0.0f;            // "время анимации": растёт только пока не нажата пауза
	float spin_degrees = 0.0f;         // накопленный угол собственного вращения
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

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// ---------------------------------------------------------------------------
// Вспомогательные функции Vulkan
// ---------------------------------------------------------------------------

// Создаёт буфер в памяти, доступной процессору, и копирует в него данные.
// Для небольшой фигуры этого достаточно (без промежуточного staging-буфера).
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

	if (vmaCopyMemoryToAllocation(context.allocator, data, out.allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to copy data into Vulkan buffer\n";
		return false;
	}

	return true;
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

	// SPIR-V -- массив 32-битных слов.
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

// Собирает графический конвейер: шейдеры + все настройки фиксированных стадий.
bool createPipeline() {
	// Layout конвейера: у нас нет дескрипторов, только один диапазон push constants.
	const VkPushConstantRange push_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		return false;
	}

	// Шейдеры. SHADER_DIR задаётся в CMakeLists.txt (папка shaders рядом с исходниками).
	VkShaderModule vert = VK_NULL_HANDLE;
	VkShaderModule frag = VK_NULL_HANDLE;

	if (!createShaderModule(std::string(SHADER_DIR) + "object.vert.spv", vert) ||
	    !createShaderModule(std::string(SHADER_DIR) + "object.frag.spv", frag)) {
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

	// Как читать вершинный буфер: одна привязка (binding 0), шаг -- sizeof(Vertex),
	// два атрибута: позиция (location 0) и цвет (location 1).
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
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
		.pVertexAttributeDescriptions = attributes,
	};

	// Каждые 3 индекса -- отдельный треугольник.
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	// Сами viewport и scissor задаём при записи команд (динамическое состояние),
	// поэтому конвейер не зависит от размера окна. Здесь только указываем, что их по одному.
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	// Лицевые грани -- те, что обходятся против часовой стрелки (если смотреть на экран).
	// Задние грани отсекаются.
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

	// Тест глубины: пиксель рисуется, только если он ближе к камере, чем уже нарисованный.
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	// Смешивание цветов выключено, пишем во все четыре канала.
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

	// Шейдерные модули после создания конвейера больше не нужны.
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

// Смещение фигуры по траектории в момент anim_time.
// Это "волнистая орбита": по кругу в плоскости XZ, при этом вверх-вниз по синусоиде.
// В момент 0 смещение нулевое, так что фигура стартует ровно из заданной позиции.
glm::vec3 trajectoryOffset() {
	const float t = state.anim_time;

	return {
		state.radius * std::sin(t),
		state.height * std::sin(float(state.wave_count) * t),
		state.radius * (std::cos(t) - 1.0f),
	};
}

// Матрица модели: переводит вершины из локальных координат фигуры в мировые.
// Порядок умножения читается "справа налево": сначала масштаб, потом повороты, потом перенос.
glm::mat4 modelMatrix() {
	const glm::vec3 spin_axis = glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));

	glm::mat4 model = glm::translate(glm::mat4(1.0f), state.position + trajectoryOffset());
	model = glm::rotate(model, glm::radians(state.spin_degrees), spin_axis);
	model = glm::rotate(model, glm::radians(state.rotation_degrees.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::rotate(model, glm::radians(state.rotation_degrees.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(state.rotation_degrees.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::scale(model, state.scale);
	return model;
}

// Матрица вида: камера в точке (0, 0, camera_distance), смотрит в начало координат.
glm::mat4 viewMatrix() {
	return glm::lookAt(glm::vec3(0.0f, 0.0f, camera_distance),
	                   glm::vec3(0.0f, 0.0f, 0.0f),
	                   glm::vec3(0.0f, 1.0f, 0.0f));
}

// Матрица проекции: перспективная или ортографическая, в зависимости от переключателя.
glm::mat4 projectionMatrix(float aspect) {
	glm::mat4 projection(1.0f);

	if (state.perspective) {
		projection = glm::perspective(glm::radians(state.fov_degrees), aspect, near_plane, far_plane);
	} else {
		const float half_width = ortho_half_height * aspect;
		projection = glm::ortho(-half_width, half_width,
		                        -ortho_half_height, ortho_half_height,
		                        near_plane, far_plane);
	}

	// В Vulkan ось Y экрана направлена вниз (в OpenGL, для которого придуманы
	// формулы GLM, -- вверх). Переворачиваем Y, чтобы картинка не была перевёрнутой.
	projection[1][1] *= -1.0f;

	return projection;
}

// ---------------------------------------------------------------------------
// Интерфейс
// ---------------------------------------------------------------------------

void drawUserInterface() {
	// Положение и ширина окна при первом запуске (высота 0 -- по содержимому).
	ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380.0f, 0.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Controls");

	ImGui::SeparatorText("Projection");
	if (ImGui::RadioButton("Perspective", state.perspective)) {
		state.perspective = true;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton("Orthographic", !state.perspective)) {
		state.perspective = false;
	}
	if (state.perspective) {
		ImGui::SliderFloat("FOV (deg)", &state.fov_degrees, 20.0f, 90.0f);
	}

	ImGui::SeparatorText("Transform");
	ImGui::DragFloat3("Position", glm::value_ptr(state.position), 0.01f);
	ImGui::DragFloat3("Rotation (deg)", glm::value_ptr(state.rotation_degrees), 0.5f);
	ImGui::DragFloat3("Scale", glm::value_ptr(state.scale), 0.01f, 0.05f, 5.0f, "%.2f",
	                  ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", glm::value_ptr(state.tint));

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(state.playing ? "Pause" : "Play")) {
		state.playing = !state.playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
		state.anim_time = 0.0f;
		state.spin_degrees = 0.0f;
	}
	ImGui::SliderFloat("Speed", &state.speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Orbit radius", &state.radius, 0.0f, 3.0f);
	ImGui::SliderFloat("Wave height", &state.height, 0.0f, 2.0f);
	ImGui::SliderInt("Waves per orbit", &state.wave_count, 1, 8);
	ImGui::SliderFloat("Spin (deg/s)", &state.spin_speed_degrees, -360.0f, 360.0f);

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
	// Не сохранять положение окон интерфейса в imgui.ini
	// (иначе рядом с программой появляется лишний файл).
	ImGui::GetIO().IniFilename = nullptr;

	const Mesh mesh = makeTruncatedTetrahedron();
	index_count = uint32_t(mesh.indices.size());

	if (!createBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !createBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
	                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer) ||
	    !createPipeline()) {
		// main() при неудаче initialize() наш shutdown() не вызывает,
		// поэтому всё созданное успеваем убрать здесь.
		shutdown();
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	// Уничтожение VK_NULL_HANDLE разрешено, так что функция безопасна
	// и после частично неудавшейся инициализации.
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);

	vmaDestroyBuffer(context.allocator, index_buffer.buffer, index_buffer.allocation);
	vmaDestroyBuffer(context.allocator, vertex_buffer.buffer, vertex_buffer.allocation);
}

void update(double time) {
	// dt -- сколько секунд прошло с прошлого кадра. Ограничиваем сверху,
	// чтобы после долгой задержки (например, при перетаскивании окна) фигура не прыгала.
	float dt = 0.0f;
	if (has_last_time) {
		dt = std::min(float(time - last_time), 0.1f);
	}
	last_time = time;
	has_last_time = true;

	// Время анимации копится только пока нет паузы,
	// поэтому после "Play" фигура продолжает с того же места.
	if (state.playing) {
		state.anim_time += dt * state.speed;
		state.spin_degrees += dt * state.speed * state.spin_speed_degrees;
	}

	drawUserInterface();
}

void render(const graphics::internal::FrameData& fd) {
	// Если не удалось получить кадр, рисовать нечего.
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	const VkExtent2D extent = context.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	// Итоговая матрица: сначала модель, потом вид, потом проекция (читаем справа налево).
	const PushConstants constants = {
		.mvp = projectionMatrix(aspect) * viewMatrix() * modelMatrix(),
		.tint = glm::vec4(state.tint, 1.0f),
	};

	VkCommandBuffer cmd = fd.command_buffer;

	vkResetCommandBuffer(cmd, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	// Render pass из graphics_internal.cpp имеет два вложения: цвет (0) и глубина (1).
	// Оба очищаются в начале прохода, поэтому нужно два значения очистки.
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

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &vertex_offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
	                   sizeof(PushConstants), &constants);

	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
