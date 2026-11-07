#pragma once

#include <functional>
#include <string>

#include "graphics_internal.hpp"

// Мелкие помощники для работы с Vulkan: буферы, шейдеры и одноразовые команды GPU.
// Вынесены из application.cpp, чтобы там остались только сцена и отрисовка.
namespace vkutil {

// Буфер Vulkan вместе с выделенной под него памятью (память выделяет VMA).
struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
};

// Создаёт пул команд для одноразовых операций (копирование в VRAM и т.п.).
bool initialize();
void shutdown();

// Буфер в памяти, ДОСТУПНОЙ ПРОЦЕССОРУ (host visible). Подходит для данных,
// которые меняются каждый кадр (uniform-буферы). data может быть nullptr.
bool createHostBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out);

// Перезаписывает часть host-буфера (вызывать можно каждый кадр).
void updateHostBuffer(const Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset = 0);

// Буфер в видеопамяти (VRAM). Процессор туда писать напрямую не может, поэтому данные
// сначала кладутся в промежуточный staging-буфер, а потом GPU копирует их командой.
bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out);

void destroyBuffer(Buffer& buffer);

// Читает скомпилированный шейдер (SPIR-V) из файла и создаёт VkShaderModule.
bool createShaderModule(const std::string& path, VkShaderModule& module);

// Записывает команды в одноразовый буфер команд, отправляет их на GPU и ждёт завершения.
bool submitOneTime(const std::function<void(VkCommandBuffer)>& record);

} // namespace vkutil
