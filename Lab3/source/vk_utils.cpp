#include "vk_utils.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace vkutil {

namespace {

using graphics::internal::context;

VkCommandPool command_pool = VK_NULL_HANDLE;

} // namespace

bool initialize() {
	const VkCommandPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,   // буферы живут очень недолго
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create command pool for one-time commands\n";
		return false;
	}

	return true;
}

void shutdown() {
	vkDestroyCommandPool(context.device, command_pool, nullptr);
	command_pool = VK_NULL_HANDLE;
}

bool createHostBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	// HOST_ACCESS_SEQUENTIAL_WRITE: процессор будет только писать, последовательно.
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

void updateHostBuffer(const Buffer& buffer, const void* data, VkDeviceSize size, VkDeviceSize offset) {
	vmaCopyMemoryToAllocation(context.allocator, data, buffer.allocation, offset, size);
}

bool createDeviceBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& out) {
	// 1. Staging-буфер: виден процессору, служит только источником копирования.
	Buffer staging;
	if (!createHostBuffer(data, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		destroyBuffer(staging);
		return false;
	}

	// 2. Целевой буфер: без флагов доступа процессору, поэтому VMA выберет VRAM.
	//    TRANSFER_DST нужен, чтобы в него можно было копировать.
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &out.buffer, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device-local Vulkan buffer\n";
		destroyBuffer(staging);
		return false;
	}

	// 3. Команда GPU: скопировать из staging в целевой буфер. Ждём её завершения.
	const bool ok = submitOneTime([&](VkCommandBuffer cmd) {
		const VkBufferCopy region = { .srcOffset = 0, .dstOffset = 0, .size = size };
		vkCmdCopyBuffer(cmd, staging.buffer, out.buffer, 1, &region);
	});

	// 4. Staging больше не нужен.
	destroyBuffer(staging);
	return ok;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = Buffer{};
}

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

bool submitOneTime(const std::function<void(VkCommandBuffer)>& record) {
	const VkCommandBufferAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(context.device, &allocate_info, &cmd) != VK_SUCCESS) {
		std::cerr << "Failed to allocate one-time command buffer\n";
		return false;
	}

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);
	record(cmd);
	vkEndCommandBuffer(cmd);

	const VkSubmitInfo submit_info = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};

	// Для простоты ждём, пока очередь не станет пустой: загрузка идёт один раз при старте.
	const bool ok = vkQueueSubmit(context.graphics_queue, 1, &submit_info, VK_NULL_HANDLE) == VK_SUCCESS &&
	                vkQueueWaitIdle(context.graphics_queue) == VK_SUCCESS;
	if (!ok) {
		std::cerr << "Failed to submit one-time commands\n";
	}

	vkFreeCommandBuffers(context.device, command_pool, 1, &cmd);
	return ok;
}

} // namespace vkutil
