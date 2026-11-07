#include "texture.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

// Реализация stb_image подключается ровно в одном .cpp-файле.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_TGA
#include <stb_image.h>

namespace {

using graphics::internal::context;

// Создаёт изображение нужного размера в VRAM, заполняет его пикселями и строит мип-уровни.
//   pixels -- RGBA8, width * height * 4 байт.
bool uploadImage(const uint8_t* pixels, uint32_t width, uint32_t height, VkFormat format, Texture& out) {
	// Сколько уровней уменьшения: 1024 -> 512 -> ... -> 1.
	uint32_t mip_levels = uint32_t(std::floor(std::log2(float(std::max(width, height))))) + 1;

	// Мипы строятся командой blit с линейной фильтрацией. Если формат её не умеет -- обходимся одним уровнем.
	VkFormatProperties format_properties;
	vkGetPhysicalDeviceFormatProperties(context.physical_device, format, &format_properties);
	if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ||
	    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) ||
	    !(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
		mip_levels = 1;
	}
	out.mip_levels = mip_levels;

	// 1. Staging-буфер с пикселями.
	const VkDeviceSize size = VkDeviceSize(width) * height * 4;
	vkutil::Buffer staging;
	if (!vkutil::createHostBuffer(pixels, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) {
		vkutil::destroyBuffer(staging);
		return false;
	}

	// 2. Само изображение. Оно и источник (для построения мипов), и приёмник, и читается шейдером.
	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = format,
		.extent = {width, height, 1},
		.mipLevels = mip_levels,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};

	if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
	                   &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image\n";
		vkutil::destroyBuffer(staging);
		return false;
	}

	// 3. Команды GPU: копирование + мипы + перевод в состояние "для чтения шейдером".
	const bool ok = vkutil::submitOneTime([&](VkCommandBuffer cmd) {
		// Удобная функция: барьер для диапазона мип-уровней (меняет раскладку изображения в памяти).
		auto barrier = [&](uint32_t base_mip, uint32_t count,
		                   VkImageLayout old_layout, VkImageLayout new_layout,
		                   VkAccessFlags src_access, VkAccessFlags dst_access,
		                   VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
			const VkImageMemoryBarrier b = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
				.srcAccessMask = src_access,
				.dstAccessMask = dst_access,
				.oldLayout = old_layout,
				.newLayout = new_layout,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = out.image,
				.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base_mip, count, 0, 1},
			};
			vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
		};

		// Все уровни -> "приёмник копирования".
		barrier(0, mip_levels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		        0, VK_ACCESS_TRANSFER_WRITE_BIT,
		        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

		// Копируем пиксели из staging-буфера в уровень 0.
		const VkBufferImageCopy region = {
			.bufferOffset = 0,
			.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
			.imageExtent = {width, height, 1},
		};
		vkCmdCopyBufferToImage(cmd, staging.buffer, out.image,
		                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		// Каждый следующий уровень = предыдущий, уменьшенный вдвое (blit с линейной фильтрацией).
		int32_t mip_width = int32_t(width);
		int32_t mip_height = int32_t(height);
		for (uint32_t level = 1; level < mip_levels; ++level) {
			// Предыдущий уровень готов: делаем его источником для blit.
			barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
			        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

			const int32_t next_width = std::max(mip_width / 2, 1);
			const int32_t next_height = std::max(mip_height / 2, 1);

			const VkImageBlit blit = {
				.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
				.srcOffsets = {{0, 0, 0}, {mip_width, mip_height, 1}},
				.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
				.dstOffsets = {{0, 0, 0}, {next_width, next_height, 1}},
			};
			vkCmdBlitImage(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			               out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			               1, &blit, VK_FILTER_LINEAR);

			// Предыдущий уровень больше не нужен для blit: отдаём его шейдеру.
			barrier(level - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
			        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

			mip_width = next_width;
			mip_height = next_height;
		}

		// Последний уровень остался в "приёмнике копирования" -- тоже отдаём шейдеру.
		barrier(mip_levels - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
		        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
	});

	vkutil::destroyBuffer(staging);
	if (!ok) {
		return false;
	}

	// 4. "Вид" изображения: именно его (вместе с сэмплером) привязываем к дескриптору.
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = out.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = format,
		.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mip_levels, 0, 1},
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &out.view) != VK_SUCCESS) {
		std::cerr << "Failed to create image view\n";
		return false;
	}

	return true;
}

} // namespace

bool createSampler(VkSampler& sampler) {
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,                    // при увеличении -- билинейная
		.minFilter = VK_FILTER_LINEAR,                    // при уменьшении -- билинейная
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,      // плюс смешивание мип-уровней = трилинейная
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,   // за границей [0;1] текстура повторяется
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.minLod = 0.0f,
		.maxLod = VK_LOD_CLAMP_NONE,                      // использовать все доступные мип-уровни
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
	};

	if (vkCreateSampler(context.device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create sampler\n";
		return false;
	}
	return true;
}

void destroySampler(VkSampler sampler) {
	vkDestroySampler(context.device, sampler, nullptr);
}

bool loadTextureFromFile(const std::string& path, bool srgb, Texture& out) {
	int width = 0;
	int height = 0;
	int channels = 0;

	// Последний аргумент 4: что бы ни было в файле, получаем RGBA8 (4 байта на пиксель).
	stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
	if (pixels == nullptr) {
		std::cerr << "Failed to load image: " << path << " (" << stbi_failure_reason() << ")\n";
		return false;
	}

	const VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
	const bool ok = uploadImage(pixels, uint32_t(width), uint32_t(height), format, out);

	stbi_image_free(pixels);
	return ok;
}

bool createSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb, Texture& out) {
	const uint8_t pixel[4] = {r, g, b, a};
	const VkFormat format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
	return uploadImage(pixel, 1, 1, format, out);
}

void destroyTexture(Texture& texture) {
	vkDestroyImageView(context.device, texture.view, nullptr);
	if (texture.image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}
	texture = Texture{};
}
