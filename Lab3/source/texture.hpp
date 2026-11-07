#pragma once

#include <string>

#include "vk_utils.hpp"

// Текстура на GPU: изображение (VkImage) + его "вид" (VkImageView), который читает шейдер.
struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VkImageView view = VK_NULL_HANDLE;
	uint32_t mip_levels = 1;
};

// Создаёт сэмплер (правила чтения текстуры: фильтрация, повтор, мип-уровни),
// общий для всех текстур. Вызывать один раз; уничтожать через destroySampler().
bool createSampler(VkSampler& sampler);
void destroySampler(VkSampler sampler);

// Загружает изображение из файла (PNG, JPEG, BMP, TGA... -- через stb_image),
// копирует пиксели в VRAM и строит мип-уровни.
//   srgb = true  -- цветная текстура (albedo, emissive): шейдер получит линейные значения;
//   srgb = false -- данные (specular): значения берутся как есть.
bool loadTextureFromFile(const std::string& path, bool srgb, Texture& out);

// Текстура 1x1 с одним заданным пикселем. Заглушка вместо отсутствующих specular/emissive.
bool createSolidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a, bool srgb, Texture& out);

void destroyTexture(Texture& texture);
