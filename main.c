#include <stdint.h>
#include <ft2build.h>
#include FT_FREETYPE_H
#include <tinymsdf.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

int main(int argc, const char **argv) {
#define ABORT(msg) { printf(msg); return 1; }
	
	const char *input = "D:\\Workspaces\\Mattparks\\tinymsdf\\fonts\\Roboto-Regular.ttf";
	const char *output = "output.png";
	const char *testRender = "test.png";
	unicode_t unicode = 'p';
	int width = 26, height = 26;
	int testWidth = 512, testHeight = 512;

	FT_Library library;
	if (FT_Init_FreeType(&library)) {
		ABORT("Failed to initalize Freetype");
	}
	FT_Face face;
	if (FT_New_Face(library, input, 0, &face)) {
		FT_Done_FreeType(library);
		ABORT("Failed to load font file");
	}

	float *pixels = NULL;
	if (tinymsdf_generate_mtsdf(&pixels, width, height, face, unicode)) {
		FT_Done_Face(face);
		FT_Done_FreeType(library);
		ABORT("Failed to load glyph from font file");
	}

	FT_Done_Face(face);
	FT_Done_FreeType(library);

	uint8_t *bitmap = malloc(4 * width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			size_t index = 4 * ((y * height) + x);
			bitmap[index + 0] = (uint8_t)(255 * (pixels[index + 0] + height) / height);
			bitmap[index + 1] = (uint8_t)(255 * (pixels[index + 1] + height) / height);
			bitmap[index + 2] = (uint8_t)(255 * (pixels[index + 2] + height) / height);
			bitmap[index + 3] = (uint8_t)(255 * (pixels[index + 3] + height) / height);
		}
	}
	stbi_write_png(output, width, height, 4, bitmap, width * 4);
	free(bitmap);

	free(pixels);
	return 0;
}
