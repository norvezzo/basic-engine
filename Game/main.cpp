#include "InputManager.h"
// #include "../DisplayGLFW/display.h"
#include "game.h"
#include "../res/includes/glm/glm.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include <iostream>

const unsigned char STRONG_EDGE = 255;
const unsigned char WEAK_EDGE = 100;
const unsigned char NON_EDGE = 0;

unsigned char toGrayscale(unsigned char red, unsigned char green, unsigned char blue) {
	return static_cast<unsigned char>(0.299 * red + 0.587 * green + 0.114 * blue);
}

void grayScale(unsigned char* imgData, int width, int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 4;
			unsigned char red = imgData[index];
			unsigned char green = imgData[index + 1];
			unsigned char blue = imgData[index + 2];
			unsigned char gray = toGrayscale(red, green, blue);
			imgData[index] = gray;
			imgData[index + 1] = gray;
			imgData[index + 2] = gray;
		}
	}
}

float* generateGaussianKernel(int kernelSize, float sigma) {
	float* kernel = new float[kernelSize * kernelSize];
	float sum = 0.0; // for normalization
	int k = kernelSize / 2;

	for (int i = -k; i <= k; i++) {
		for (int j = -k; j <= k; j++) {
			float sqDist = i * i + j * j;
			int index = (i + k) * kernelSize + (j + k);
			kernel[index] = exp(-sqDist / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			sum += kernel[index];
		}
	}
	for (int i = 0; i < kernelSize * kernelSize; i++)
		kernel[i] /= sum;

	return kernel;
}

void applyGaussianFilter(unsigned char* imgData, int width, int height, float* kernel, int kernelSize) {
	unsigned char* output = new unsigned char[width * height * 4];
	int k = kernelSize / 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float sum = 0.0;
			for (int dy = -k; dy <= k; dy++) {
				for (int dx = -k; dx <= k; dx++) {
					int iy = y + dy;
					int ix = x + dx;
					if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
						int index = (iy * width + ix) * 4;
						int kernelIndex = (dy + k) * kernelSize + (dx + k);
						sum += imgData[index] * kernel[kernelIndex];
					}
				}
			}
			int outputIndex = (y * width + x) * 4;
			output[outputIndex] = static_cast<unsigned char>(sum);
			output[outputIndex + 1] = output[outputIndex];
			output[outputIndex + 2] = output[outputIndex];
			output[outputIndex + 3] = 255;
		}
	}
	std::memcpy(imgData, output, width * height * 4);
	delete[] output;
}

void findIntensityGradient(unsigned char* imgData, int width, int height, float* gradientMagnitude, float* gradientDirection) {
	const int GX[3][3] = {
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1}
	};
	const int GY[3][3] = {
		{1, 2, 1},
		{0, 0, 0},
		{-1, -2, -1}
	};
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			float gx = 0, gy = 0;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int ix = x + dx;
					int iy = y + dy;
					int index = (iy * width + ix) * 4;
					unsigned char pixelValue = imgData[index];
					gx += pixelValue * GX[dy + 1][dx + 1];
					gy += pixelValue * GY[dy + 1][dx + 1];
				}
			}
			int index = y * width + x;
			gradientMagnitude[index] = sqrt(gx * gx + gy * gy);
			gradientDirection[index] = atan2(gy, gx);
		}
	}
}

void applyGradientMagnitudeThresholding(float* gradientMagnitude, int width, int height, float threshold) {
	int size = width * height;
	for (int i = 0; i < size; i++)
		if (gradientMagnitude[i] < threshold)
			gradientMagnitude[i] = 0;
}

float findMaxGradientMagnitude(float* gradientMagnitude, int size) {
	float maxVal = 0;
	for (int i = 0; i < size; i++)
		if (gradientMagnitude[i] > maxVal)
			maxVal = gradientMagnitude[i];

	return maxVal;
}

float calculateThreshold(float* gradientMagnitude, int size) {
	float maxGradient = findMaxGradientMagnitude(gradientMagnitude, size);
	return maxGradient * 0.2;
}

void applyDoubleThreshold(float* gradientMagnitude, unsigned char* edges, int width, int height, float lowThreshold, float highThreshold) {
	int size = width * height;
	for (int i = 0; i < size; i++) {
		if (gradientMagnitude[i] >= highThreshold) {
			edges[i] = STRONG_EDGE;
		}
		else if (gradientMagnitude[i] >= lowThreshold) {
			edges[i] = WEAK_EDGE;
		}
		else {
			edges[i] = NON_EDGE;
		}
	}
}

void trackEdgesByHysteresis(unsigned char* edges, int width, int height) {
	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			int index = y * width + x;
			if (edges[index] == WEAK_EDGE) {
				bool connectedToStrongEdge = false;
				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						if (dx == 0 && dy == 0) continue; //skip the center pixel, only checking 8 neighbors
						int neighborIndex = (y + dy) * width + (x + dx);
						if (edges[neighborIndex] == STRONG_EDGE) {
							connectedToStrongEdge = true;
							break;
						}
					}
					if (connectedToStrongEdge) break;
				}
				if (connectedToStrongEdge) {
					edges[index] = STRONG_EDGE;
				}
				else {
					edges[index] = NON_EDGE;
				}
			}
		}
	}
}

void cannyEdge(unsigned char* imgData, int width, int height) {
	int kernelSize = 5;
	float sigma = 1.0;
	float* kernel = generateGaussianKernel(kernelSize, sigma);
	applyGaussianFilter(imgData, width, height, kernel, kernelSize);

	float* gradientMagnitude = new float[width * height];
	float* gradientDirection = new float[width * height];
	findIntensityGradient(imgData, width, height, gradientMagnitude, gradientDirection);
	float threshold = calculateThreshold(gradientMagnitude, width * height);

	applyGradientMagnitudeThresholding(gradientMagnitude, width, height, threshold);

	float maxGradientMagnitude = findMaxGradientMagnitude(gradientMagnitude, width * height);
	float highThreshold = maxGradientMagnitude * 0.15;
	float lowThreshold = maxGradientMagnitude * 0.05;
	unsigned char* edges = new unsigned char[width * height];
	applyDoubleThreshold(gradientMagnitude, edges, width, height, lowThreshold, highThreshold);

	trackEdgesByHysteresis(edges, width, height);

	for (int i = 0; i < width * height; i++) {
		if (edges[i] == STRONG_EDGE) {
			imgData[i * 4] = 255;
			imgData[i * 4 + 1] = 255;
			imgData[i * 4 + 2] = 255;
			imgData[i * 4 + 3] = 255;
		}
		else {
			imgData[i * 4] = 0;
			imgData[i * 4 + 1] = 0;
			imgData[i * 4 + 2] = 0;
			imgData[i * 4 + 3] = 255;
		}
	}

	delete[] kernel;
	delete[] gradientMagnitude;
	delete[] gradientDirection;
	delete[] edges;
}

void halftonePattern(unsigned char* imgData, int width, int height) {
	for (int y = 0; y < height; y += 2) {
		for (int x = 0; x < width; x += 2) {
			int totalIntensity = 0;
			for (int dy = 0; dy < 2; dy++) {
				for (int dx = 0; dx < 2; dx++) {
					if (x + dx < width && y + dy < height) {
						int index = ((y + dy) * width + (x + dx)) * 4;
						totalIntensity += imgData[index];
					}
				}
			}
			float avgIntensity = totalIntensity / 4.0;
			int blackSubPixels = (256 - avgIntensity) / 48;
			for (int dy = 0; dy < 2; dy++) {
				for (int dx = 0; dx < 2; dx++) {
					if (x + dx < width && y + dy < height) {
						int subIndex = ((y + dy) * width + (x + dx)) * 4;
						if (blackSubPixels > 0) {
							imgData[subIndex] = imgData[subIndex + 1] = imgData[subIndex + 2] = 0;
							blackSubPixels--;
						}
						else {
							imgData[subIndex] = imgData[subIndex + 1] = imgData[subIndex + 2] = 255;
						}
						imgData[subIndex + 3] = 255;
					}
				}
			}
		}
	}
}

unsigned char closestColor(unsigned char pixelValue) {
	int level = pixelValue / 16;
	return static_cast<unsigned char>(level * 16);
}

void floydSteinberg(unsigned char* imgData, int width, int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 4;
			unsigned char oldPixel = imgData[index];
			unsigned char newPixel = closestColor(oldPixel);
			imgData[index] = imgData[index + 1] = imgData[index + 2] = newPixel;
			int quantError = oldPixel - newPixel;
			if (x + 1 < width) {
				imgData[index + 4] += quantError * 7 / 16;
			}
			if (x - 1 >= 0 && y + 1 < height) {
				imgData[index + (width - 1) * 4] += quantError * 3 / 16;
			}
			if (y + 1 < height) {
				imgData[index + width * 4] += quantError * 5 / 16;
			}
			if (x + 1 < width && y + 1 < height) {
				imgData[index + (width + 1) * 4] += quantError * 1 / 16;
			}
		}
	}
}

void writeImage(const unsigned char* imgData, int width, int height, const std::string& filename, bool isGrayscale) {
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Failed to open " << filename << std::endl;
		return;
	}
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x) * 4;
			unsigned char value = imgData[index];
			int outputValue;
			if (isGrayscale) {
				outputValue = (value * 15) / 255;
			}
			else {
				outputValue = value > 127 ? 1 : 0;
			}
			file << outputValue;
			if ((y == height - 1) && (x == width - 1)) break;
			file << ",";
		}
	}
	file.close();
}

int main(int argc,char *argv[])
{
	const int DISPLAY_WIDTH = 512;
	const int DISPLAY_HEIGHT = 512;
	const float CAMERA_ANGLE = 0.0f;
	const float NEAR = 1.0f;
	const float FAR = 100.0f;

	Game *scn = new Game(CAMERA_ANGLE,(float)DISPLAY_WIDTH/DISPLAY_HEIGHT,NEAR,FAR);
	
	Display display(DISPLAY_WIDTH, DISPLAY_HEIGHT, "OpenGL");
	
	Init(display);
	
	scn->Init();

	display.SetScene(scn);

	int width, height, channels;
	unsigned char* imgData = stbi_load("../res/textures/lena256.jpg", &width, &height, &channels, 4);

	int imgSize = width * height * 4;
	unsigned char* copy = new unsigned char[imgSize];

	grayScale(imgData, width, height);
	std::memcpy(copy, imgData, imgSize);
	scn->AddTexture(width, height, copy);
	scn->SetShapeTex(0, 0);
	scn->CustomDraw(1, 0, scn->BACK, true, false, 0);

	std::memcpy(copy, imgData, imgSize);
	cannyEdge(copy, width, height);
	scn->AddTexture(width, height, copy);
	scn->SetShapeTex(0, 1);
	scn->CustomDraw(1, 0, scn->BACK, false, false, 1);
	writeImage(copy, width, height, "img4.txt", false);

	std::memcpy(copy, imgData, imgSize);
	halftonePattern(copy, width, height);
	scn->AddTexture(width, height, copy);
	scn->SetShapeTex(0, 2);
	scn->CustomDraw(1, 0, scn->BACK, false, false, 2);
	writeImage(copy, width, height, "img5.txt", false);

	std::memcpy(copy, imgData, imgSize);
	floydSteinberg(copy, width, height);
	scn->AddTexture(width, height, copy);
	scn->SetShapeTex(0, 3);
	scn->CustomDraw(1, 0, scn->BACK, false, false, 3);
	writeImage(copy, width, height, "img6.txt", true);

	scn->Motion();
	display.SwapBuffers();

	while(!display.CloseWindow())
	{
		//scn->Draw(1,0,scn->BACK,true,false);
		//scn->Motion();
		//display.SwapBuffers();
		display.PollEvents();	
			
	}
	delete scn;
	stbi_image_free(imgData);
	delete[] copy;
	return 0;
}

