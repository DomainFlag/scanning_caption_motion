#include <iostream>
#include <fstream>
#include <array>
#include <math.h>
#include <sstream>

#include "Eigen.h"

#include "VirtualSensor.h"

struct Vertex
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	// position stored as 4 floats (4th component is supposed to be 1.0)
	Vector4f position;
	// color stored as 4 unsigned char
	Vector4uc color;
};

float ComputeDistance(Vertex& v1, Vertex& v2) {
	if (v1.position.size() != v2.position.size())
		return -1;

	float acc = 0.0f;
	for (int g = 0; g < v1.position.size(); g++) {
		acc += powf(v1.position[g] - v2.position[g], 2.0f);
	}

	return sqrt(acc);
}

int WriteFace(Vertex* vertices, std::stringstream & ss, float & edgeThreshold, unsigned int v1, unsigned int v2, unsigned int v3)
{
	if (ComputeDistance(vertices[v1], vertices[v2]) > edgeThreshold || ComputeDistance(vertices[v2], vertices[v3]) > edgeThreshold ||
		ComputeDistance(vertices[v1], vertices[v3]) > edgeThreshold) {
		return 0;
	}

	ss << 3 << " " << v1 << " " << v2 << " " << v3 << std::endl;

	return 1;
}

bool WriteMesh(Vertex* vertices, unsigned int width, unsigned int height, const std::string& filename)
{
	float edgeThreshold = 0.1f; // 10cm

	// TODO 2: use the OFF file format to save the vertices grid (http://www.geomview.org/docs/html/OFF.html)
	// - have a look at the "off_sample.off" file to see how to store the vertices and triangles
	// - for debugging we recommend to first only write out the vertices (set the number of faces to zero)
	// - for simplicity write every vertex to file, even if it is not valid (position.x() == MINF) (note that all vertices in the off file have to be valid, thus, if a point is not valid write out a dummy point like (0,0,0))
	// - use a simple triangulation exploiting the grid structure (neighboring vertices build a triangle, two triangles per grid cell)
	// - you can use an arbitrary triangulation of the cells, but make sure that the triangles are consistently oriented
	// - only write triangles with valid vertices and an edge length smaller then edgeThreshold

	// TODO: Get number of vertices
	unsigned int nVertices = width * height;

	// TODO: Determine number of valid faces
	unsigned int nFaces = 0;

	std::stringstream ss;
    for (unsigned int idw = 0 ; idw < width-1; idw++) {
        for (unsigned int idh = 0; idh < height - 1; idh++) {

            unsigned int topLeftIndex = idh * width + idw;
            unsigned int bottomLeftIndex = topLeftIndex + width;

            nFaces += WriteFace(vertices, ss, edgeThreshold, topLeftIndex, topLeftIndex + 1, bottomLeftIndex + 1);
            nFaces += WriteFace(vertices, ss, edgeThreshold, topLeftIndex, bottomLeftIndex + 1, bottomLeftIndex);
        }
    }

	// Write off file
	std::ofstream outFile(filename);
	if (!outFile.is_open()) return false;

	// write header
	outFile << "COFF" << std::endl;
	outFile << nVertices << " " << nFaces << " 0" << std::endl;

	// TODO: save vertices
	outFile << "# list of vertices" << std::endl;

	for (unsigned int g = 0; g < nVertices; g++) {
		if (vertices[g].position(0) == MINF) {
			outFile << "0 0 0 255 255 255 255" << std::endl;
		} else {
			for (int h = 0; h < 3; h++) {
				outFile << (vertices[g].position(h) / vertices[g].position(3))  << " ";
			}

			for (int h = 0; h < 4; h++) {
				outFile << (unsigned int) vertices[g].color(h) << " ";
			}

			outFile << std::endl;
		}
	}

	// TODO: save valid faces
	outFile << "# list of faces" << std::endl;
	outFile << ss.str();

	// close file
	outFile.close();

	return true;
}

int main()
{
	// Make sure this path points to the data folder
	std::string filenameIn = "../../data/rgbd_dataset_freiburg1_xyz/";
	std::string filenameBaseOut = "mesh_";

	// load video
	std::cout << "Initialize virtual sensor..." << std::endl;
	VirtualSensor sensor;
	if (!sensor.Init(filenameIn))
	{
		std::cout << "Failed to initialize the sensor!\nCheck file path!" << std::endl;
		return -1;
	}

	// convert video to meshes
	while (sensor.ProcessNextFrame())
	{
		// get ptr to the current depth frame
		// depth is stored in row major (get dimensions via sensor.GetDepthImageWidth() / GetDepthImageHeight())
		float* depthMap = sensor.GetDepth();
		// get ptr to the current color frame
		// color is stored as RGBX in row major (4 byte values per pixel, get dimensions via sensor.GetColorImageWidth() / GetColorImageHeight())
		BYTE* colorMap = sensor.GetColorRGBX();

		// get depth intrinsics
		Matrix3f depthIntrinsics = sensor.GetDepthIntrinsics();
		float fX = depthIntrinsics(0, 0);
		float fY = depthIntrinsics(1, 1);
		float cX = depthIntrinsics(0, 2);
		float cY = depthIntrinsics(1, 2);

		// compute inverse depth extrinsics
		Matrix3f depthIntrinsicsInv = sensor.GetDepthIntrinsics().inverse();
		Matrix4f depthExtrinsicsInv = sensor.GetDepthExtrinsics().inverse();

		Matrix4f trajectory = sensor.GetTrajectory();
		Matrix4f trajectoryInv = sensor.GetTrajectory().inverse();

		// TODO 1: back-projection
		// write result to the vertices array below, keep pixel ordering!
		// if the depth value at idx is invalid (MINF) write the following values to the vertices array
		// vertices[idx].position = Vector4f(MINF, MINF, MINF, MINF);
		// vertices[idx].color = Vector4uc(0,0,0,0);
		// otherwise apply back-projection and transform the vertex to world space, use the corresponding color from the colormap
		unsigned int totalSize = sensor.GetDepthImageWidth() * sensor.GetDepthImageHeight();

		unsigned int offsetWidthSize = sensor.GetDepthImageWidth() / 2;
		unsigned int offsetHeightSize = sensor.GetDepthImageHeight() / 2;

		Vertex* vertices = new Vertex[totalSize];
		for(unsigned int g = 0; g < totalSize; g++) {
		    float depth = depthMap[g];

			if (depth == MINF) {
				vertices[g].position = Vector4f(MINF, MINF, MINF, MINF);
				vertices[g].color = Vector4uc(0, 0, 0, 0);
			} else {
				// Compute vertex position
				float x = g % sensor.GetDepthImageWidth();
				float y = (float) (g - x) / sensor.GetDepthImageHeight();
				Vector3f imagePixels = Vector3f((x - offsetWidthSize) * depth, (y - offsetHeightSize) * depth, depth);

				// Compute camera position
				Vector3f imageCamera = depthIntrinsicsInv * imagePixels;

				// Compute world position
				vertices[g].position = trajectoryInv * depthExtrinsicsInv * Vector4f(imageCamera(0), imageCamera(1), imageCamera(2), 1.0f);

				// Compute vertex color
				unsigned int colorIndex = g * 4;
				vertices[g].color = Vector4uc(colorMap[colorIndex], colorMap[colorIndex + 1], colorMap[colorIndex + 2], colorMap[colorIndex + 3]);
			}
		}

		// write mesh file
		std::stringstream ss;
		ss << filenameBaseOut << sensor.GetCurrentFrameCnt() << ".off";
		if (!WriteMesh(vertices, sensor.GetDepthImageWidth(), sensor.GetDepthImageHeight(), ss.str()))
		{
			std::cout << "Failed to write mesh!\nCheck file path!" << std::endl;
			return -1;
		}

		// free mem
		delete[] vertices;
	}

	return 0;
}
