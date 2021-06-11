#define GLM_ENABLE_EXPERIMENTAL
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <arrrgh.hpp>
#include <experimental/filesystem>
#include <fstream>
#include <random>
#include <set>
#include <lodepng.h>
#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/SurfaceRemeshing.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/utilities/fileutils.h>
#include <shapeDescriptor/utilities/read/MeshLoader.h>
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/dump/meshDumper.h>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/utilities/kernels/spinOriginBufferGenerator.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/dump/descriptorImages.h>
#include <projectSymmetry/visualisation/GLUtils.h>
#include <projectSymmetry/visualisation/VAOGenerator.h>
#include <projectSymmetry/visualisation/shader.hpp>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <projectSymmetry/descriptors/binaryRICIConverter.h>


using namespace std::chrono_literals;

int main(int argc, const char** argv) {
    arrrgh::parser parser("querysetgenerator", "Create partial query meshes from an object set set.");
    const auto& objectDirectory = parser.add<std::string>(
            "object-directory", "The directory containing objects from which the partial queries should be generated.", '\0', arrrgh::Required, "");
    const auto& targetDirectory = parser.add<std::string>(
            "output-directory", "The directory where the partial queries should be written to.", '\0', arrrgh::Required, "");
    const auto& enableTriangleRedistribution = parser.add<bool>(
            "redistribute-triangles", "Enables a remeshing step.", '\0', arrrgh::Optional, false);
    const auto& randomSeedParameter = parser.add<unsigned long>(
            "random-seed", "The random seed used to determine.", '\0', arrrgh::Required, 0);

    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message.", 'h', arrrgh::Optional, false);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    std::random_device rd("/dev/urandom");
    size_t randomSeed = randomSeedParameter.value() != 0 ? randomSeedParameter.value() : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);

    GLFWwindow* window = GLinitialise();

    std::vector<ShapeDescriptor::cpu::float3> screenQuadVertices = {{0, 0, 0},
                                                                    {1, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 1, 0}};
    std::vector<ShapeDescriptor::cpu::float3> screenQuadTexCoords ={{0, 0, 0},
                                                                    {1, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 1, 0}};
    std::vector<ShapeDescriptor::cpu::float3> screenQuadColours =  {{1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1}};
    BufferObject screenQuadVAO = generateVertexArray(screenQuadVertices.data(), screenQuadTexCoords.data(), screenQuadColours.data(), 6);

    Shader objectIDShader;
    objectIDShader.makeBasicShader("../res/shaders/objectIDShader.vert", "../res/shaders/objectIDShader.frag");
    Shader fullscreenQuadShader;
    fullscreenQuadShader.makeBasicShader("../res/shaders/fullscreenquad.vert", "../res/shaders/fullscreenquad.frag");

    // Create offscreen renderer

    const unsigned int offscreenTextureWidth = 4 * 7680;
    const unsigned int offscreenTextureHeight = 4 * 4230;

    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, offscreenTextureWidth, offscreenTextureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, offscreenTextureWidth, offscreenTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    std::vector<unsigned char> localFramebufferCopy(3 * offscreenTextureWidth * offscreenTextureHeight);

    std::cout << "Generating queries from objects in " << objectDirectory.value() << "..." << std::endl;

    std::vector<std::experimental::filesystem::path> haystackFiles = ShapeDescriptor::utilities::listDirectory(objectDirectory.value());

    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        std::cout << "Processing " << (i + 1) << "/" << haystackFiles.size() << ": " << haystackFiles.at(i) << std::endl;

        // Handle other events
        glfwPollEvents();

        // Flip buffers
        glfwSwapBuffers(window);

        int windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);

        ShapeDescriptor::cpu::Mesh loadedMesh = ShapeDescriptor::utilities::loadMesh(haystackFiles.at(i), true);

        ShapeDescriptor::cpu::float3 averageSum = {0, 0, 0};
        for(unsigned int vertex = 0; vertex < loadedMesh.vertexCount; vertex++) {
            averageSum += loadedMesh.vertices[vertex];
        }
        averageSum.x /= float(loadedMesh.vertexCount);
        averageSum.y /= float(loadedMesh.vertexCount);
        averageSum.z /= float(loadedMesh.vertexCount);

        std::vector<ShapeDescriptor::cpu::float3> vertexColours(loadedMesh.vertexCount);
        for(unsigned int triangle = 0; triangle < loadedMesh.vertexCount / 3; triangle++) {
            float red = float((triangle & 0x00FF0000U) >> 16U) / 255.0f;
            float green = float((triangle & 0x0000FF00U) >> 8U) / 255.0f;
            float blue = float((triangle & 0x000000FFU) >> 0U) / 255.0f;

            vertexColours.at(3 * triangle + 0) = {red, green, blue};
            vertexColours.at(3 * triangle + 1) = {red, green, blue};
            vertexColours.at(3 * triangle + 2) = {red, green, blue};
        }

        BufferObject buffers = generateVertexArray(loadedMesh.vertices, loadedMesh.normals, vertexColours.data(), loadedMesh.vertexCount);

        objectIDShader.activate();

        float yaw = float(distribution(generator) * 2.0 * M_PI);
        float pitch = float((distribution(generator) - 0.5) * M_PI);
        float roll = float(distribution(generator) * 2.0 * M_PI);

        glm::mat4 objectProjection = glm::perspective(1.57f, (float) windowWidth / (float) windowHeight, 1.0f, 10000.0f);
        glm::mat4 positionTransformation = glm::translate(glm::vec3(0, 0, -200.0f));
        positionTransformation *= glm::rotate(roll, glm::vec3(0, 0, 1));
        positionTransformation *= glm::rotate(yaw, glm::vec3(1, 0, 0));
        positionTransformation *= glm::rotate(pitch, glm::vec3(0, 1, 0));
        positionTransformation *= glm::translate(-glm::vec3(averageSum.x, averageSum.y, averageSum.z));
        glm::mat4 objectTransformation = objectProjection * positionTransformation;
        glUniformMatrix4fv(16, 1, false, glm::value_ptr(objectTransformation));

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(buffers.VAOID);
        glEnable(GL_DEPTH_TEST);
        glDrawElements(GL_TRIANGLES, loadedMesh.vertexCount, GL_UNSIGNED_INT, nullptr);

        // Do visibility testing

        glBindTexture(GL_TEXTURE_2D, texture);
        glReadPixels(0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_RGB, GL_UNSIGNED_BYTE, localFramebufferCopy.data());

        /*unsigned int error = lodepng::encode("temp.png", localFramebufferCopy, offscreenTextureWidth, offscreenTextureHeight, LCT_RGB, 8);

        if(error)
        {
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        }*/

        std::vector<bool> triangleAppearsInImage(loadedMesh.vertexCount / 3);

        for(size_t pixel = 0; pixel < offscreenTextureWidth * offscreenTextureHeight; pixel++) {
            unsigned int triangleIndex =
                    (((unsigned int) localFramebufferCopy.at(3 * pixel + 0)) << 16U) |
                    (((unsigned int) localFramebufferCopy.at(3 * pixel + 1)) << 8U) |
                    (((unsigned int) localFramebufferCopy.at(3 * pixel + 2)) << 0U);

            // Test if pixel is background
            if(triangleIndex == 0x00FFFFFF) {
                continue;
            }

            triangleAppearsInImage.at(triangleIndex) = true;
        }

        ShapeDescriptor::cpu::Mesh outMesh(loadedMesh.vertexCount);

        unsigned int visibleVertexCount = 0;
        for(unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++) {
            if(triangleAppearsInImage.at(triangle)) {
                outMesh.vertices[visibleVertexCount + 0] = loadedMesh.vertices[3 * triangle + 0];
                outMesh.vertices[visibleVertexCount + 1] = loadedMesh.vertices[3 * triangle + 1];
                outMesh.vertices[visibleVertexCount + 2] = loadedMesh.vertices[3 * triangle + 2];

                outMesh.normals[visibleVertexCount + 0] = loadedMesh.normals[3 * triangle + 0];
                outMesh.normals[visibleVertexCount + 1] = loadedMesh.normals[3 * triangle + 1];
                outMesh.normals[visibleVertexCount + 2] = loadedMesh.normals[3 * triangle + 2];

                visibleVertexCount += 3;
            }
        }

        outMesh.vertexCount = visibleVertexCount;

        // DEBUGGING

        ShapeDescriptor::cpu::Mesh modifiedQueryMesh = loadedMesh.clone();
        for(unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++) {
            if(triangleAppearsInImage.at(triangle)) {
                modifiedQueryMesh.vertices[visibleVertexCount + 0] = {5000, 5000, 5000};
                modifiedQueryMesh.vertices[visibleVertexCount + 1] = {5000, 5000, 5000};
                modifiedQueryMesh.vertices[visibleVertexCount + 2] = {5000, 5000, 5000};
            }
        }
        ShapeDescriptor::gpu::Mesh gpuModifiedQueryMesh = ShapeDescriptor::copy::hostMeshToDevice(modifiedQueryMesh);
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins = ShapeDescriptor::utilities::generateSpinOriginBuffer(
                gpuModifiedQueryMesh);

        ShapeDescriptor::gpu::Mesh gpuOutMesh = ShapeDescriptor::copy::hostMeshToDevice(outMesh);
        ShapeDescriptor::gpu::Mesh gpuHaystackMesh = ShapeDescriptor::copy::hostMeshToDevice(loadedMesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> queryDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuOutMesh, descriptorOrigins, 100);
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedDescriptors = convertRICIToModifiedQUICCI(queryDescriptors);

        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors = ShapeDescriptor::gpu::generateQUICCImages(gpuHaystackMesh, descriptorOrigins, 100);

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> cpuQueryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(convertedDescriptors);
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> cpuHaystackDescriptors = ShapeDescriptor::copy::deviceArrayToHost(haystackDescriptors);

        cpuQueryDescriptors.length = 5000;
        cpuHaystackDescriptors.length = 5000;
        ShapeDescriptor::dump::descriptors(cpuQueryDescriptors, "query_filtered_descriptors.png", 50);
        ShapeDescriptor::dump::descriptors(cpuHaystackDescriptors, "haystack_filtered_descriptors.png", 50);


        // END DEBUGGING

        std::string filename = std::experimental::filesystem::path(haystackFiles.at(i)).filename();
        std::experimental::filesystem::path outputMeshFile = std::experimental::filesystem::path(targetDirectory.value()) / filename;
        ShapeDescriptor::dump::mesh(outMesh, outputMeshFile);

        if(enableTriangleRedistribution.value() || cpuQueryDescriptors.length > 9) {
            std::cout << "Remeshing.. " << outputMeshFile.string() << std::endl;
            pmp::SurfaceMesh mesh;
            mesh.read(outputMeshFile.string());
            pmp::SurfaceRemeshing remesher(mesh);

            // Using the same approach as PMP library's remeshing tool
            pmp::Scalar totalEdgeLength(0);
            for (const auto& edgeInMesh : mesh.edges()) {
                totalEdgeLength += distance(mesh.position(mesh.vertex(edgeInMesh, 0)),
                                            mesh.position(mesh.vertex(edgeInMesh, 1)));
            }
            pmp::Scalar averageEdgeLength = totalEdgeLength / (pmp::Scalar) mesh.n_edges();

            remesher.uniform_remeshing(averageEdgeLength);
            mesh.write(outputMeshFile.string());
        }

        ShapeDescriptor::cpu::Mesh remeshedMesh = ShapeDescriptor::utilities::loadMesh(outputMeshFile.string());
        ShapeDescriptor::gpu::Mesh gpuRemeshedMesh = ShapeDescriptor::copy::hostMeshToDevice(remeshedMesh);

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> queryRemeshedDescriptors = ShapeDescriptor::gpu::generateRadialIntersectionCountImages(gpuRemeshedMesh, descriptorOrigins, 100);
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> convertedRemeshedDescriptors = convertRICIToModifiedQUICCI(queryRemeshedDescriptors);

        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> cpuRemeshedQueryDescriptors = ShapeDescriptor::copy::deviceArrayToHost(convertedRemeshedDescriptors);

        cpuRemeshedQueryDescriptors.length = 5000;
        ShapeDescriptor::dump::descriptors(cpuRemeshedQueryDescriptors, "query_remeshed_descriptors.png", 50);

        ShapeDescriptor::free::mesh(outMesh);

        // Draw visible version

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);
        glClearColor(0.5, 0.5, 0.5, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        fullscreenQuadShader.activate();

        glBindVertexArray(screenQuadVAO.VAOID);
        glDisable(GL_DEPTH_TEST);
        glBindTextureUnit(0, texture);

        glm::mat4 fullscreenProjection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

        glUniformMatrix4fv(16, 1, false, glm::value_ptr(fullscreenProjection));

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        ShapeDescriptor::free::mesh(loadedMesh);

        glDeleteBuffers(1, &buffers.vertexBufferID);
        glDeleteBuffers(1, &buffers.normalBufferID);
        glDeleteBuffers(1, &buffers.colourBufferID);
        glDeleteBuffers(1, &buffers.indexBufferID);
        glDeleteVertexArrays(1, &buffers.VAOID);

        //std::this_thread::sleep_for(100ms);
    }

    glDeleteFramebuffers(1, &fbo);

    std::cout << std::endl << "Done." << std::endl;
}